// I'll be Needin' Stitches (C++17 port of week02/stitching.py, features.py and mosaic.py).
//
// Turns a nadir drone video into one mosaic:
//   1. sample every Nth frame (optionally resized and undistorted),
//   2. track keyframes with grid bucketed ORB/SIFT features and RANSAC similarities,
//   3. verify loop closures between overlapping non consecutive keyframes,
//   4. bundle adjust every similarity with Levenberg Marquardt,
//   5. compensate exposure gains (Brown and Lowe),
//   6. blend with a Laplacian pyramid (multiband), feathering or overwrite.
// `--method stitcher` hands the sampled frames to cv::Stitcher in SCANS mode instead.

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
// OpenCV 5 moved estimateAffinePartial2D, intersectConvexConvex and friends out
// of calib3d/imgproc into the geometry module, which opencv.hpp does not
// include. OpenCV 4 has no such header, so the include is guarded.
#if CV_VERSION_MAJOR >= 5 && __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using Mat3 = cv::Matx33d;
using Complex = std::complex<double>;

struct StitchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    return buffer;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Python style modulo: always non negative for a positive divisor.
int pymod(int a, int m) { return ((a % m) + m) % m; }

cv::Point2d apply(const Mat3& m, const cv::Point2d& p) {
    return {m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2), m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2)};
}

std::vector<cv::Point2d> frame_corners(int w, int h) { return {{0, 0}, {double(w), 0}, {double(w), double(h)}, {0, double(h)}}; }

Mat3 translation(double tx, double ty) { return Mat3(1, 0, tx, 0, 1, ty, 0, 0, 1); }

// ================================================================= config ====

struct StitchConfig {
    int every = 5;
    double scale = 1.0;
    double k1 = 0.0;
    std::string detector = "orb";
    int n_features = 2500;
    double ratio = 0.8;
    double ransac_threshold = 3.0;
    int min_inliers = 30;
    double keyframe_motion = 0.18;  // fraction of the frame diagonal
    bool loop_closure = true;
    double loop_min_overlap = 0.25;
    int loop_max_per_frame = 4;
    bool bundle_adjust = true;
    bool gain_compensation = true;
    std::string blend = "multiband";
    double max_canvas_pixels = 150e6;
    bool verbose = false;
};

// =============================================================== features ====

struct Features {
    std::vector<cv::Point2f> points;
    cv::Mat descriptors;
    cv::Size size;
};

struct Alignment {
    Mat3 matrix;
    std::vector<cv::Point2f> src, dst;  // inlier correspondences
    int matches = 0;
    int inliers() const { return static_cast<int>(src.size()); }
};

// Keep the strongest keypoints in each grid cell so the transform is
// estimated from points spread over the whole frame, not one textured corner.
std::vector<cv::KeyPoint> bucket_keypoints(const std::vector<cv::KeyPoint>& keypoints, int width, int height, int cols,
                                           int rows, int limit) {
    if (keypoints.empty()) return {};
    size_t per_cell = static_cast<size_t>(std::max(1, limit / (cols * rows)));
    // Cells are visited in order of first appearance, like a Python dict.
    std::vector<std::vector<cv::KeyPoint>> cells;
    std::map<int, size_t> cell_index;
    for (const auto& kp : keypoints) {
        int cx = std::min(cols - 1, static_cast<int>(double(kp.pt.x) * cols / width));
        int cy = std::min(rows - 1, static_cast<int>(double(kp.pt.y) * rows / height));
        int key = cy * cols + cx;
        auto it = cell_index.find(key);
        if (it == cell_index.end()) {
            it = cell_index.emplace(key, cells.size()).first;
            cells.emplace_back();
        }
        cells[it->second].push_back(kp);
    }
    std::vector<cv::KeyPoint> kept;
    for (auto& bucket : cells) {
        std::stable_sort(bucket.begin(), bucket.end(),
                         [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
        kept.insert(kept.end(), bucket.begin(), bucket.begin() + std::min(per_cell, bucket.size()));
    }
    return kept;
}

class FeatureExtractor {
   public:
    FeatureExtractor(const std::string& kind, int n_features) : n_features_(n_features) {
        // CLAHE first so low contrast terrain (water, snow) still yields keypoints.
        clahe_ = cv::createCLAHE(2.5, cv::Size(8, 8));
        // Over detect, then bucket down to n_features.
        if (kind == "orb") {
            detector_ = cv::ORB::create(n_features * 3, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 7);
            norm = cv::NORM_HAMMING;
        } else if (kind == "sift") {
            detector_ = cv::SIFT::create(n_features * 2, 3, 0.02);
            norm = cv::NORM_L2;
        } else {
            throw std::invalid_argument("Unknown detector '" + kind + "'; use orb or sift");
        }
    }

    Features operator()(const cv::Mat& image) const {
        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image;
        clahe_->apply(gray, gray);
        std::vector<cv::KeyPoint> keypoints;
        detector_->detect(gray, keypoints);
        keypoints = bucket_keypoints(keypoints, gray.cols, gray.rows, 6, 4, n_features_);
        Features f;
        detector_->compute(gray, keypoints, f.descriptors);
        for (const auto& kp : keypoints) f.points.push_back(kp.pt);
        f.size = gray.size();
        return f;
    }

    int norm = cv::NORM_HAMMING;

   private:
    int n_features_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<cv::Feature2D> detector_;
};

class Matcher {
   public:
    Matcher(int norm, double ratio) : ratio_(ratio), matcher_(norm, false) {}

    // Symmetric matching: a pair survives only if it passes the ratio test in
    // both directions and each point picks the other.
    std::vector<std::pair<int, int>> operator()(const Features& src, const Features& dst) const {
        std::vector<std::pair<int, int>> pairs;
        if (src.descriptors.empty() || dst.descriptors.empty() || src.points.size() < 2 || dst.points.size() < 2) return pairs;
        auto forward = ratio_matches(src.descriptors, dst.descriptors);
        auto backward = ratio_matches(dst.descriptors, src.descriptors);
        for (size_t i = 0; i < forward.size(); ++i) {
            int j = forward[i];
            if (j >= 0 && backward[j] == static_cast<int>(i)) pairs.emplace_back(static_cast<int>(i), j);
        }
        return pairs;
    }

   private:
    std::vector<int> ratio_matches(const cv::Mat& a, const cv::Mat& b) const {
        std::vector<std::vector<cv::DMatch>> knn;
        matcher_.knnMatch(a, b, knn, 2);
        std::vector<int> good(a.rows, -1);
        for (const auto& pair : knn) {
            if (pair.size() == 2 && double(pair[0].distance) < ratio_ * double(pair[1].distance))
                good[pair[0].queryIdx] = pair[0].trainIdx;
            else if (pair.size() == 1)
                good[pair[0].queryIdx] = pair[0].trainIdx;
        }
        return good;
    }

    double ratio_;
    mutable cv::BFMatcher matcher_;
};

// RANSAC 4 DOF similarity (right model for a nadir camera at near constant
// altitude) with Levenberg Marquardt refinement, then sanity checks.
std::optional<Alignment> align(const Features& src, const Features& dst, const Matcher& matcher, double threshold,
                               int min_inliers, std::string& reason, double min_inlier_ratio = 0.2,
                               double max_scale_change = 0.3) {
    auto pairs = matcher(src, dst);
    int n = static_cast<int>(pairs.size());
    if (n < min_inliers) {
        reason = format("%d symmetric matches", n);
        return std::nullopt;
    }
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto& [i, j] : pairs) {
        src_pts.push_back(src.points[i]);
        dst_pts.push_back(dst.points[j]);
    }
    cv::Mat inliers;
    cv::Mat affine = n < 3 ? cv::Mat()
                           : cv::estimateAffinePartial2D(src_pts, dst_pts, inliers, cv::RANSAC, threshold, 5000, 0.999, 20);
    if (affine.empty()) {
        reason = "RANSAC failed";
        return std::nullopt;
    }
    int count = cv::countNonZero(inliers);
    if (count < min_inliers || count < min_inlier_ratio * n) {
        reason = format("%d/%d inliers", count, n);
        return std::nullopt;
    }
    Alignment a;
    a.matrix = Mat3(affine.at<double>(0, 0), affine.at<double>(0, 1), affine.at<double>(0, 2), affine.at<double>(1, 0),
                    affine.at<double>(1, 1), affine.at<double>(1, 2), 0, 0, 1);
    double scale = std::sqrt(std::abs(a.matrix(0, 0) * a.matrix(1, 1) - a.matrix(0, 1) * a.matrix(1, 0)));
    if (std::abs(scale - 1) > max_scale_change) {
        reason = format("implausible scale %.2f", scale);
        return std::nullopt;
    }
    const uchar* mask = inliers.ptr<uchar>(0);
    for (int k = 0; k < n; ++k) {
        if (mask[k]) {
            a.src.push_back(src_pts[k]);
            a.dst.push_back(dst_pts[k]);
        }
    }
    a.matches = n;
    return a;
}

// ======================================================= bundle adjustment ====

struct Constraint {
    int i, j;
    std::vector<cv::Point2d> src;  // points in frame i
    std::vector<cv::Point2d> dst;  // matching points in frame j
    bool loop_closure = false;
};

struct AdjustmentReport {
    double rms_before = 0, rms_after = 0;
    int constraints = 0, loop_closures = 0, correspondences = 0;
};

// Similarity as (a, b, tx, ty), projected onto the similarity group.
using Params = std::vector<double>;  // 4 per frame

Params to_params(const std::vector<Mat3>& transforms) {
    Params p;
    for (const auto& m : transforms) {
        p.push_back((m(0, 0) + m(1, 1)) / 2);
        p.push_back((m(1, 0) - m(0, 1)) / 2);
        p.push_back(m(0, 2));
        p.push_back(m(1, 2));
    }
    return p;
}

std::vector<Mat3> from_params(const Params& p) {
    std::vector<Mat3> out;
    for (size_t k = 0; k + 3 < p.size(); k += 4) out.emplace_back(p[k], -p[k + 1], p[k + 2], p[k + 1], p[k], p[k + 3], 0, 0, 1);
    return out;
}

// A similarity is z -> alpha z + beta in the complex plane.
Complex alpha_of(const Params& p, int frame) { return {p[4 * frame], p[4 * frame + 1]}; }
Complex beta_of(const Params& p, int frame) { return {p[4 * frame + 2], p[4 * frame + 3]}; }

// Residuals are measured in frame pixels (both directions) rather than in
// mosaic pixels: the mosaic space version is minimized by shrinking every
// frame, which collapses the scale.
double constraint_rms(const Params& p, const std::vector<Constraint>& constraints) {
    double sum = 0;
    size_t count = 0;
    for (const auto& c : constraints) {
        Complex ai = alpha_of(p, c.i), bi = beta_of(p, c.i), aj = alpha_of(p, c.j), bj = beta_of(p, c.j);
        for (size_t k = 0; k < c.src.size(); ++k) {
            Complex z(c.src[k].x, c.src[k].y), q(c.dst[k].x, c.dst[k].y);
            sum += std::norm((ai * z + bi - bj) / aj - q) + std::norm((aj * q + bj - bi) / ai - z);
            count += 2;
        }
    }
    return count ? std::sqrt(sum / count) : 0.0;
}

// Accumulates J^T W J, J^T W r and the weighted cost. With `cost_only` the
// Jacobian is skipped (used to test Levenberg Marquardt candidate steps).
double normal_equations(const Params& p, const std::vector<Constraint>& constraints,
                        const std::vector<std::vector<double>>& robust, int n, cv::Mat* H, cv::Mat* g) {
    if (H) *H = cv::Mat::zeros(4 * n, 4 * n, CV_64F);
    if (g) *g = cv::Mat::zeros(4 * n, 1, CV_64F);
    double cost = 0;
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
        const auto& c = constraints[ci];
        const auto& w = robust[ci];
        for (int direction = 0; direction < 2; ++direction) {
            // Forward: frame i points into frame j pixels; backward: the mirror.
            int ia = direction ? c.j : c.i, ib = direction ? c.i : c.j;
            const auto& xs = direction ? c.dst : c.src;
            const auto& ys = direction ? c.src : c.dst;
            Complex alpha_a = alpha_of(p, ia), beta_a = beta_of(p, ia);
            Complex alpha_b = alpha_of(p, ib), beta_b = beta_of(p, ib);
            int idx[8] = {4 * ia, 4 * ia + 1, 4 * ia + 2, 4 * ia + 3, 4 * ib, 4 * ib + 1, 4 * ib + 2, 4 * ib + 3};
            for (size_t k = 0; k < xs.size(); ++k) {
                Complex x(xs[k].x, xs[k].y), y(ys[k].x, ys[k].y);
                Complex mapped = alpha_a * x + beta_a - beta_b;
                Complex r = mapped / alpha_b - y;
                cost += w[k] * std::norm(r);
                if (!H) continue;
                // The residual is holomorphic, so each complex derivative d
                // gives real Jacobian columns [Re d, -Im d] and [Im d, Re d].
                Complex d[4] = {x / alpha_b, 1.0 / alpha_b, -mapped / (alpha_b * alpha_b), -1.0 / alpha_b};
                double jr[8], ji[8];
                for (int t = 0; t < 4; ++t) {
                    jr[2 * t] = d[t].real();
                    jr[2 * t + 1] = -d[t].imag();
                    ji[2 * t] = d[t].imag();
                    ji[2 * t + 1] = d[t].real();
                }
                for (int u = 0; u < 8; ++u) {
                    double* row = H->ptr<double>(idx[u]);
                    for (int v = 0; v < 8; ++v) row[idx[v]] += w[k] * (jr[u] * jr[v] + ji[u] * ji[v]);
                    g->at<double>(idx[u]) += w[k] * (jr[u] * r.real() + ji[u] * r.imag());
                }
            }
        }
    }
    return cost;
}

std::vector<Mat3> bundle_adjust(const std::vector<Mat3>& transforms, const std::vector<Constraint>& constraints,
                                AdjustmentReport& report, double huber, int fixed = 0, int iterations = 15,
                                size_t max_points = 80, uint64_t seed = 0) {
    const int n = static_cast<int>(transforms.size());
    // Random subset per constraint keeps the problem small. mt19937_64 is
    // fully specified, so the subset is identical on every platform (it does
    // not reproduce NumPy's generator).
    std::mt19937_64 rng(seed);
    std::vector<Constraint> sampled;
    for (const auto& c : constraints) {
        if (c.src.size() <= max_points) {
            sampled.push_back(c);
            continue;
        }
        std::vector<size_t> order(c.src.size());
        for (size_t k = 0; k < order.size(); ++k) order[k] = k;
        for (size_t k = 0; k < max_points; ++k) std::swap(order[k], order[k + rng() % (order.size() - k)]);
        Constraint s{c.i, c.j, {}, {}, c.loop_closure};
        for (size_t k = 0; k < max_points; ++k) {
            s.src.push_back(c.src[order[k]]);
            s.dst.push_back(c.dst[order[k]]);
        }
        sampled.push_back(std::move(s));
    }

    Params params = to_params(transforms);
    report = AdjustmentReport{};
    report.rms_before = constraint_rms(params, sampled);
    report.constraints = static_cast<int>(sampled.size());
    for (const auto& c : sampled) {
        report.loop_closures += c.loop_closure;
        report.correspondences += static_cast<int>(c.src.size());
    }
    if (n < 2 || sampled.empty()) {
        report.rms_after = report.rms_before;
        return from_params(params);
    }

    std::vector<std::vector<double>> robust;
    for (const auto& c : sampled) robust.emplace_back(c.src.size(), 1.0);
    std::vector<int> free_idx;
    for (int k = 0; k < 4 * n; ++k)
        if (k < 4 * fixed || k >= 4 * fixed + 4) free_idx.push_back(k);
    const int nf = static_cast<int>(free_idx.size());

    double damping = 1e-3;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        cv::Mat H, g;
        double cost = normal_equations(params, sampled, robust, n, &H, &g);
        cv::Mat H_ff(nf, nf, CV_64F), g_f(nf, 1, CV_64F);
        for (int a = 0; a < nf; ++a) {
            g_f.at<double>(a) = -g.at<double>(free_idx[a]);
            for (int b = 0; b < nf; ++b) H_ff.at<double>(a, b) = H.at<double>(free_idx[a], free_idx[b]);
        }
        for (int attempt = 0; attempt < 8; ++attempt) {
            cv::Mat A = H_ff.clone();
            for (int a = 0; a < nf; ++a) A.at<double>(a, a) += damping * (H_ff.at<double>(a, a) + 1e-9);
            cv::Mat step;
            // The damped system is symmetric positive definite, so Cholesky is
            // the cheap path; LU is the fallback for numerically awkward cases.
            if (!cv::solve(A, g_f, step, cv::DECOMP_CHOLESKY) && !cv::solve(A, g_f, step, cv::DECOMP_LU)) {
                damping *= 10;
                continue;
            }
            Params candidate = params;
            for (int a = 0; a < nf; ++a) candidate[free_idx[a]] += step.at<double>(a);
            double new_cost = normal_equations(candidate, sampled, robust, n, nullptr, nullptr);
            if (new_cost < cost) {
                params = std::move(candidate);
                damping = std::max(damping / 10, 1e-7);
                break;
            }
            damping *= 10;
        }
        // Huber reweighting from the symmetric residual in frame pixels.
        for (size_t ci = 0; ci < sampled.size(); ++ci) {
            const auto& c = sampled[ci];
            Complex ai = alpha_of(params, c.i), bi = beta_of(params, c.i), aj = alpha_of(params, c.j), bj = beta_of(params, c.j);
            for (size_t k = 0; k < c.src.size(); ++k) {
                Complex z(c.src[k].x, c.src[k].y), q(c.dst[k].x, c.dst[k].y);
                double error =
                    std::sqrt((std::norm((ai * z + bi - bj) / aj - q) + std::norm((aj * q + bj - bi) / ai - z)) / 2);
                robust[ci][k] = error <= huber ? 1.0 : huber / std::max(error, 1e-12);
            }
        }
    }
    auto adjusted = from_params(params);
    report.rms_after = constraint_rms(to_params(adjusted), sampled);
    return adjusted;
}

// ============================================================ canvas math ====

struct Canvas {
    int width = 0, height = 0;
    Mat3 offset = Mat3::eye();  // mosaic coordinates to canvas pixels
};

struct Roi {
    int x0, y0, x1, y1;
    cv::Rect rect() const { return {x0, y0, x1 - x0, y1 - y0}; }
};

Canvas plan_canvas(const std::vector<cv::Size>& sizes, const std::vector<Mat3>& transforms, double max_pixels,
                   int pad_multiple = 1) {
    double min_x = INFINITY, min_y = INFINITY, max_x = -INFINITY, max_y = -INFINITY;
    for (size_t k = 0; k < sizes.size(); ++k) {
        for (const auto& c : frame_corners(sizes[k].width, sizes[k].height)) {
            cv::Point2d p = apply(transforms[k], c);
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
    }
    double x0 = std::floor(min_x), y0 = std::floor(min_y);
    Canvas canvas;
    canvas.width = static_cast<int>(std::ceil(max_x) - x0);
    canvas.height = static_cast<int>(std::ceil(max_y) - y0);
    canvas.width += pymod(-canvas.width, pad_multiple);
    canvas.height += pymod(-canvas.height, pad_multiple);
    if (double(canvas.width) * canvas.height > max_pixels)
        throw StitchError(format("Canvas would be %dx%d px; reduce --scale or raise the limit. "
                                 "A huge canvas usually means alignment drifted.",
                                 canvas.width, canvas.height));
    canvas.offset = translation(-x0, -y0);
    return canvas;
}

Roi frame_roi(const Mat3& m, cv::Size size, const Canvas& canvas, int margin = 0, int multiple = 1) {
    double min_x = INFINITY, min_y = INFINITY, max_x = -INFINITY, max_y = -INFINITY;
    for (const auto& c : frame_corners(size.width, size.height)) {
        cv::Point2d p = apply(m, c);
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    int x0 = static_cast<int>(std::floor(min_x)) - margin, y0 = static_cast<int>(std::floor(min_y)) - margin;
    int x1 = static_cast<int>(std::ceil(max_x)) + margin, y1 = static_cast<int>(std::ceil(max_y)) + margin;
    x0 -= pymod(x0, multiple);
    y0 -= pymod(y0, multiple);
    x1 += pymod(-x1, multiple);
    y1 += pymod(-y1, multiple);
    return {std::max(0, x0), std::max(0, y0), std::min(canvas.width, x1), std::min(canvas.height, y1)};
}

bool roi_empty(const Roi& r) { return r.x1 <= r.x0 || r.y1 <= r.y0; }

cv::Mat warp(const cv::Mat& image, const Mat3& m, const Roi& roi, int border = cv::BORDER_CONSTANT) {
    Mat3 local = translation(-roi.x0, -roi.y0) * m;
    cv::Matx23d affine(local(0, 0), local(0, 1), local(0, 2), local(1, 0), local(1, 1), local(1, 2));
    cv::Mat out;
    cv::warpAffine(image, out, affine, cv::Size(roi.x1 - roi.x0, roi.y1 - roi.y0), cv::INTER_LINEAR, border);
    return out;
}

// 1 at the frame center falling to 0 at the border.
cv::Mat feather_weight(cv::Size size) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    if (size.width > 2 && size.height > 2) mask(cv::Rect(1, 1, size.width - 2, size.height - 2)).setTo(255);
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 3);
    double max_value;
    cv::minMaxLoc(dist, nullptr, &max_value);
    return dist / std::max(max_value, 1.0);
}

double footprint_overlap(const Mat3& m1, const Mat3& m2, int w, int h) {
    std::vector<cv::Point2f> p1, p2;
    for (const auto& c : frame_corners(w, h)) {
        p1.emplace_back(apply(m1, c));
        p2.emplace_back(apply(m2, c));
    }
    std::vector<cv::Point2f> intersection;
    double area = cv::intersectConvexConvex(p1, p2, intersection, true);
    return area / (double(w) * h * std::abs(m1(0, 0) * m1(1, 1) - m1(0, 1) * m1(1, 0)));
}

// ====================================================== gain compensation ====

// One gain per frame from the mean intensities of every overlapping pair
// (Brown and Lowe 2007), with a prior that keeps gains near 1.
std::vector<double> gain_compensation(const std::vector<cv::Mat>& frames, const std::vector<Mat3>& transforms,
                                      double sigma_n = 10.0, double sigma_g = 0.1, double work_pixels = 1e6) {
    const int n = static_cast<int>(frames.size());
    if (n < 2) return std::vector<double>(n, 1.0);
    std::vector<cv::Size> sizes;
    for (const auto& f : frames) sizes.push_back(f.size());
    Canvas full = plan_canvas(sizes, transforms, 1e12);
    // Work on a downscaled canvas: means over a million pixels are plenty.
    double scale = std::min(1.0, std::sqrt(work_pixels / (double(full.width) * full.height)));
    Mat3 shrink(scale, 0, 0, 0, scale, 0, 0, 0, 1);
    Canvas canvas;
    canvas.width = static_cast<int>(std::ceil(full.width * scale)) + 1;
    canvas.height = static_cast<int>(std::ceil(full.height * scale)) + 1;

    struct Warped {
        Roi roi;
        cv::Mat gray, valid;
    };
    std::vector<Warped> warped;
    for (int k = 0; k < n; ++k) {
        Mat3 m = shrink * full.offset * transforms[k];
        Roi roi = frame_roi(m, sizes[k], canvas);
        cv::Mat gray, gray32;
        cv::cvtColor(frames[k], gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray32, CV_32F);
        cv::Mat ones = cv::Mat::ones(gray.size(), CV_32F);
        warped.push_back({roi, warp(gray32, m, roi), warp(ones, m, roi) > 0.99});
    }

    cv::Mat counts = cv::Mat::zeros(n, n, CV_64F), means = cv::Mat::zeros(n, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        const auto& a = warped[i];
        for (int j = i + 1; j < n; ++j) {
            const auto& b = warped[j];
            int x0 = std::max(a.roi.x0, b.roi.x0), y0 = std::max(a.roi.y0, b.roi.y0);
            int x1 = std::min(a.roi.x1, b.roi.x1), y1 = std::min(a.roi.y1, b.roi.y1);
            if (x1 <= x0 || y1 <= y0) continue;
            cv::Rect si(x0 - a.roi.x0, y0 - a.roi.y0, x1 - x0, y1 - y0);
            cv::Rect sj(x0 - b.roi.x0, y0 - b.roi.y0, x1 - x0, y1 - y0);
            cv::Mat both = a.valid(si) & b.valid(sj);
            int count = cv::countNonZero(both);
            if (count < 50) continue;
            counts.at<double>(i, j) = counts.at<double>(j, i) = count;
            means.at<double>(i, j) = cv::mean(a.gray(si), both)[0];
            means.at<double>(j, i) = cv::mean(b.gray(sj), both)[0];
        }
    }

    cv::Mat A = cv::Mat::zeros(n, n, CV_64F), rhs = cv::Mat::zeros(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double c = counts.at<double>(i, j);
            if (i == j || c == 0) continue;
            double mij = means.at<double>(i, j), mji = means.at<double>(j, i);
            A.at<double>(i, i) += c * (2 * mij * mij / (sigma_n * sigma_n) + 1 / (sigma_g * sigma_g));
            A.at<double>(i, j) -= 2 * c * mij * mji / (sigma_n * sigma_n);
            rhs.at<double>(i) += c / (sigma_g * sigma_g);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (A.at<double>(i, i) == 0) {  // frame overlaps nothing: keep its gain at 1
            A.at<double>(i, i) = 1.0;
            rhs.at<double>(i) = 1.0;
        }
    }
    cv::Mat solution;
    if (!cv::solve(A, rhs, solution, cv::DECOMP_LU)) cv::solve(A, rhs, solution, cv::DECOMP_SVD);
    return std::vector<double>(solution.begin<double>(), solution.end<double>());
}

// ================================================================ blending ====

using BlendResult = std::pair<cv::Mat, cv::Mat>;  // (BGR mosaic, 8 bit coverage mask)

// Each canvas pixel belongs to the frame whose warped center weight is highest.
std::pair<cv::Mat, cv::Mat> seam_labels(const std::vector<cv::Size>& sizes, const std::vector<Mat3>& transforms,
                                        const Canvas& canvas) {
    cv::Mat best = cv::Mat::zeros(canvas.height, canvas.width, CV_32F);
    cv::Mat labels(canvas.height, canvas.width, CV_32S, cv::Scalar(-1));
    std::map<std::pair<int, int>, cv::Mat> cache;
    for (size_t index = 0; index < sizes.size(); ++index) {
        auto key = std::make_pair(sizes[index].width, sizes[index].height);
        if (!cache.count(key)) cache[key] = feather_weight(sizes[index]) + 1e-3;
        Roi roi = frame_roi(transforms[index], sizes[index], canvas);
        if (roi_empty(roi)) continue;
        cv::Mat weight = warp(cache[key], transforms[index], roi);
        cv::Mat region = best(roi.rect());
        cv::Mat better = weight > region;
        weight.copyTo(region, better);
        labels(roi.rect()).setTo(static_cast<int>(index), better);
    }
    return {labels, best > 0};
}

BlendResult overwrite_blend(const std::vector<cv::Mat>& frames, const std::vector<Mat3>& transforms, const Canvas& canvas) {
    cv::Mat out = cv::Mat::zeros(canvas.height, canvas.width, CV_8UC3);
    cv::Mat coverage = cv::Mat::zeros(canvas.height, canvas.width, CV_8U);
    for (size_t k = 0; k < frames.size(); ++k) {
        Roi roi = frame_roi(transforms[k], frames[k].size(), canvas);
        if (roi_empty(roi)) continue;
        cv::Mat warped = warp(frames[k], transforms[k], roi, cv::BORDER_REPLICATE);
        cv::Mat valid = warp(cv::Mat::ones(frames[k].size(), CV_32F), transforms[k], roi) > 0.5;
        warped.copyTo(out(roi.rect()), valid);
        coverage(roi.rect()).setTo(255, valid);
    }
    return {out, coverage};
}

BlendResult feather_blend(const std::vector<cv::Mat>& frames, const std::vector<Mat3>& transforms, const Canvas& canvas) {
    cv::Mat accum = cv::Mat::zeros(canvas.height, canvas.width, CV_32FC3);
    cv::Mat total = cv::Mat::zeros(canvas.height, canvas.width, CV_32F);
    std::map<std::pair<int, int>, cv::Mat> cache;
    for (size_t k = 0; k < frames.size(); ++k) {
        auto key = std::make_pair(frames[k].cols, frames[k].rows);
        if (!cache.count(key)) cache[key] = feather_weight(frames[k].size());
        Roi roi = frame_roi(transforms[k], frames[k].size(), canvas);
        if (roi_empty(roi)) continue;
        cv::Mat weight = warp(cache[key], transforms[k], roi);
        cv::Mat warped, weight3;
        warp(frames[k], transforms[k], roi, cv::BORDER_REPLICATE).convertTo(warped, CV_32F);
        cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight3);
        cv::Mat accum_roi = accum(roi.rect()), total_roi = total(roi.rect());
        accum_roi += warped.mul(weight3);
        total_roi += weight;
    }
    cv::Mat coverage = total > 1e-4;
    cv::Mat safe, safe3, out32, out;
    cv::max(total, 1e-12, safe);
    cv::merge(std::vector<cv::Mat>{safe, safe, safe}, safe3);
    cv::divide(accum, safe3, out32);
    out32.convertTo(out, CV_8U);
    out.setTo(0, coverage == 0);
    return {out, coverage};
}

// Burt and Adelson: Laplacian pyramids of each frame mixed with Gaussian
// pyramids of the seam masks, so low frequencies blend over a wide region
// and fine detail over a narrow one.
BlendResult multiband_blend(const std::vector<cv::Mat>& frames, const std::vector<Mat3>& transforms, const Canvas& canvas,
                            int levels = 5) {
    const int multiple = 1 << levels;
    std::vector<cv::Size> sizes;
    for (const auto& f : frames) sizes.push_back(f.size());
    auto [labels, coverage] = seam_labels(sizes, transforms, canvas);

    std::vector<cv::Size> shapes{cv::Size(canvas.width, canvas.height)};
    for (int l = 0; l < levels; ++l) shapes.emplace_back((shapes.back().width + 1) / 2, (shapes.back().height + 1) / 2);
    std::vector<cv::Mat> bands, weights;
    for (const auto& s : shapes) {
        bands.push_back(cv::Mat::zeros(s, CV_32FC3));
        weights.push_back(cv::Mat::zeros(s, CV_32F));
    }

    for (size_t index = 0; index < frames.size(); ++index) {
        Roi roi = frame_roi(transforms[index], sizes[index], canvas, multiple * 2, multiple);
        if ((roi.x1 - roi.x0) % multiple || (roi.y1 - roi.y0) % multiple) {
            // Clipped by the canvas edge; the canvas itself is padded to a multiple.
            roi.x1 = std::min(canvas.width, roi.x0 + ((roi.x1 - roi.x0 + multiple - 1) / multiple) * multiple);
            roi.y1 = std::min(canvas.height, roi.y0 + ((roi.y1 - roi.y0 + multiple - 1) / multiple) * multiple);
        }
        if (roi_empty(roi)) continue;
        cv::Mat mask8 = labels(roi.rect()) == static_cast<int>(index);
        if (cv::countNonZero(mask8) == 0) continue;
        cv::Mat mask, image;
        mask8.convertTo(mask, CV_32F, 1.0 / 255);
        warp(frames[index], transforms[index], roi, cv::BORDER_REPLICATE).convertTo(image, CV_32F);

        std::vector<cv::Mat> gaussian{image}, mask_pyramid{mask};
        for (int l = 0; l < levels; ++l) {
            cv::Mat g, m;
            cv::pyrDown(gaussian.back(), g);
            cv::pyrDown(mask_pyramid.back(), m);
            gaussian.push_back(g);
            mask_pyramid.push_back(m);
        }
        for (int level = 0; level <= levels; ++level) {
            cv::Mat band;
            if (level < levels) {
                cv::Mat up;
                cv::pyrUp(gaussian[level + 1], up, gaussian[level].size());
                band = gaussian[level] - up;
            } else {
                band = gaussian[level];
            }
            const cv::Mat& m = mask_pyramid[level];
            int lx = roi.x0 >> level, ly = roi.y0 >> level;
            int w = std::min(band.cols, shapes[level].width - lx);
            int h = std::min(band.rows, shapes[level].height - ly);
            if (w <= 0 || h <= 0) continue;
            cv::Rect src(0, 0, w, h), dst(lx, ly, w, h);
            cv::Mat m3;
            cv::merge(std::vector<cv::Mat>{m(src), m(src), m(src)}, m3);
            cv::Mat band_dst = bands[level](dst), weight_dst = weights[level](dst);
            band_dst += band(src).mul(m3);
            weight_dst += m(src);
        }
    }

    cv::Mat result;
    for (int level = levels; level >= 0; --level) {
        cv::Mat safe, safe3, normalized;
        cv::max(weights[level], 1e-6, safe);
        cv::merge(std::vector<cv::Mat>{safe, safe, safe}, safe3);
        cv::divide(bands[level], safe3, normalized);
        if (result.empty()) {
            result = normalized;
        } else {
            cv::Mat up;
            cv::pyrUp(result, up, shapes[level]);
            result = up + normalized;
        }
    }
    cv::Mat out;
    result.convertTo(out, CV_8U);  // saturates to 0..255
    out.setTo(0, coverage == 0);
    return {out, coverage};
}

// ================================================================ pipeline ====

struct StitchResult {
    cv::Mat panorama, coverage;
    std::vector<Mat3> transforms;  // keyframe to panorama pixels
    std::vector<int> frame_indices;
    int sampled = 0, rejected = 0, relocalized = 0, sequential_constraints = 0, loop_closures = 0;
    std::optional<AdjustmentReport> adjustment;
    std::vector<double> gains;
    std::vector<std::pair<std::string, double>> timings;

    std::string summary() const {
        double covered = 100.0 * cv::countNonZero(coverage) / double(coverage.total());
        std::string s = format("panorama        %d x %d px, %.0f%% covered\n", panorama.cols, panorama.rows, covered);
        s += format("frames          %d sampled, %zu keyframes, %d rejected, %d relocalized\n", sampled, frame_indices.size(),
                    rejected, relocalized);
        s += format("constraints     %d sequential + %d loop closures\n", sequential_constraints, loop_closures);
        if (adjustment)
            s += format("bundle adjust   residual RMS %.2f px -> %.2f px over %d correspondences\n", adjustment->rms_before,
                        adjustment->rms_after, adjustment->correspondences);
        if (!gains.empty())
            s += format("gains           %.3f .. %.3f\n", *std::min_element(gains.begin(), gains.end()),
                        *std::max_element(gains.begin(), gains.end()));
        double total = 0;
        std::string parts;
        for (const auto& [name, t] : timings) {
            parts += (parts.empty() ? "" : ", ") + format("%s %.2fs", name.c_str(), t);
            total += t;
        }
        return s + "timing          " + parts + format(" (total %.2fs)", total);
    }

    // Same format as StitchResult.save_transforms (np.savetxt).
    void save_transforms(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw StitchError("could not write " + path);
        out << "# frame,m00,m01,m02,m10,m11,m12\n";
        for (size_t k = 0; k < transforms.size(); ++k) {
            const Mat3& m = transforms[k];
            out << format("%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", frame_indices[k], m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
                          m(1, 2));
        }
    }
};

// Removes radial lens distortion with the simulator's camera model
// (f = half the larger side, principal point at the center), cached per size.
class Undistorter {
   public:
    explicit Undistorter(double k1) : k1_(k1) {}

    cv::Mat operator()(const cv::Mat& frame) {
        if (k1_ == 0.0) return frame;
        if (frame.size() != size_) {
            size_ = frame.size();
            double f = std::max(size_.width, size_.height) / 2.0;
            cv::Matx33d K(f, 0, size_.width / 2.0, 0, f, size_.height / 2.0, 0, 0, 1);
            cv::Matx41d dist(k1_, 0, 0, 0);
            cv::initUndistortRectifyMap(K, dist, cv::noArray(), K, size_, CV_32FC1, map_x_, map_y_);
        }
        cv::Mat out;
        cv::remap(frame, out, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return out;
    }

   private:
    double k1_;
    cv::Size size_;
    cv::Mat map_x_, map_y_;
};

// Streams (frame_index, frame) for every `every`-th frame.
class FrameSource {
   public:
    FrameSource(const std::string& path, int every, double scale, double k1) : every_(every), scale_(scale), undistort_(k1) {
        if (every < 1) throw std::invalid_argument("every must be at least 1");
        if (!fs::is_regular_file(path)) throw std::invalid_argument("No video found at " + path);
        if (!cap_.open(path)) throw StitchError("OpenCV could not open " + path + " as a video");
    }

    bool next(int& index, cv::Mat& frame) {
        cv::Mat raw;
        while (cap_.read(raw)) {
            int current = index_++;
            if (current % every_ != 0) continue;
            if (scale_ != 1.0) cv::resize(raw, raw, cv::Size(), scale_, scale_, cv::INTER_AREA);
            index = current;
            frame = undistort_(raw);
            return true;
        }
        return false;
    }

   private:
    cv::VideoCapture cap_;
    int every_;
    double scale_;
    Undistorter undistort_;
    int index_ = 0;
};

struct Keyframe {
    int index;
    cv::Mat frame;
    Features features;
    Mat3 transform;
};

std::vector<cv::Point2d> to_double(const std::vector<cv::Point2f>& pts) { return {pts.begin(), pts.end()}; }

class Tracker {
   public:
    explicit Tracker(const StitchConfig& config)
        : config_(config), extract_(config.detector, config.n_features), matcher_(extract_.norm, config.ratio) {}

    std::optional<Alignment> align_features(const Features& src, const Features& dst, std::string& reason) const {
        return align(src, dst, matcher_, config_.ransac_threshold, config_.min_inliers, reason);
    }

    void push(int index, const cv::Mat& frame) {
        Features features = extract_(frame);
        if (keyframes.empty()) {
            add_keyframe(index, frame, std::move(features), std::nullopt, 0);
            return;
        }
        int key = static_cast<int>(keyframes.size()) - 1;
        std::string reason;
        auto alignment = align_features(features, keyframes[key].features, reason);
        if (!alignment && candidate_) {
            // Motion outran the keyframe: promote the last good frame and retry.
            Candidate c = std::move(*candidate_);
            add_keyframe(c.index, c.frame, std::move(c.features), std::move(c.alignment), key);
            key = static_cast<int>(keyframes.size()) - 1;
            alignment = align_features(features, keyframes[key].features, reason);
        }
        if (!alignment) {
            // Lost track: try every earlier keyframe and keep the best.
            int best_key = -1;
            for (int k = static_cast<int>(keyframes.size()) - 1; k >= 0; --k) {
                std::string ignored;
                auto a = align_features(features, keyframes[k].features, ignored);
                if (a && (!alignment || a->inliers() > alignment->inliers())) {
                    alignment = std::move(a);
                    best_key = k;
                }
            }
            if (!alignment) {
                ++rejected;
                log(format("frame %d: rejected (%s)", index, reason.c_str()));
                return;
            }
            ++relocalized;
            log(format("frame %d: relocalized against keyframe %d", index, best_key));
            add_keyframe(index, frame, std::move(features), std::move(alignment), best_key);
            return;
        }

        if (motion(*alignment, features.size) >= config_.keyframe_motion) {
            int inliers = alignment->inliers();
            add_keyframe(index, frame, std::move(features), std::move(alignment), key);
            log(format("frame %d: keyframe %zu (%d inliers)", index, keyframes.size() - 1, inliers));
        } else {
            candidate_ = Candidate{index, frame, std::move(features), std::move(*alignment)};
        }
    }

    // Keep the final frame of the flight so the mosaic reaches the end.
    void finish() {
        if (candidate_) {
            Candidate c = std::move(*candidate_);
            add_keyframe(c.index, c.frame, std::move(c.features), std::move(c.alignment), static_cast<int>(keyframes.size()) - 1);
        }
    }

    void log(const std::string& message) const {
        if (config_.verbose) std::printf("%s\n", message.c_str());
    }

    std::vector<Keyframe> keyframes;
    std::vector<Constraint> constraints;
    int rejected = 0, relocalized = 0;

   private:
    struct Candidate {
        int index;
        cv::Mat frame;
        Features features;
        Alignment alignment;
    };

    void add_keyframe(int index, const cv::Mat& frame, Features features, std::optional<Alignment> alignment, int reference) {
        Mat3 transform = Mat3::eye();
        if (alignment) {
            transform = keyframes[reference].transform * alignment->matrix;
            constraints.push_back({static_cast<int>(keyframes.size()), reference, to_double(alignment->src),
                                   to_double(alignment->dst), false});
        }
        keyframes.push_back({index, frame, std::move(features), transform});
        candidate_.reset();
    }

    static double motion(const Alignment& a, cv::Size size) {
        cv::Point2d center(size.width / 2.0, size.height / 2.0);
        cv::Point2d moved = apply(a.matrix, center);
        return cv::norm(moved - center) / std::hypot(size.width, size.height);
    }

    const StitchConfig& config_;
    FeatureExtractor extract_;
    Matcher matcher_;
    std::optional<Candidate> candidate_;
};

// Verify predicted overlaps between non consecutive keyframes (the next
// lawnmower pass looking at the same ground).
std::vector<Constraint> find_loop_closures(const Tracker& tracker, const StitchConfig& config) {
    const auto& keyframes = tracker.keyframes;
    std::set<std::pair<int, int>> connected;
    for (const auto& c : tracker.constraints) {
        connected.insert({c.i, c.j});
        connected.insert({c.j, c.i});
    }
    std::vector<Constraint> closures;
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        const auto& ki = keyframes[i];
        int w = ki.features.size.width, h = ki.features.size.height;
        double diagonal = std::hypot(w, h);
        std::vector<std::pair<double, int>> candidates;
        for (int j = i - 2; j >= 0; --j) {
            if (connected.count({i, j})) continue;
            double overlap = footprint_overlap(ki.transform, keyframes[j].transform, w, h);
            if (overlap >= config.loop_min_overlap) candidates.emplace_back(overlap, j);
        }
        std::sort(candidates.rbegin(), candidates.rend());
        if (static_cast<int>(candidates.size()) > config.loop_max_per_frame) candidates.resize(config.loop_max_per_frame);
        for (const auto& [overlap, j] : candidates) {
            std::string reason;
            auto alignment = tracker.align_features(ki.features, keyframes[j].features, reason);
            if (!alignment) continue;
            // A verified match that disagrees wildly with odometry is a repeated
            // texture (water, forest), not a real revisit.
            Mat3 predicted = keyframes[j].transform.inv() * ki.transform;
            cv::Point2d center(w / 2.0, h / 2.0);
            double disagreement = cv::norm(apply(predicted, center) - apply(alignment->matrix, center));
            if (disagreement > 0.25 * diagonal) {
                tracker.log(format("loop %d->%d: rejected, disagrees with odometry by %.0f px", i, j, disagreement));
                continue;
            }
            closures.push_back({i, j, to_double(alignment->src), to_double(alignment->dst), true});
            connected.insert({i, j});
            connected.insert({j, i});
        }
    }
    return closures;
}

cv::Mat apply_gain(const cv::Mat& frame, double gain) {
    // Lookup table reproduces clip(frame * gain, 0, 255) truncated to uint8.
    cv::Mat lut(1, 256, CV_8U);
    for (int v = 0; v < 256; ++v) lut.at<uchar>(v) = static_cast<uchar>(std::clamp(v * gain, 0.0, 255.0));
    cv::Mat out;
    cv::LUT(frame, lut, out);
    return out;
}

StitchResult stitch_features(FrameSource& source, const StitchConfig& config) {
    StitchResult result;
    auto start = Clock::now();
    Tracker tracker(config);
    int index;
    cv::Mat frame;
    while (source.next(index, frame)) {
        ++result.sampled;
        tracker.push(index, frame);
    }
    tracker.finish();
    result.timings.emplace_back("track", seconds_since(start));
    if (tracker.keyframes.empty()) throw StitchError("The video produced no frames");

    std::vector<Mat3> transforms;
    for (const auto& k : tracker.keyframes) transforms.push_back(k.transform);
    std::vector<Constraint> constraints = tracker.constraints;
    std::vector<Constraint> closures;
    if (config.loop_closure && tracker.keyframes.size() > 2) {
        start = Clock::now();
        closures = find_loop_closures(tracker, config);
        result.timings.emplace_back("loops", seconds_since(start));
    }
    result.sequential_constraints = static_cast<int>(constraints.size());
    result.loop_closures = static_cast<int>(closures.size());

    if (config.bundle_adjust && transforms.size() > 1) {
        start = Clock::now();
        std::vector<Constraint> all = constraints;
        all.insert(all.end(), closures.begin(), closures.end());
        AdjustmentReport report;
        transforms = bundle_adjust(transforms, all, report, config.ransac_threshold);
        result.adjustment = report;
        result.timings.emplace_back("adjust", seconds_since(start));
    }

    std::vector<cv::Mat> frames;
    for (const auto& k : tracker.keyframes) frames.push_back(k.frame);
    if (config.gain_compensation && frames.size() > 1) {
        start = Clock::now();
        result.gains = gain_compensation(frames, transforms);
        result.timings.emplace_back("gains", seconds_since(start));
    }

    start = Clock::now();
    if (!result.gains.empty())
        for (size_t k = 0; k < frames.size(); ++k) frames[k] = apply_gain(frames[k], result.gains[k]);
    std::vector<cv::Size> sizes;
    for (const auto& f : frames) sizes.push_back(f.size());
    int multiple = config.blend == "multiband" ? 32 : 1;
    Canvas canvas = plan_canvas(sizes, transforms, config.max_canvas_pixels, multiple);
    for (auto& m : transforms) m = canvas.offset * m;
    BlendResult blended = config.blend == "multiband" ? multiband_blend(frames, transforms, canvas)
                          : config.blend == "feather" ? feather_blend(frames, transforms, canvas)
                                                      : overwrite_blend(frames, transforms, canvas);
    result.timings.emplace_back("blend", seconds_since(start));

    result.panorama = blended.first;
    result.coverage = blended.second;
    result.transforms = transforms;
    for (const auto& k : tracker.keyframes) result.frame_indices.push_back(k.index);
    result.rejected = tracker.rejected;
    result.relocalized = tracker.relocalized;
    return result;
}

StitchResult stitch_with_opencv(FrameSource& source) {
    StitchResult result;
    std::vector<cv::Mat> images;
    int index;
    cv::Mat frame;
    while (source.next(index, frame)) {
        images.push_back(frame);
        result.frame_indices.push_back(index);
    }
    result.sampled = static_cast<int>(images.size());
    if (images.size() < 2) throw StitchError("cv::Stitcher needs at least two frames");
    auto start = Clock::now();
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
    cv::Stitcher::Status status = stitcher->stitch(images, result.panorama);
    if (status != cv::Stitcher::OK) {
        const char* reasons[] = {"ok", "need more images (not enough overlap or features)", "homography estimation failed",
                                 "camera parameter adjustment failed"};
        int s = static_cast<int>(status);
        throw StitchError(std::string("cv::Stitcher failed: ") + (s >= 0 && s < 4 ? reasons[s] : format("status %d", s)));
    }
    std::vector<cv::Mat> channels;
    cv::split(result.panorama, channels);
    cv::Mat brightest = channels[0];
    for (size_t c = 1; c < channels.size(); ++c) brightest = cv::max(brightest, channels[c]);
    result.coverage = brightest > 0;
    result.timings.emplace_back("stitcher", seconds_since(start));
    return result;
}

bool has_display() {
#if defined(__APPLE__) || defined(_WIN32)
    return true;
#else
    return std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY");
#endif
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s VIDEO [--out FILE] [--every N] [--scale F] [--k1 F] [--detector orb|sift]\n"
                 "       [--features N] [--blend multiband|feather|overwrite] [--no-loops] [--no-adjust]\n"
                 "       [--no-gains] [--transforms CSV] [--method features|stitcher] [--no-show] [-v]\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    StitchConfig config;
    std::string video, out = "stitched.jpg", transforms_path, method = "features";
    bool show = true;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--out") out = value();
            else if (arg == "--every") config.every = std::stoi(value());
            else if (arg == "--scale") config.scale = std::stod(value());
            else if (arg == "--k1") config.k1 = std::stod(value());
            else if (arg == "--detector") config.detector = value();
            else if (arg == "--features") config.n_features = std::stoi(value());
            else if (arg == "--blend") config.blend = value();
            else if (arg == "--no-loops") config.loop_closure = false;
            else if (arg == "--no-adjust") config.bundle_adjust = false;
            else if (arg == "--no-gains") config.gain_compensation = false;
            else if (arg == "--transforms") transforms_path = value();
            else if (arg == "--method") method = value();
            else if (arg == "--no-show") show = false;
            else if (arg == "-v" || arg == "--verbose") config.verbose = true;
            else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else if (arg.size() > 1 && arg[0] == '-') throw std::invalid_argument("unknown option " + arg);
            else if (video.empty()) video = arg;
            else throw std::invalid_argument("unexpected argument " + arg);
        }
        if (video.empty()) throw std::invalid_argument("missing VIDEO");
        if (config.detector != "orb" && config.detector != "sift") throw std::invalid_argument("--detector must be orb or sift");
        if (config.blend != "multiband" && config.blend != "feather" && config.blend != "overwrite")
            throw std::invalid_argument("--blend must be multiband, feather or overwrite");
        if (method != "features" && method != "stitcher") throw std::invalid_argument("--method must be features or stitcher");
        if (config.scale <= 0) throw std::invalid_argument("--scale must be positive");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    StitchResult result;
    try {
        FrameSource source(video, config.every, config.scale, config.k1);
        result = method == "features" ? stitch_features(source, config) : stitch_with_opencv(source);
    } catch (const StitchError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    fs::path out_path(out);
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path());
    if (!cv::imwrite(out, result.panorama)) {
        std::fprintf(stderr, "error: could not write %s\n", out.c_str());
        return 1;
    }
    std::printf("%s\n", result.summary().c_str());
    std::printf("wrote %s\n", out.c_str());
    if (!transforms_path.empty() && !result.transforms.empty()) {
        result.save_transforms(transforms_path);
        std::printf("wrote %s\n", transforms_path.c_str());
    }
    if (show && has_display()) {
        cv::Mat preview = result.panorama;
        const int limit = 1600;
        int longest = std::max(preview.rows, preview.cols);
        if (longest > limit) {
            double factor = double(limit) / longest;
            cv::resize(preview, preview, cv::Size(), factor, factor, cv::INTER_AREA);
        }
        cv::imshow("stitched", preview);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
