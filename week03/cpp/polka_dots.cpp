// Polka dots (C++17 port of week03/dots.py and the scale space search in week03/blobs.py).
//
// Finds every polka dot in a photo and measures it:
//   1. Detect. log, dog and doh search scale space on the CIELAB image
//      (per channel responses, both signs, so the search is color aware and
//      polarity free). contrast runs SimpleBlobDetector on a local Delta E
//      map; gray is the one line baseline on the grayscale image.
//   2. Colors. Each candidate's dot color (its core) and background color
//      (a ring just outside) are estimated in CIELAB.
//   3. Refine. Along 48 rays the edge is where the color crosses halfway from
//      background to dot; a robust circle fit re-centers the rays, then an
//      ellipse fit gives the roundness and the aspect ratio.
//   4. Verify and filter. The inside must be one color, the ring outside a
//      different one, and mostly one color; then radius limits and overlaps.
//
// Usage: polka_dots IMAGE [--method log] [--min-radius 3] [--max-radius R] [--min-contrast 10]
//                         [--out annotated.png] [--json dots.json] [--no-show]

#include <opencv2/opencv.hpp>
// OpenCV 5 moved fitEllipse, moments, contourArea and friends out of imgproc
// into the new geometry module, which opencv.hpp does not include. OpenCV 4
// has no such header, so the include is guarded.
#if CV_VERSION_MAJOR >= 5 && __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

const double kPi = 3.14159265358979323846;
const double kSqrt2 = std::sqrt(2.0);
// Normalized LoG response of a unit contrast disk at its matched scale.
const double DISK_PEAK = 2.0 / std::exp(1.0);

// Python's round(): half to even, which is what nearbyint does in the default mode.
double py_round(double v) { return std::nearbyint(v); }

// np.linspace(start, stop, num, endpoint): step multiples plus an exact endpoint.
std::vector<double> linspace(double start, double stop, int num, bool endpoint = true) {
    std::vector<double> v(num);
    if (num == 1) {
        v[0] = start;
        return v;
    }
    double step = (stop - start) / (endpoint ? num - 1 : num);
    for (int i = 0; i < num; ++i) v[i] = i * step + start;
    if (endpoint) v[num - 1] = stop;
    return v;
}

// np.geomspace: 10 ** linspace(log10(start), log10(stop)) with exact endpoints.
std::vector<double> geomspace(double start, double stop, int num) {
    std::vector<double> v = linspace(std::log10(start), std::log10(stop), num);
    for (double& x : v) x = std::pow(10.0, x);
    v.front() = start;
    if (num > 1) v.back() = stop;
    return v;
}

// ------------------------------------------------------------ color names ----

// OpenCV stores hue as 0..179 (degrees / 2), saturation and value as 0..255.
const int BLACK_MAX_V = 50;
const int ACHROMATIC_MAX_S = 50;
const int WHITE_MIN_V = 190;
const int BROWN_MAX_V = 150;

struct HsvRange {
    cv::Vec3i lower, upper;
};

struct ColorBand {
    std::string name;
    std::vector<HsvRange> ranges;
};

HsvRange hue(int lo, int hi, int v_lo = BLACK_MAX_V + 1, int v_hi = 255) {
    return {cv::Vec3i(lo, ACHROMATIC_MAX_S + 1, v_lo), cv::Vec3i(hi, 255, v_hi)};
}

// Same bands, same order as COLOR_BANDS in week02/colors.py.
const std::vector<ColorBand>& color_bands() {
    static const std::vector<ColorBand> bands = {
        {"red", {hue(0, 8), hue(170, 179)}},
        {"brown", {hue(9, 30, BLACK_MAX_V + 1, BROWN_MAX_V)}},
        {"orange", {hue(9, 20, BROWN_MAX_V + 1)}},
        {"yellow", {hue(21, 30, BROWN_MAX_V + 1), hue(31, 34)}},
        {"green", {hue(35, 85)}},
        {"cyan", {hue(86, 100)}},
        {"blue", {hue(101, 130)}},
        {"purple", {hue(131, 150)}},
        {"pink", {hue(151, 169)}},
        {"black", {{cv::Vec3i(0, 0, 0), cv::Vec3i(179, 255, BLACK_MAX_V)}}},
        {"gray", {{cv::Vec3i(0, 0, BLACK_MAX_V + 1), cv::Vec3i(179, ACHROMATIC_MAX_S, WHITE_MIN_V - 1)}}},
        {"white", {{cv::Vec3i(0, 0, WHITE_MIN_V), cv::Vec3i(179, ACHROMATIC_MAX_S, 255)}}},
    };
    return bands;
}

// Name a single BGR color with the HSV bands (week02 colors.name_color).
std::string name_color(const cv::Vec3i& bgr) {
    cv::Mat pixel(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]));
    cv::Mat hsv;
    cv::cvtColor(pixel, hsv, cv::COLOR_BGR2HSV);
    cv::Vec3b p = hsv.at<cv::Vec3b>(0, 0);
    for (const auto& band : color_bands()) {
        for (const auto& r : band.ranges) {
            bool inside = true;
            for (int c = 0; c < 3; ++c) inside = inside && r.lower[c] <= p[c] && p[c] <= r.upper[c];
            if (inside) return band.name;
        }
    }
    return "unknown";
}

// Float CIELAB image in Delta E units (L in 0..100): image / 255 then BGR2Lab.
cv::Mat lab_image(const cv::Mat& image) {
    cv::Mat f(image.size(), CV_32FC3);
    for (int y = 0; y < image.rows; ++y) {
        const uchar* src = image.ptr<uchar>(y);
        float* dst = f.ptr<float>(y);
        // Divide (not multiply by 1 / 255) so the floats match NumPy bit for bit.
        for (int i = 0; i < image.cols * 3; ++i) dst[i] = static_cast<float>(src[i]) / 255.0f;
    }
    cv::Mat lab;
    cv::cvtColor(f, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Vec3i lab_to_bgr(const cv::Vec3d& lab) {
    cv::Mat pixel(1, 1, CV_32FC3, cv::Scalar(lab[0], lab[1], lab[2]));
    cv::Mat bgr;
    cv::cvtColor(pixel, bgr, cv::COLOR_Lab2BGR);
    cv::Vec3f v = bgr.at<cv::Vec3f>(0, 0);
    cv::Vec3i out;
    for (int c = 0; c < 3; ++c) {
        float scaled = v[c] * 255.0f;
        out[c] = static_cast<int>(std::clamp(static_cast<double>(std::nearbyint(scaled)), 0.0, 255.0));
    }
    return out;
}

// Name a CIELAB color, calling light low chroma colors pastel rather than white.
std::string color_name(const cv::Vec3d& lab) {
    double chroma = std::hypot(lab[1], lab[2]);
    std::string name = name_color(lab_to_bgr(lab));
    if ((name == "white" || name == "gray") && chroma >= 8) {
        double h = std::fmod(std::atan2(lab[2], lab[1]) * 180.0 / kPi, 360.0);
        if (h < 0) h += 360.0;
        // Pastels: hue angle in CIELAB.
        if (200 <= h && h < 290) return "light blue";
        if (290 <= h || h < 30) return "pink";
        if (30 <= h && h < 70) return "beige";
        if (70 <= h && h < 125) return "cream";
        return "light green";
    }
    return name;
}

// ------------------------------------------------------------------- blobs ----

struct Blob {
    double x = 0, y = 0, radius = 0;
    double response = 0;
    std::string color;
    cv::Vec3i bgr{0, 0, 0};
    double contrast = 0;  // Delta E between the blob and its surroundings (0 until measured)
    std::optional<double> roundness, aspect;
    std::vector<float> signature;  // per channel signed response at the peak (scale space only)
    cv::Vec3f dot_lab, background_lab;  // the Python ``_lab`` pair
};

// The knobs of SimpleBlobDetector used by this pipeline (all filters on).
struct BlobFilter {
    double min_area = 20.0, max_area = 1e12;
    double min_circularity = 0.75, min_convexity = 0.9, min_inertia = 0.4;
    int blob_color = 0;  // 0 dark blobs, 255 bright blobs, -1 both
    double min_threshold = 10.0, max_threshold = 250.0, threshold_step = 10.0;
    int min_repeatability = 2;
    double min_dist_between_blobs = 10.0;
};

std::vector<Blob> detect_simple(const cv::Mat& gray, const BlobFilter& f) {
    cv::SimpleBlobDetector::Params p;
    p.minThreshold = static_cast<float>(f.min_threshold);
    p.maxThreshold = static_cast<float>(f.max_threshold);
    p.thresholdStep = static_cast<float>(f.threshold_step);
    p.minRepeatability = static_cast<size_t>(f.min_repeatability);
    p.minDistBetweenBlobs = static_cast<float>(f.min_dist_between_blobs);
    p.filterByColor = f.blob_color >= 0;
    if (f.blob_color >= 0) p.blobColor = static_cast<uchar>(f.blob_color);
    p.filterByArea = true;
    p.minArea = static_cast<float>(f.min_area);
    p.maxArea = static_cast<float>(f.max_area);
    p.filterByCircularity = true;
    p.minCircularity = static_cast<float>(f.min_circularity);
    p.filterByConvexity = true;
    p.minConvexity = static_cast<float>(f.min_convexity);
    p.filterByInertia = true;
    p.minInertiaRatio = static_cast<float>(f.min_inertia);
    std::vector<cv::KeyPoint> keypoints;
    cv::SimpleBlobDetector::create(p)->detect(gray, keypoints);
    std::vector<Blob> blobs;
    for (const auto& k : keypoints) {
        Blob b;
        b.x = k.pt.x;
        b.y = k.pt.y;
        b.radius = k.size / 2.0;
        blobs.push_back(b);
    }
    return blobs;
}

// Area of intersection of two circles divided by the area of the smaller one.
double circle_overlap(double r1, double r2, double d) {
    double small = std::min(r1, r2), big = std::max(r1, r2);
    if (d <= big - small) return 1.0;
    if (d >= r1 + r2) return 0.0;
    double dd = std::max(d, 1e-12);
    double a1 = std::clamp((dd * dd + r1 * r1 - r2 * r2) / (2 * dd * r1), -1.0, 1.0);
    double a2 = std::clamp((dd * dd + r2 * r2 - r1 * r1) / (2 * dd * r2), -1.0, 1.0);
    double area = r1 * r1 * std::acos(a1) + r2 * r2 * std::acos(a2) -
                  0.5 * std::sqrt(std::max((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2), 0.0));
    return area / (kPi * small * small);
}

// Stable sort by response, strongest first (Python's sorted(..., reverse=True)).
void sort_by_response(std::vector<Blob>& blobs) {
    std::stable_sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.response > b.response; });
}

// Greedy non maximum suppression on circles, strongest response first.
std::vector<Blob> prune_overlaps(std::vector<Blob> blobs, double overlap) {
    sort_by_response(blobs);
    std::vector<Blob> kept;
    for (auto& blob : blobs) {
        bool drop = false;
        for (const auto& k : kept) {
            double d = std::hypot(k.x - blob.x, k.y - blob.y);
            if (d < k.radius + blob.radius && circle_overlap(k.radius, blob.radius, d) > overlap) {
                drop = true;
                break;
            }
        }
        if (!drop) kept.push_back(std::move(blob));
    }
    return kept;
}

// Drop peaks that are the surround of a much stronger blob: within ``reach``
// of its radii, ``ratio`` times weaker, and with a per channel signature
// pointing the opposite way (cosine below -0.7).
std::vector<Blob> suppress_sidelobes(std::vector<Blob> blobs, double ratio = 3.0, double reach = 1.6) {
    if (blobs.size() < 2) return blobs;
    sort_by_response(blobs);
    const size_t n = blobs.size();
    bool have = true;
    for (const auto& b : blobs) have = have && !b.signature.empty();
    std::vector<double> norms(n, 0.0);
    if (have)
        for (size_t i = 0; i < n; ++i) {
            double s = 0;
            for (float v : blobs[i].signature) s += static_cast<double>(v) * v;
            norms[i] = std::sqrt(s);
        }
    std::vector<char> alive(n, 1);
    double weakest = blobs.front().response;
    for (const auto& b : blobs) weakest = std::min(weakest, b.response);
    const double floor = weakest * ratio;
    for (size_t i = 0; i < n; ++i) {
        const Blob& strong = blobs[i];
        if (strong.response < floor) break;
        if (!alive[i]) continue;
        for (size_t j = i + 1; j < n; ++j) {
            const Blob& weak = blobs[j];
            bool hit = std::hypot(weak.x - strong.x, weak.y - strong.y) < reach * strong.radius + weak.radius &&
                       weak.response * ratio < strong.response;
            if (hit && have) {
                // A ring is the parent's own response with the sign flipped, in the same channels.
                double dot = 0;
                for (size_t c = 0; c < strong.signature.size(); ++c)
                    dot += static_cast<double>(weak.signature[c]) * strong.signature[c];
                double cosine = dot / std::max(norms[j] * norms[i], 1e-12);
                hit = cosine < -0.7;
            }
            if (hit) alive[j] = 0;
        }
    }
    std::vector<Blob> out;
    for (size_t i = 0; i < n; ++i)
        if (alive[i]) out.push_back(std::move(blobs[i]));
    return out;
}

// ------------------------------------------------------------ scale space ----

// Geometrically spaced scales covering [min_sigma, max_sigma].
std::vector<double> sigma_ladder(double min_sigma, double max_sigma, int per_octave) {
    if (min_sigma <= 0 || max_sigma < min_sigma) throw std::invalid_argument("need 0 < min_sigma <= max_sigma");
    int count = std::max(2, static_cast<int>(std::ceil(per_octave * std::log2(max_sigma / min_sigma))) + 1);
    return geomspace(min_sigma, max_sigma, count);
}

// Downsampled copies of the channels, so large scales are filtered on small
// images. Scale normalized responses do not change with resolution.
class Pyramid {
public:
    explicit Pyramid(std::vector<cv::Mat> channels, double base_sigma = 4.0)
        : base_sigma_(base_sigma), height_(channels[0].rows), width_(channels[0].cols) {
        levels_[1] = std::move(channels);
    }

    size_t channel_count() const { return levels_.at(1).size(); }

    int factor(double sigma) const {
        if (sigma < 2 * base_sigma_) return 1;
        int f = 1 << static_cast<int>(std::floor(std::log2(sigma / base_sigma_)));
        return std::min(f, std::max(1, std::min(height_, width_) / 16));
    }

    const std::vector<cv::Mat>& at(int factor) {
        auto it = levels_.find(factor);
        if (it != levels_.end()) return it->second;
        cv::Size size(std::max(1, static_cast<int>(py_round(static_cast<double>(width_) / factor))),
                      std::max(1, static_cast<int>(py_round(static_cast<double>(height_) / factor))));
        std::vector<cv::Mat> small;
        for (const auto& c : levels_.at(1)) {
            cv::Mat s;
            cv::resize(c, s, size, 0, 0, cv::INTER_AREA);
            small.push_back(s);
        }
        return levels_[factor] = small;
    }

    cv::Mat upsample(const cv::Mat& response) const {
        if (response.rows == height_ && response.cols == width_) return response;
        cv::Mat up;
        cv::resize(response, up, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
        return up;
    }

private:
    double base_sigma_;
    int height_, width_;
    std::map<int, std::vector<cv::Mat>> levels_;
};

cv::Mat blur(const cv::Mat& channel, double sigma) {
    cv::Mat out;
    cv::GaussianBlur(channel, out, cv::Size(0, 0), sigma, sigma, cv::BORDER_REFLECT);
    return out;
}

// Elementwise float32 helpers, written out so the arithmetic matches NumPy's
// float32 array times Python float (the scalar is rounded to float32 first).
cv::Mat scaled(const cv::Mat& m, double factor) {
    cv::Mat out(m.size(), CV_32F);
    const float f = static_cast<float>(factor);
    for (int y = 0; y < m.rows; ++y) {
        const float* src = m.ptr<float>(y);
        float* dst = out.ptr<float>(y);
        for (int x = 0; x < m.cols; ++x) dst[x] = src[x] * f;
    }
    return out;
}

// One scale of the search. ``maps[0]`` is the combined magnitude; with
// several channels, maps[1:] are each channel's signed response, positive
// and negative, so a faint blob living in one channel (pale blue on white is
// almost pure b) is not drowned by a strong neighbor in another channel.
struct Slice {
    double sigma = 0;  // effective sigma
    std::vector<cv::Mat> maps, dilated;
    std::vector<cv::Mat> signed_maps;  // per channel signed responses
};

Slice make_slice(Pyramid& pyr, const std::vector<double>& sigmas, size_t i, const std::string& kind) {
    const double ratio = sigmas.size() > 1 ? sigmas[1] / sigmas[0] : 1.6;
    const double sigma = sigmas[i];
    const int f = pyr.factor(sigma);
    const auto& chans = pyr.at(f);
    const double sl = sigma / f;
    const bool multi = chans.size() > 1;
    Slice out;
    out.sigma = sigma;
    std::vector<cv::Mat> per;
    if (kind == "log") {
        for (const auto& c : chans) {
            cv::Mat lap;
            cv::Laplacian(blur(c, sl), lap, CV_32F, 1, 1, 0, cv::BORDER_REFLECT);
            per.push_back(scaled(lap, -(sl * sl)));
        }
    } else if (kind == "dog") {
        double upper = i + 1 < sigmas.size() ? sigmas[i + 1] : sigma * ratio;
        double k = upper / sigma;
        for (const auto& c : chans) {
            cv::Mat diff = blur(c, sl) - blur(c, upper / f);
            cv::Mat q(diff.size(), CV_32F);
            const float div = static_cast<float>(k - 1.0);
            for (int y = 0; y < diff.rows; ++y) {
                const float* s = diff.ptr<float>(y);
                float* d = q.ptr<float>(y);
                for (int x = 0; x < diff.cols; ++x) d[x] = s[x] / div;
            }
            per.push_back(q);
        }
        out.sigma = std::sqrt(sigma * upper);
    } else {
        // Second derivative kernels: [1 -2 1] and its transpose, and the cross derivative / 4.
        const cv::Matx13f dxx(1.f, -2.f, 1.f);
        const cv::Matx31f dyy(1.f, -2.f, 1.f);
        const cv::Matx33f dxy(0.25f, 0.f, -0.25f, 0.f, 0.f, 0.f, -0.25f, 0.f, 0.25f);
        const float sl4 = static_cast<float>(std::pow(sl, 4));
        for (const auto& c : chans) {
            cv::Mat b = blur(c, sl), lxx, lyy, lxy;
            cv::filter2D(b, lxx, CV_32F, dxx, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
            cv::filter2D(b, lyy, CV_32F, dyy, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
            cv::filter2D(b, lxy, CV_32F, dxy, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
            cv::Mat r(b.size(), CV_32F);
            for (int y = 0; y < b.rows; ++y) {
                const float *xx = lxx.ptr<float>(y), *yy = lyy.ptr<float>(y), *xy = lxy.ptr<float>(y);
                float* d = r.ptr<float>(y);
                for (int x = 0; x < b.cols; ++x) {
                    float det = sl4 * (xx[x] * yy[x] - xy[x] * xy[x]);
                    float trace = xx[x] + yy[x];
                    float sign = trace > 0 ? 1.f : (trace < 0 ? -1.f : 0.f);
                    // Signed like the Laplacian, so bright and dark blobs separate per channel.
                    d[x] = -sign * 2.0f * std::sqrt(std::max(det, 0.0f));
                }
            }
            per.push_back(r);
        }
    }

    cv::Mat combined;
    if (multi) {
        // sqrt of the summed squares; for DoH, 2 sqrt(sum det) equals |LoG| at the center of a round blob.
        combined = per[0].mul(per[0]);
        for (size_t c = 1; c < per.size(); ++c) combined += per[c].mul(per[c]);
        cv::sqrt(combined, combined);
    } else {
        combined = cv::abs(per[0]);  // polarity "both"
    }
    out.maps.push_back(pyr.upsample(combined));
    for (const auto& p : per) out.signed_maps.push_back(pyr.upsample(p));
    if (multi)
        for (const auto& up : out.signed_maps) {
            out.maps.push_back(up);
            out.maps.push_back(-up);
        }
    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    for (const auto& m : out.maps) {
        cv::Mat d;
        cv::dilate(m, d, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        out.dilated.push_back(d);
    }
    return out;
}

// Sub pixel, sub scale refinement of a peak at the middle slice of ``mini``
// by a 3D quadratic fit (as in SIFT). Returns (ds, dy, dx) and the value, or
// nothing when the peak is on the border or the offset leaves the voxel.
std::optional<std::pair<cv::Vec3d, double>> refine_peak(const cv::Mat* mini[3], int y, int x) {
    const int h = mini[1]->rows, w = mini[1]->cols;
    if (y <= 0 || y >= h - 1 || x <= 0 || x >= w - 1) return std::nullopt;
    auto at = [&](int ds, int dy, int dx) { return static_cast<double>(mini[1 + ds]->at<float>(y + dy, x + dx)); };
    double c = at(0, 0, 0);
    cv::Vec3d g((at(1, 0, 0) - at(-1, 0, 0)) / 2, (at(0, 1, 0) - at(0, -1, 0)) / 2, (at(0, 0, 1) - at(0, 0, -1)) / 2);
    double hss = at(1, 0, 0) + at(-1, 0, 0) - 2 * c;
    double hyy = at(0, 1, 0) + at(0, -1, 0) - 2 * c;
    double hxx = at(0, 0, 1) + at(0, 0, -1) - 2 * c;
    double hsy = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) / 4;
    double hsx = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) / 4;
    double hyx = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) / 4;
    cv::Matx33d hessian(hss, hsy, hsx, hsy, hyy, hyx, hsx, hyx, hxx);
    if (std::abs(cv::determinant(hessian)) <= 1e-12) return std::nullopt;
    cv::Vec3d offset;
    if (!cv::solve(hessian, g, offset, cv::DECOMP_LU)) return std::nullopt;
    offset = -offset;
    for (int k = 0; k < 3; ++k)
        if (std::abs(offset[k]) >= 1.0) return std::nullopt;
    return std::make_pair(offset, c + 0.5 * g.dot(offset));
}

// Interpolate sigma at a fractional slice index, geometrically (np.interp on logs).
double sigma_at(const std::vector<double>& sigmas, double index) {
    const int n = static_cast<int>(sigmas.size());
    if (index <= 0) return sigmas.front();
    if (index >= n - 1) return sigmas.back();
    int j = static_cast<int>(std::floor(index));
    double lo = std::log(sigmas[j]), hi = std::log(sigmas[j + 1]);
    return std::exp((hi - lo) * (index - j) + lo);
}

// Find blobs with radius in [min_radius, max_radius] by scale space search.
// The volume is streamed three slices at a time, so memory does not grow
// with the number of scales.
std::vector<Blob> detect_scale_space(const cv::Mat& image, const std::string& kind, double min_radius, double max_radius,
                                     int per_octave, double threshold, double overlap) {
    std::vector<double> sigmas = sigma_ladder(min_radius / kSqrt2, max_radius / kSqrt2, per_octave);
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    Pyramid pyr(channels);
    std::vector<Blob> blobs;
    std::vector<double> effective;

    auto process = [&](const Slice* prev, const Slice& cur, const Slice* next, int index) {
        for (size_t m = 0; m < cur.maps.size(); ++m) {
            const cv::Mat& response = cur.maps[m];
            const cv::Mat* mini[3] = {prev ? &prev->maps[m] : &response, &response, next ? &next->maps[m] : &response};
            for (int y = 0; y < response.rows; ++y) {
                const float* r = response.ptr<float>(y);
                const float* d0 = cur.dilated[m].ptr<float>(y);
                const float* dp = prev ? prev->dilated[m].ptr<float>(y) : nullptr;
                const float* dn = next ? next->dilated[m].ptr<float>(y) : nullptr;
                for (int x = 0; x < response.cols; ++x) {
                    float v = r[x];
                    if (!(v > threshold) || v < d0[x] || (dp && v < dp[x]) || (dn && v < dn[x])) continue;
                    Blob b;
                    b.x = x;
                    b.y = y;
                    b.radius = index;  // holds the fractional scale index until the end
                    b.response = v;
                    if (prev && next) {
                        if (auto fit = refine_peak(mini, y, x)) {
                            b.radius += fit->first[0];
                            b.y += fit->first[1];
                            b.x += fit->first[2];
                            b.response = fit->second;
                        }
                    }
                    for (const auto& s : cur.signed_maps) b.signature.push_back(s.at<float>(y, x));
                    blobs.push_back(std::move(b));
                }
            }
        }
    };

    std::deque<Slice> window;
    for (size_t i = 0; i < sigmas.size(); ++i) {
        window.push_back(make_slice(pyr, sigmas, i, kind));
        effective.push_back(window.back().sigma);
        if (window.size() == 2) {
            process(nullptr, window[0], &window[1], 0);
        } else if (window.size() == 3) {
            process(&window[0], window[1], &window[2], static_cast<int>(effective.size()) - 2);
            window.pop_front();
        }
    }
    if (window.size() == 2)
        process(&window[0], window[1], nullptr, static_cast<int>(effective.size()) - 1);
    else if (window.size() == 1)
        process(nullptr, window[0], nullptr, 0);

    for (auto& b : blobs) b.radius = kSqrt2 * sigma_at(effective, b.radius);
    blobs = suppress_sidelobes(std::move(blobs));
    for (auto& b : blobs) b.signature.clear();
    return prune_overlaps(std::move(blobs), overlap);
}

// ------------------------------------------------------- contrast method ----

// Local background color: a median filter wider than the largest dot, on a
// downsampled image so the kernel stays small, then upsampled.
cv::Mat background_estimate(const cv::Mat& lab, double max_radius) {
    const int h = lab.rows, w = lab.cols;
    const double diameter = 2.5 * max_radius;
    const int factor = std::max(1, static_cast<int>(std::ceil(diameter / 31)));
    cv::Mat small;
    cv::resize(lab, small, cv::Size(std::max(1, w / factor), std::max(1, h / factor)), 0, 0, cv::INTER_AREA);
    int ksize = static_cast<int>(diameter / factor) | 1;
    int short_side = std::min(small.rows, small.cols);
    ksize = std::max(3, std::min(ksize, short_side > 3 ? (short_side - 1) | 1 : 3));
    const float offsets[3] = {0.f, 128.f, 128.f};
    const float scale[3] = {2.55f, 1.f, 1.f};
    cv::Mat as_u8(small.size(), CV_8UC3);
    for (int y = 0; y < small.rows; ++y) {
        const float* s = small.ptr<float>(y);
        uchar* d = as_u8.ptr<uchar>(y);
        for (int x = 0; x < small.cols; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = std::clamp((s[3 * x + c] + offsets[c]) * scale[c], 0.f, 255.f);
                d[3 * x + c] = static_cast<uchar>(v);  // truncation, like astype(np.uint8)
            }
    }
    cv::Mat med_u8;
    cv::medianBlur(as_u8, med_u8, ksize);
    cv::Mat med(small.size(), CV_32FC3);
    for (int y = 0; y < small.rows; ++y) {
        const uchar* s = med_u8.ptr<uchar>(y);
        float* d = med.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x)
            for (int c = 0; c < 3; ++c) d[3 * x + c] = static_cast<float>(s[3 * x + c]) / scale[c] - offsets[c];
    }
    cv::Mat out;
    cv::resize(med, out, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    return out;
}

// Delta E between every pixel and its local background, smoothed.
cv::Mat contrast_map(const cv::Mat& lab, double max_radius, double smooth = 1.5) {
    cv::Mat bg = background_estimate(lab, max_radius);
    cv::Mat delta(lab.size(), CV_32F);
    for (int y = 0; y < lab.rows; ++y) {
        const float* a = lab.ptr<float>(y);
        const float* b = bg.ptr<float>(y);
        float* d = delta.ptr<float>(y);
        for (int x = 0; x < lab.cols; ++x) {
            float e0 = a[3 * x] - b[3 * x], e1 = a[3 * x + 1] - b[3 * x + 1], e2 = a[3 * x + 2] - b[3 * x + 2];
            d[x] = std::sqrt(e0 * e0 + e1 * e1 + e2 * e2);
        }
    }
    if (smooth > 0) cv::GaussianBlur(delta, delta, cv::Size(0, 0), smooth);
    return delta;
}

// ------------------------------------------------------ verify and refine ----

struct DotSettings {
    std::string method = "log";
    double min_radius = 3.0;
    double max_radius = 0.0;  // 0: a sixth of the short image side
    double min_contrast = 10.0;  // Delta E between dot and surroundings
    int per_octave = 5;
    double peak_fraction = 0.5;  // scale space peaks must reach this fraction of an ideal dot's response at min_contrast
    double overlap = 0.3;
    bool verify = true;
    bool refine = true;
    double min_fill = 0.8;  // fraction of the inside that matches the dot color
    double max_leak = 0.35;  // fraction of the outside ring that also matches it
    double min_surround = 0.6;  // fraction of the outside ring that matches the local background
    double min_roundness = 0.93;  // 1 - RMS distance of the edge from the fitted ellipse / radius
    double min_aspect = 0.6;  // minor / major axis; dots seen at an angle are ellipses
    BlobFilter blob_filter = [] {
        BlobFilter f;
        f.blob_color = 255;
        f.min_threshold = 40;
        f.max_threshold = 250;
        f.threshold_step = 15;
        return f;
    }();
};

// Bilinear samples of a float image at arbitrary points (cv::remap,
// INTER_LINEAR, BORDER_REPLICATE), plus an inside-the-image mask.
void sample(const cv::Mat& lab, const std::vector<double>& xs, const std::vector<double>& ys,
            std::vector<cv::Vec3f>& values, std::vector<char>& valid) {
    const int n = static_cast<int>(xs.size());
    const double w = lab.cols, h = lab.rows;
    values.resize(n);
    valid.resize(n);
    cv::Mat mx(n, 1, CV_32F), my(n, 1, CV_32F);
    for (int i = 0; i < n; ++i) {
        valid[i] = xs[i] >= 0 && xs[i] <= w - 1 && ys[i] >= 0 && ys[i] <= h - 1;
        mx.at<float>(i) = static_cast<float>(xs[i]);
        my.at<float>(i) = static_cast<float>(ys[i]);
    }
    // cv::remap is limited to 32767 rows, so sample a column in chunks.
    const int step = 32000;
    for (int start = 0; start < n; start += step) {
        int end = std::min(n, start + step);
        cv::Mat out;
        cv::remap(lab, out, mx.rowRange(start, end), my.rowRange(start, end), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        for (int i = start; i < end; ++i) values[i] = out.at<cv::Vec3f>(i - start);
    }
}

const int kAngles = 24;
const std::vector<double> CORE = {0.0, 0.15, 0.3};
const std::vector<double> INNER = {0.0, 0.3, 0.5, 0.7};
const std::vector<double> OUTER = {1.3, 1.55};

// Samples on circles at ``fractions`` of each blob's radius, blob major.
struct Rings {
    size_t per_blob = 0;
    std::vector<cv::Vec3f> values;
    std::vector<char> valid;
};

Rings rings(const cv::Mat& lab, const std::vector<Blob>& blobs, const std::vector<double>& fractions) {
    static const std::vector<double> angles = linspace(0, 2 * kPi, kAngles, false);
    Rings out;
    out.per_blob = fractions.size() * kAngles;
    std::vector<double> xs, ys;
    xs.reserve(blobs.size() * out.per_blob);
    ys.reserve(blobs.size() * out.per_blob);
    for (const auto& b : blobs)
        for (double f : fractions) {
            double r = b.radius * f;
            for (double a : angles) {
                xs.push_back(b.x + r * std::cos(a));
                ys.push_back(b.y + r * std::sin(a));
            }
        }
    sample(lab, xs, ys, out.values, out.valid);
    return out;
}

// np.median over the valid samples of one blob, per channel (float32 like NumPy).
struct RingStats {
    std::vector<cv::Vec3f> samples;
    double valid_fraction = 0;
};

RingStats ring_stats(const Rings& r, size_t i) {
    RingStats s;
    size_t count = 0;
    for (size_t k = 0; k < r.per_blob; ++k) {
        size_t idx = i * r.per_blob + k;
        if (r.valid[idx]) {
            s.samples.push_back(r.values[idx]);
            ++count;
        }
    }
    s.valid_fraction = static_cast<double>(count) / r.per_blob;
    return s;
}

cv::Vec3f median(const std::vector<cv::Vec3f>& samples) {
    cv::Vec3f out;
    std::vector<float> column(samples.size());
    const size_t n = samples.size(), half = n / 2;
    for (int c = 0; c < 3; ++c) {
        for (size_t i = 0; i < n; ++i) column[i] = samples[i][c];
        std::sort(column.begin(), column.end());
        out[c] = n % 2 ? column[half] : (column[half - 1] + column[half]) / 2.0f;
    }
    return out;
}

float distance(const cv::Vec3f& a, const cv::Vec3f& b) {
    cv::Vec3f d = a - b;
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

// Attach the dot color (from its core) and the local background color (from a
// ring just outside). Only the core is trusted at this stage because
// detectors can overestimate the radius, for example on shaded fabric.
std::vector<Blob> estimate_colors(const cv::Mat& lab, std::vector<Blob> blobs, double min_contrast) {
    if (blobs.empty()) return {};
    Rings core = rings(lab, blobs, CORE), outer = rings(lab, blobs, OUTER);
    std::vector<Blob> kept;
    for (size_t i = 0; i < blobs.size(); ++i) {
        RingStats c = ring_stats(core, i), o = ring_stats(outer, i);
        if (c.valid_fraction < 0.4 || o.valid_fraction < 0.25) continue;
        cv::Vec3f dot = median(c.samples), background = median(o.samples);
        double contrast = distance(dot, background);
        if (contrast < min_contrast) continue;
        blobs[i].contrast = contrast;
        blobs[i].dot_lab = dot;
        blobs[i].background_lab = background;
        kept.push_back(std::move(blobs[i]));
    }
    return kept;
}

// Keep blobs that look like dots at their current center and radius: the
// inside is one color (fill), the ring just outside is a different color
// (leak) and mostly one color (surround).
std::vector<Blob> verify(const cv::Mat& lab, std::vector<Blob> blobs, const DotSettings& s) {
    if (blobs.empty()) return {};
    Rings inner = rings(lab, blobs, INNER), outer = rings(lab, blobs, OUTER);
    std::vector<Blob> kept;
    for (size_t i = 0; i < blobs.size(); ++i) {
        RingStats in = ring_stats(inner, i), around = ring_stats(outer, i);
        if (in.valid_fraction < 0.4 || around.valid_fraction < 0.25) continue;
        cv::Vec3f dot = median(in.samples), background = median(around.samples);
        double contrast = distance(dot, background);
        if (contrast < s.min_contrast) continue;
        double tolerance = std::max(0.5 * contrast, 4.0);
        double surround_tolerance = std::max(0.5 * contrast, 6.0);
        double fill = 0, leak = 0, surround = 0;
        for (const auto& v : in.samples) fill += distance(v, dot) < tolerance;
        for (const auto& v : around.samples) {
            leak += distance(v, dot) < tolerance;
            surround += distance(v, background) < surround_tolerance;
        }
        fill /= in.samples.size();
        leak /= around.samples.size();
        surround /= around.samples.size();
        if (fill < s.min_fill || leak > s.max_leak || surround < s.min_surround) continue;
        Blob& b = blobs[i];
        b.contrast = contrast;
        b.bgr = lab_to_bgr(cv::Vec3d(dot[0], dot[1], dot[2]));
        b.color = color_name(cv::Vec3d(dot[0], dot[1], dot[2]));
        b.dot_lab = dot;
        b.background_lab = background;
        kept.push_back(std::move(b));
    }
    return kept;
}

struct Circle {
    double cx, cy, r;
};

// Algebraic (Kasa) least squares circle fit over the kept points.
Circle kasa_fit(const std::vector<cv::Point2d>& points, const std::vector<char>& keep) {
    cv::Matx33d ata = cv::Matx33d::zeros();
    cv::Vec3d atb(0, 0, 0);
    for (size_t i = 0; i < points.size(); ++i) {
        if (!keep[i]) continue;
        double x = points[i].x, y = points[i].y;
        cv::Vec3d a(x, y, 1.0);
        double b = x * x + y * y;
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) ata(p, q) += a[p] * a[q];
            atb[p] += a[p] * b;
        }
    }
    for (int p = 0; p < 3; ++p) ata(p, p) += 1e-9;
    cv::Vec3d sol(0, 0, 0);
    cv::solve(ata, atb, sol, cv::DECOMP_LU);
    double cx = sol[0] / 2, cy = sol[1] / 2;
    return {cx, cy, std::sqrt(std::max(sol[2] + cx * cx + cy * cy, 0.0))};
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size(), half = n / 2;
    return n % 2 ? v[half] : (v[half - 1] + v[half]) / 2.0;
}

struct EllipseFit {
    double x, y, radius, roundness, aspect;
};

// Fit an ellipse to edge points; return center, equal area radius, roundness and aspect.
EllipseFit ellipse_roundness(const std::vector<cv::Point2d>& points, double cx, double cy, double r) {
    double sum = 0;
    for (const auto& p : points) {
        double res = std::hypot(p.x - cx, p.y - cy) - r;
        sum += res * res;
    }
    EllipseFit circle{cx, cy, r, 1.0 - std::sqrt(sum / points.size()) / std::max(r, 1e-6), 1.0};
    if (points.size() < 8) return circle;
    std::vector<cv::Point2f> pts32(points.begin(), points.end());
    cv::RotatedRect e = cv::fitEllipse(pts32);
    double ex = e.center.x, ey = e.center.y, d1 = e.size.width, d2 = e.size.height;
    double a = std::max(d1, d2) / 2, b = std::min(d1, d2) / 2;
    if (!(b > 0 && std::hypot(ex - cx, ey - cy) < 0.5 * r && 0.5 < std::sqrt(a * b) / r && std::sqrt(a * b) / r < 2.0))
        return circle;
    double t = e.angle * kPi / 180.0;
    double radius = std::sqrt(a * b);
    sum = 0;
    for (const auto& p : points) {
        double du = p.x - ex, dv = p.y - ey;
        double u = du * std::cos(t) + dv * std::sin(t);
        double v = -du * std::sin(t) + dv * std::cos(t);
        double res = (std::sqrt(std::pow(u / (d1 / 2), 2) + std::pow(v / (d2 / 2), 2)) - 1.0) * radius;
        sum += res * res;
    }
    return {ex, ey, radius, 1.0 - std::sqrt(sum / points.size()) / radius, b / a};
}

// Fit an ellipse to the color edge of each dot. Along 48 rays, the edge is
// where the color, projected on the line from background color to dot color,
// crosses halfway. A robust circle fit (two rounds of outlier rejection)
// re-centers the rays; the final edge points get an ellipse fit whose RMS
// residual gives the roundness.
std::vector<Blob> refine(const cv::Mat& lab, std::vector<Blob> blobs, const DotSettings& s, int iterations = 2,
                         int samples = 96, size_t chunk = 256) {
    const int n_rays = 48;
    static const std::vector<double> rays = linspace(0, 2 * kPi, n_rays, false);
    const std::vector<double> fractions = linspace(0.35, 1.8, samples);
    const double w = lab.cols, h = lab.rows;
    const size_t per_blob = static_cast<size_t>(n_rays) * samples;

    // Per dot state of the ray search.
    struct State {
        cv::Vec3d background, direction;
        double contrast, cx, cy, r;
        bool ok = true;
        std::vector<double> dist;
        std::vector<cv::Point2d> edges;
        std::vector<char> keep, has;
    };

    std::vector<Blob> out;
    std::vector<double> xs, ys, level(per_blob);
    std::vector<cv::Vec3f> values;
    std::vector<char> valid;
    for (size_t start = 0; start < blobs.size(); start += chunk) {
        const size_t n = std::min(chunk, blobs.size() - start);
        std::vector<State> states(n);
        for (size_t i = 0; i < n; ++i) {
            const Blob& b = blobs[start + i];
            State& st = states[i];
            const cv::Vec3d dot(b.dot_lab[0], b.dot_lab[1], b.dot_lab[2]);
            st.background = cv::Vec3d(b.background_lab[0], b.background_lab[1], b.background_lab[2]);
            st.direction = dot - st.background;
            st.contrast = std::max(cv::norm(st.direction), 1e-6);
            st.direction /= st.contrast;
            st.cx = b.x;
            st.cy = b.y;
            st.r = b.radius;
            st.dist.resize(samples);
            st.edges.resize(n_rays);
            st.keep.assign(n_rays, 0);
            st.has.assign(n_rays, 0);
        }
        for (int it = 0; it < iterations; ++it) {
            // One remap call for every sample of every dot in the chunk.
            xs.resize(n * per_blob);
            ys.resize(n * per_blob);
            for (size_t i = 0; i < n; ++i) {
                State& st = states[i];
                for (int k = 0; k < samples; ++k) st.dist[k] = st.r * fractions[k];
                for (int ray = 0; ray < n_rays; ++ray)
                    for (int k = 0; k < samples; ++k) {
                        xs[i * per_blob + ray * samples + k] = st.cx + std::cos(rays[ray]) * st.dist[k];
                        ys[i * per_blob + ray * samples + k] = st.cy + std::sin(rays[ray]) * st.dist[k];
                    }
            }
            sample(lab, xs, ys, values, valid);
            for (size_t i = 0; i < n; ++i) {
                State& st = states[i];
                const cv::Vec3f* vals = &values[i * per_blob];
                const char* ok_sample = &valid[i * per_blob];
                // Position of each sample on the line from background color (0) to dot color (1).
                for (size_t k = 0; k < per_blob; ++k) {
                    double proj = 0;
                    for (int c = 0; c < 3; ++c) proj += (static_cast<double>(vals[k][c]) - st.background[c]) * st.direction[c];
                    level[k] = proj / st.contrast;
                }
                int has_count = 0;
                for (int ray = 0; ray < n_rays; ++ray) {
                    const double* lv = &level[ray * samples];
                    const char* vv = &ok_sample[ray * samples];
                    int j = 0;  // first sample outside the dot (0 when there is none)
                    while (j < samples && !(lv[j] < 0.5)) ++j;
                    const bool any = j < samples;
                    if (!any) j = 0;
                    const int jj = std::clamp(j, 1, samples - 1);
                    const double a_val = lv[jj - 1], b_val = lv[jj];
                    st.has[ray] = any && j > 0 && vv[jj] && vv[jj - 1];
                    const double t = (a_val - 0.5) / std::max(a_val - b_val, 1e-6);
                    const double d_edge = st.dist[jj - 1] + t * (st.dist[jj] - st.dist[jj - 1]);
                    st.edges[ray] = {st.cx + std::cos(rays[ray]) * d_edge, st.cy + std::sin(rays[ray]) * d_edge};
                    has_count += st.has[ray];
                }
                st.ok = st.ok && has_count >= 0.35 * n_rays;
                st.keep = st.has;
                // Two rounds of outlier rejection: drop edges more than 3 MAD (at least 1 px) off the circle.
                for (int round = 0; round < 2; ++round) {
                    Circle f = kasa_fit(st.edges, st.keep);
                    std::vector<double> residual(n_rays), kept_residuals;
                    for (int ray = 0; ray < n_rays; ++ray) {
                        residual[ray] = std::abs(std::hypot(st.edges[ray].x - f.cx, st.edges[ray].y - f.cy) - f.r);
                        if (st.keep[ray]) kept_residuals.push_back(residual[ray]);
                    }
                    double mad = kept_residuals.empty() ? 0.0 : median_of(kept_residuals) * 1.4826;
                    if (std::isnan(mad)) mad = 0.0;
                    const double limit = std::max(1.0, 3.0 * mad);
                    for (int ray = 0; ray < n_rays; ++ray) st.keep[ray] = st.has[ray] && residual[ray] <= limit;
                }
                st.ok = st.ok && std::count(st.keep.begin(), st.keep.end(), 1) >= 6;
                Circle f = kasa_fit(st.edges, st.keep);
                if (st.ok && std::isfinite(f.r) && f.r > 0) {
                    st.cx = f.cx;
                    st.cy = f.cy;
                    st.r = f.r;
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            Blob& blob = blobs[start + i];
            const State& st = states[i];
            const bool inside = -0.3 * st.r <= st.cx && st.cx < w - 1 + 0.3 * st.r && -0.3 * st.r <= st.cy &&
                                st.cy < h - 1 + 0.3 * st.r;
            if (!(st.ok && inside && st.r > 0)) {
                if (!s.verify) out.push_back(std::move(blob));
                continue;
            }
            std::vector<cv::Point2d> points;
            for (int ray = 0; ray < n_rays; ++ray)
                if (st.keep[ray]) points.push_back(st.edges[ray]);
            EllipseFit e = ellipse_roundness(points, st.cx, st.cy, st.r);
            const double had = static_cast<double>(std::max<std::ptrdiff_t>(std::count(st.has.begin(), st.has.end(), 1), 1));
            // Penalize shapes where many rays disagree with the circle.
            const double roundness = e.roundness * std::sqrt(static_cast<double>(points.size()) / had);
            if (s.verify && (roundness < s.min_roundness || e.aspect < s.min_aspect)) continue;
            blob.x = e.x;
            blob.y = e.y;
            blob.radius = e.radius;
            blob.roundness = roundness;
            blob.aspect = e.aspect;
            out.push_back(std::move(blob));
        }
    }
    return out;
}

void describe_colors(std::vector<Blob>& blobs) {
    for (auto& b : blobs) {
        cv::Vec3d dot(b.dot_lab[0], b.dot_lab[1], b.dot_lab[2]);
        b.bgr = lab_to_bgr(dot);
        b.color = color_name(dot);
    }
}

// --------------------------------------------------------------- pipeline ----

struct DotReport {
    std::vector<Blob> dots;
    std::string method;
    double seconds = 0;
    size_t candidates = 0;
};

std::vector<Blob> candidates(const cv::Mat& image, const cv::Mat& lab, const DotSettings& s, double max_radius) {
    const double min_area = kPi * s.min_radius * s.min_radius * 0.8;
    const double max_area = kPi * max_radius * max_radius * 1.2;
    if (s.method == "gray") {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        BlobFilter f;
        f.min_area = min_area;
        f.max_area = max_area;
        f.blob_color = -1;
        return detect_simple(gray, f);
    }
    if (s.method == "contrast") {
        cv::Mat delta = contrast_map(lab, max_radius);
        // Scale so that 255 is a strong contrast and SimpleBlobDetector's threshold sweep spans useful levels.
        cv::Mat gray(delta.size(), CV_8U);
        const float factor = static_cast<float>(255.0 / (4.0 * s.min_contrast));
        for (int y = 0; y < delta.rows; ++y) {
            const float* d = delta.ptr<float>(y);
            uchar* g = gray.ptr<uchar>(y);
            for (int x = 0; x < delta.cols; ++x) g[x] = static_cast<uchar>(std::clamp(d[x] * factor, 0.f, 255.f));
        }
        BlobFilter f = s.blob_filter;
        f.min_area = min_area;
        f.max_area = max_area;
        f.min_threshold = 32.0;
        f.threshold_step = 16.0;
        std::vector<Blob> found = detect_simple(gray, f);
        for (auto& b : found) b.response = b.radius;
        return found;
    }
    if (s.method == "log" || s.method == "dog" || s.method == "doh") {
        double threshold = DISK_PEAK * s.min_contrast * s.peak_fraction;
        return detect_scale_space(lab, s.method, s.min_radius, max_radius, s.per_octave, threshold, s.overlap);
    }
    throw std::invalid_argument("method must be one of gray, contrast, log, dog, doh");
}

DotReport find_dots(const cv::Mat& image, const DotSettings& s) {
    auto start = std::chrono::steady_clock::now();
    cv::Mat lab = lab_image(image);
    const double max_radius = s.max_radius > 0 ? s.max_radius : std::min(image.rows, image.cols) / 6.0;
    std::vector<Blob> dots = candidates(image, lab, s, max_radius);
    const size_t count = dots.size();
    dots = estimate_colors(lab, std::move(dots), 0.6 * s.min_contrast);
    if (s.refine) dots = refine(lab, std::move(dots), s);
    if (s.verify)
        dots = verify(lab, std::move(dots), s);
    else
        describe_colors(dots);
    std::vector<Blob> sized;
    for (auto& d : dots)
        if (s.min_radius * 0.8 <= d.radius && d.radius <= max_radius * 1.25) sized.push_back(std::move(d));
    std::stable_sort(sized.begin(), sized.end(), [](const Blob& a, const Blob& b) {
        return a.contrast * a.roundness.value_or(0.5) > b.contrast * b.roundness.value_or(0.5);
    });
    dots = prune_overlaps(std::move(sized), s.overlap);
    // Reading order: rows of dots (by y in units of the dot radius), then x.
    std::stable_sort(dots.begin(), dots.end(), [](const Blob& a, const Blob& b) {
        double ra = py_round(a.y / std::max(a.radius, 1.0)), rb = py_round(b.y / std::max(b.radius, 1.0));
        return ra != rb ? ra < rb : a.x < b.x;
    });
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return {std::move(dots), s.method, seconds, count};
}

// ------------------------------------------------------------------ output ----

// Draw each dot's fitted circle in a contrasting ring, with its center.
cv::Mat annotate(const cv::Mat& image, const std::vector<Blob>& dots) {
    cv::Mat out = image.clone();
    int t = std::max(1, static_cast<int>(py_round(std::min(image.rows, image.cols) / 400.0)));
    for (const auto& d : dots) {
        cv::Point center(static_cast<int>(py_round(d.x * 16)), static_cast<int>(py_round(d.y * 16)));
        int radius = static_cast<int>(py_round(d.radius * 16));
        cv::circle(out, center, radius, cv::Scalar(255, 255, 255), t + 2, cv::LINE_AA, 4);
        cv::circle(out, center, radius, cv::Scalar(20, 20, 20), t, cv::LINE_AA, 4);
        cv::circle(out, center, 2 * 16, cv::Scalar(20, 20, 20), -1, cv::LINE_AA, 4);
    }
    return out;
}

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    return buffer;
}

// Shortest decimal that round trips, like Python's float repr.
std::string json_number(double v) {
    for (int precision = 12; precision <= 17; ++precision) {
        std::string s = format("%.*g", precision, v);
        if (std::strtod(s.c_str(), nullptr) == v) return s;
    }
    return format("%.17g", v);
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

// Color counts, most common first (ties keep first seen order).
std::vector<std::pair<std::string, int>> counts(const std::vector<Blob>& dots) {
    std::vector<std::pair<std::string, int>> out;
    for (const auto& d : dots) {
        std::string name = d.color.empty() ? "?" : d.color;
        auto it = std::find_if(out.begin(), out.end(), [&](const auto& kv) { return kv.first == name; });
        if (it == out.end())
            out.emplace_back(name, 1);
        else
            ++it->second;
    }
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

std::string to_json(const DotReport& report) {
    std::ostringstream o;
    o << "{\n  \"method\": " << json_string(report.method) << ",\n";
    o << "  \"count\": " << report.dots.size() << ",\n";
    o << "  \"counts_by_color\": {";
    auto by_color = counts(report.dots);
    for (size_t i = 0; i < by_color.size(); ++i)
        o << (i ? ", " : "") << json_string(by_color[i].first) << ": " << by_color[i].second;
    o << "},\n";
    o << "  \"seconds\": " << json_number(report.seconds) << ",\n";
    o << "  \"candidates\": " << report.candidates << ",\n";
    o << "  \"dots\": [";
    for (size_t i = 0; i < report.dots.size(); ++i) {
        const Blob& d = report.dots[i];
        o << (i ? "," : "") << "\n    {\"x\": " << json_number(d.x) << ", \"y\": " << json_number(d.y)
          << ", \"radius\": " << json_number(d.radius) << ", \"color\": " << json_string(d.color)
          << ", \"bgr\": [" << d.bgr[0] << ", " << d.bgr[1] << ", " << d.bgr[2] << "]"
          << ", \"contrast\": " << json_number(d.contrast)
          << ", \"roundness\": " << (d.roundness ? json_number(*d.roundness) : "null")
          << ", \"aspect\": " << (d.aspect ? json_number(*d.aspect) : "null") << "}";
    }
    o << (report.dots.empty() ? "]\n}\n" : "\n  ]\n}\n");
    return o.str();
}

bool has_display() {
#if defined(__APPLE__) || defined(_WIN32)
    return true;
#else
    return std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY");
#endif
}

int usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s IMAGE [--method gray|contrast|log|dog|doh] [--min-radius F] [--max-radius F]\n"
                 "       [--min-contrast F] [--out FILE] [--json FILE] [--no-show]\n",
                 prog);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path, out_path, json_path;
    bool show = true;
    DotSettings settings;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--method") settings.method = value();
            else if (arg == "--min-radius") settings.min_radius = std::stod(value());
            else if (arg == "--max-radius") settings.max_radius = std::stod(value());
            else if (arg == "--min-contrast") settings.min_contrast = std::stod(value());
            else if (arg == "--out") out_path = value();
            else if (arg == "--json") json_path = value();
            else if (arg == "--no-show") show = false;
            else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            }
            else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("unknown option " + arg);
            else if (image_path.empty()) image_path = arg;
            else throw std::invalid_argument("unexpected argument " + arg);
        }
        const std::vector<std::string> methods = {"gray", "contrast", "log", "dog", "doh"};
        if (std::find(methods.begin(), methods.end(), settings.method) == methods.end())
            throw std::invalid_argument("method must be one of gray, contrast, log, dog, doh (simple and contour are Python only)");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return usage(argv[0]);
    }
    if (image_path.empty()) return usage(argv[0]);

    if (!fs::is_regular_file(image_path)) {
        std::fprintf(stderr, "error: No image found at %s\n", image_path.c_str());
        return 2;
    }
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::fprintf(stderr, "error: OpenCV could not decode %s as an image\n", image_path.c_str());
        return 2;
    }

    DotReport report = find_dots(image, settings);
    std::printf("%s: %dx%d px, %zu dots (%s, %zu candidates, %.3f s)\n", image_path.c_str(), image.cols, image.rows,
                report.dots.size(), report.method.c_str(), report.candidates, report.seconds);
    for (const auto& [name, n] : counts(report.dots)) std::printf("  %-12s %d\n", name.c_str(), n);

    cv::Mat annotated = annotate(image, report.dots);
    if (!out_path.empty()) {
        cv::imwrite(out_path, annotated);
        std::printf("wrote %s\n", out_path.c_str());
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << to_json(report);
        std::printf("wrote %s\n", json_path.c_str());
    }
    if (show && has_display()) {
        cv::imshow(report.method + ": " + std::to_string(report.dots.size()) + " dots", annotated);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
