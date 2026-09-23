// Game pieces (C++17 port of week03/objects.py with the default ObjectSettings).
//
// Detects and classifies cones, cubes and rings with classical vision:
//   1. Foreground. Seeds are saturated, bright pixels (plastic, not floor or
//      cardboard); each color grows by hysteresis into weaker saturation of
//      the same hue as long as it stays connected to a seed.
//   2. Colors. Foreground hues are clustered with k-means on the hue circle,
//      the number of clusters chosen by an elbow rule.
//   3. Instances. A wide closing bridges printed logos, then touching pieces
//      of one color are cut between convexity defects when the distance
//      transform shows more than one thick center.
//   4. Shape and class. Color free descriptors (hole, solidity, rectangle
//      fill, triangularity) and geometric rules: a ring has a central hole, a
//      cube has a convex silhouette, a cone's hull is nearly a triangle.
//
// Usage: game_pieces IMAGE [--out annotated.png] [--json pieces.json] [--no-show]

#include <opencv2/opencv.hpp>
// OpenCV 5 moved moments, contourArea, convexHull, minAreaRect and
// minEnclosingTriangle into the new geometry module, which opencv.hpp does
// not include. OpenCV 4 has no such header, so the include is guarded.
#if CV_VERSION_MAJOR >= 5 && __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

const std::vector<std::string> CLASSES = {"cone", "cube", "ring"};

cv::Scalar class_bgr(const std::string& label) {
    if (label == "cone") return {0, 200, 255};
    if (label == "cube") return {230, 90, 120};
    if (label == "ring") return {60, 60, 240};
    return {180, 180, 180};
}

struct ObjectSettings {
    int min_saturation = 110;  // seeds: clearly colored plastic
    int grow_saturation = 70;  // hysteresis: shaded sides of a piece, if connected to a seed of the same hue
    double hue_tolerance = 8.0;  // OpenCV hue units (half degrees)
    int min_value = 120;
    double min_area_fraction = 0.004;  // of the image
    bool split_touching = true;
};

struct Features {
    double area;
    double hole_ratio;  // largest hole area / filled area
    double hole_centered;  // distance of the hole center from the blob center, / equivalent radius
    double rect_fill;  // area / minimum area rectangle
    double triangle_fill;  // area / minimum enclosing triangle
    double triangularity;  // convex hull area / minimum enclosing triangle
    double solidity;  // area / convex hull area
    double elongation;  // minor / major side of the minimum area rectangle
};

struct Piece {
    std::string label;
    double confidence;
    cv::Mat mask;  // 0/1 CV_8U
    cv::Rect bbox;
    cv::Point2d center;
    double hue;
    Features features;
    std::array<double, 3> scores;  // cone, cube, ring
};

// 0/1 CV_8U mask of the nonzero pixels.
cv::Mat binary(const cv::Mat& m) {
    cv::Mat out = (m > 0) / 255;
    return out;
}

// ------------------------------------------------------------- foreground ----

cv::Mat blurred_hsv(const cv::Mat& image) {
    cv::Mat blurred, hsv;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

cv::Mat foreground_mask(const cv::Mat& image, const ObjectSettings& s) {
    cv::Mat hsv = blurred_hsv(image);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, s.min_saturation, s.min_value), cv::Scalar(255, 255, 255), mask);
    return mask;
}

// Cluster centers on the hue circle (OpenCV hue, 0..180). Hues become unit
// vectors so 179 and 1 are neighbors; k grows until the spread stops
// improving much (elbow), then centers closer than ``merge_degrees`` merge.
std::vector<double> hue_clusters(const std::vector<uchar>& hues, int max_k = 6, double merge_degrees = 20.0) {
    // Python samples 20000 points at random with NumPy's generator; a regular
    // stride gives the same distribution without depending on it.
    const size_t limit = 20000;
    std::vector<size_t> picks;
    if (hues.size() > limit) {
        for (size_t i = 0; i < limit; ++i) picks.push_back(i * hues.size() / limit);
    } else {
        for (size_t i = 0; i < hues.size(); ++i) picks.push_back(i);
    }
    cv::Mat points(static_cast<int>(picks.size()), 2, CV_32F);
    for (size_t i = 0; i < picks.size(); ++i) {
        double angle = hues[picks[i]] * 2 * kPi / 180;
        points.at<float>(static_cast<int>(i), 0) = static_cast<float>(std::cos(angle));
        points.at<float>(static_cast<int>(i), 1) = static_cast<float>(std::sin(angle));
    }
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 1e-3);
    cv::setRNGSeed(0);
    std::optional<std::pair<double, cv::Mat>> best;
    for (int n = 1; n <= max_k; ++n) {
        if (points.rows < n) break;
        cv::Mat labels, centers;
        double compactness = cv::kmeans(points, n, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
        // Elbow: stop adding clusters once the spread barely improves.
        if (!best || compactness < 0.6 * best->first)
            best = std::make_pair(compactness, centers);
        else
            break;
    }
    std::vector<float> degrees;
    for (int i = 0; i < best->second.rows; ++i) {
        float d = static_cast<float>(std::atan2(best->second.at<float>(i, 1), best->second.at<float>(i, 0)) * 180.0 / kPi);
        d = std::fmod(d, 360.0f);
        if (d < 0) d += 360.0f;
        degrees.push_back(d / 2);  // back to 0..180
    }
    std::sort(degrees.begin(), degrees.end());
    std::vector<double> merged;
    for (float h : degrees) {
        bool apart = true;
        for (double m : merged) {
            double d = std::abs(h - m);
            apart = apart && std::min(d, 180 - d) * 2 > merge_degrees;
        }
        if (apart) merged.push_back(h);
    }
    return merged;
}

double hue_distance(double a, double b) {
    double d = std::abs(a - b);
    return std::min(d, 180 - d);
}

// ---------------------------------------------------------------- shapes ----

using Contour = std::vector<cv::Point>;

// Index of the largest contour by area (the first one on ties, like Python's max).
size_t largest(const std::vector<Contour>& contours) {
    size_t best = 0;
    double best_area = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > best_area) {
            best_area = a;
            best = i;
        }
    }
    return best;
}

cv::Mat fill_holes(const cv::Mat& mask) {
    std::vector<Contour> contours;
    cv::findContours(binary(mask), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat filled = cv::Mat::zeros(mask.size(), CV_8U);
    cv::drawContours(filled, contours, -1, cv::Scalar(1), cv::FILLED);
    return filled;
}

// Well separated maxima of the distance transform: roughly one per convex piece.
std::vector<cv::Point> distance_peaks(const cv::Mat& filled) {
    cv::Mat dist;
    cv::distanceTransform(filled, dist, cv::DIST_L2, 5);
    double peak = 0;
    cv::minMaxLoc(dist, nullptr, &peak);
    if (peak <= 0) return {};
    int window = std::max(3, static_cast<int>(0.9 * peak) | 1);
    cv::Mat dilated;
    cv::dilate(dist, dilated, cv::Mat::ones(window, window, CV_8U));
    // NumPy compares float32 pixels with the threshold rounded to float32.
    const float floor = static_cast<float>(0.55 * peak);
    cv::Mat local_max = (dist >= dilated) & (dist >= floor);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(local_max / 255, labels, stats, centroids, 8);
    std::vector<cv::Point> seeds;
    for (int i = 1; i < n; ++i)
        seeds.emplace_back(static_cast<int>(std::nearbyint(centroids.at<double>(i, 0))),
                           static_cast<int>(std::nearbyint(centroids.at<double>(i, 1))));
    std::stable_sort(seeds.begin(), seeds.end(),
                     [&](const cv::Point& a, const cv::Point& b) { return dist.at<float>(a) > dist.at<float>(b); });
    std::vector<cv::Point> merged;
    for (const auto& s : seeds) {
        bool apart = true;
        for (const auto& m : merged) apart = apart && std::hypot(s.x - m.x, s.y - m.y) > 1.2 * peak;
        if (apart) merged.push_back(s);
    }
    return merged;
}

double side(const cv::Point2d& p, const cv::Point2d& q, const cv::Point& point) {
    return (q.x - p.x) * (point.y - p.y) - (q.y - p.y) * (point.x - p.x);
}

// The shortest segment between two convexity defects that separates two peaks, inside the mask.
std::optional<std::pair<cv::Point2d, cv::Point2d>> defect_cut(const cv::Mat& filled, const cv::Point& a,
                                                              const cv::Point& b) {
    std::vector<Contour> contours;
    cv::findContours(filled, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return std::nullopt;
    const Contour& contour = contours[largest(contours)];
    std::vector<int> hull;
    cv::convexHull(contour, hull, false, false);
    if (hull.size() < 4) return std::nullopt;
    std::sort(hull.begin(), hull.end());
    std::vector<cv::Vec4i> defects;
    try {
        cv::convexityDefects(contour, hull, defects);
    } catch (const cv::Exception&) {
        return std::nullopt;
    }
    const double scale = std::sqrt(static_cast<double>(cv::countNonZero(filled)));
    std::vector<std::pair<cv::Point2d, double>> points;
    for (const auto& d : defects) {
        double depth = d[3] / 256.0;
        if (depth >= 0.04 * scale) points.emplace_back(cv::Point2d(contour[d[2]]), depth);
    }
    std::optional<std::tuple<double, cv::Point2d, cv::Point2d>> best;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const auto& [p, dp] = points[i];
            const auto& [q, dq] = points[j];
            double length = cv::norm(q - p);
            if (length < 1) continue;
            if (side(p, q, a) * side(p, q, b) >= 0) continue;  // both peaks on the same side
            const int steps = std::max(2, static_cast<int>(length));
            double inside = 0;
            for (int k = 0; k < steps; ++k) {
                double t = k == steps - 1 ? 1.0 : k * (1.0 / (steps - 1));  // np.linspace(0, 1, steps)
                double lx = p.x + t * (q.x - p.x), ly = p.y + t * (q.y - p.y);
                int x = std::clamp(static_cast<int>(lx), 0, filled.cols - 1);
                int y = std::clamp(static_cast<int>(ly), 0, filled.rows - 1);
                inside += filled.at<uchar>(y, x);
            }
            if (inside / steps < 0.85) continue;
            double score = length / (dp + dq);
            if (!best || score < std::get<0>(*best)) best = std::make_tuple(score, p, q);
        }
    }
    if (!best) return std::nullopt;
    return std::make_pair(std::get<1>(*best), std::get<2>(*best));
}

// Split touching pieces of one color. Each piece is convex or nearly so, so
// two touching pieces form a blob with two thick centers (distance transform
// peaks) and a notch on each side where they meet (convexity defects). The
// blob is cut along the shortest defect to defect segment that separates the
// peaks. A single cone is concave too, but has one thick center.
std::vector<cv::Mat> split_touching(const cv::Mat& mask, int min_area) {
    cv::Mat labels;
    int count = cv::connectedComponents(mask, labels, 8, CV_32S);
    std::vector<cv::Mat> out, queue;
    for (int i = 1; i < count; ++i) queue.push_back((labels == i) / 255);
    while (!queue.empty()) {
        cv::Mat part = queue.back();
        queue.pop_back();
        if (cv::countNonZero(part) < min_area) continue;
        cv::Mat filled = fill_holes(part);
        std::vector<cv::Point> peaks = distance_peaks(filled);
        auto cut = peaks.size() >= 2 ? defect_cut(filled, peaks[0], peaks[1]) : std::nullopt;
        if (!cut) {
            out.push_back(part);
            continue;
        }
        cv::Mat severed = filled.clone();
        cv::Point p(static_cast<int>(std::nearbyint(cut->first.x)), static_cast<int>(std::nearbyint(cut->first.y)));
        cv::Point q(static_cast<int>(std::nearbyint(cut->second.x)), static_cast<int>(std::nearbyint(cut->second.y)));
        cv::line(severed, p, q, cv::Scalar(0), 3);
        cv::Mat piece_labels;
        int n = cv::connectedComponents(severed, piece_labels, 8, CV_32S);
        std::vector<cv::Mat> pieces;
        for (int j = 1; j < n; ++j) {
            cv::Mat pc = (piece_labels == j) / 255;
            if (cv::countNonZero(pc) >= min_area) pieces.push_back(pc);
        }
        if (pieces.size() < 2) {
            out.push_back(part);
            continue;
        }
        // Give the cut line back to the nearest piece, then restore the part's own holes.
        cv::Mat owner = cv::Mat::zeros(mask.size(), CV_32S);
        for (size_t j = 0; j < pieces.size(); ++j) owner.setTo(static_cast<int>(j + 1), pieces[j]);
        cv::Mat background = (owner == 0) / 255, dist, nearest;
        cv::distanceTransform(background, dist, nearest, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
        double max_label = 0;
        cv::minMaxLoc(nearest, nullptr, &max_label);
        std::vector<int> label_of_pixel(static_cast<size_t>(max_label) + 1, 0);
        for (int y = 0; y < owner.rows; ++y) {
            const int* o = owner.ptr<int>(y);
            const int* l = nearest.ptr<int>(y);
            for (int x = 0; x < owner.cols; ++x)
                if (o[x] > 0) label_of_pixel[l[x]] = o[x];
        }
        std::vector<cv::Mat> next(pieces.size());
        for (auto& m : next) m = cv::Mat::zeros(mask.size(), CV_8U);
        for (int y = 0; y < owner.rows; ++y) {
            const int* l = nearest.ptr<int>(y);
            const uchar* src = part.ptr<uchar>(y);
            for (int x = 0; x < owner.cols; ++x) {
                int j = label_of_pixel[l[x]];
                if (j > 0 && src[x]) next[j - 1].at<uchar>(y, x) = 1;
            }
        }
        for (auto& m : next) queue.push_back(m);
        if (out.size() + queue.size() > 32) break;
    }
    return out;
}

// Fill holes smaller than ``fraction`` of the filled area.
cv::Mat close_small_holes(const cv::Mat& mask, double fraction) {
    cv::Mat filled = fill_holes(mask);
    cv::Mat solid = binary(mask), holes;
    cv::subtract(filled, solid, holes);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(holes, labels, stats, centroids, 8);
    const double limit = fraction * cv::countNonZero(filled);
    cv::Mat out = solid;
    for (int j = 1; j < n; ++j)
        if (stats.at<int>(j, cv::CC_STAT_AREA) < limit) out.setTo(1, labels == j);
    return out;
}

// Shape descriptors of one instance mask.
Features describe(const cv::Mat& mask) {
    cv::Mat filled = fill_holes(mask);
    std::vector<Contour> contours;
    cv::findContours(filled, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    const Contour& contour = contours[largest(contours)];
    const double area = cv::countNonZero(filled);
    cv::Mat holes;
    cv::subtract(filled, binary(mask), holes);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(holes, labels, stats, centroids, 8);
    double hole_ratio = 0.0, hole_centered = 1.0;
    cv::Moments m = cv::moments(filled, true);
    double cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    double radius = std::sqrt(area / kPi);
    if (n > 1) {
        int j = 1;
        for (int k = 2; k < n; ++k)
            if (stats.at<int>(k, cv::CC_STAT_AREA) > stats.at<int>(j, cv::CC_STAT_AREA)) j = k;
        hole_ratio = stats.at<int>(j, cv::CC_STAT_AREA) / area;
        hole_centered = std::hypot(centroids.at<double>(j, 0) - cx, centroids.at<double>(j, 1) - cy) / radius;
    }
    cv::RotatedRect rect = cv::minAreaRect(contour);
    double rw = rect.size.width, rh = rect.size.height;
    double rect_area = std::max(rw * rh, 1e-6);
    std::vector<cv::Point2f> contour32(contour.begin(), contour.end());
    std::vector<cv::Point2f> triangle;
    double tri_area = cv::minEnclosingTriangle(contour32, triangle);
    Contour hull;
    cv::convexHull(contour, hull);
    double hull_area = std::max(cv::contourArea(hull), 1e-6);
    // Fill ratios compare polygon areas with polygon areas (all measured through pixel centers).
    double polygon_area = cv::contourArea(contour);
    Features f;
    f.area = area;
    f.hole_ratio = hole_ratio;
    f.hole_centered = hole_centered;
    f.rect_fill = polygon_area / rect_area;
    f.triangle_fill = polygon_area / std::max(tri_area, 1e-6);
    f.triangularity = hull_area / std::max(tri_area, 1e-6);
    f.solidity = polygon_area / hull_area;
    f.elongation = std::min(rw, rh) / std::max(std::max(rw, rh), 1e-6);
    return f;
}

double ramp(double x, double lo, double hi) { return std::clamp((x - lo) / (hi - lo), 0.0, 1.0); }

// Scores in [0, 1] for each class from shape alone; the best one wins.
//   ring: a large hole near the middle.
//   cube: a convex polyhedron, so a convex silhouette that fills most of its
//         minimum rectangle and is not a triangle.
//   cone: concave where the body meets the base flange, and a hull that is
//         close to a triangle, standing or lying down.
std::pair<std::string, double> classify(const Features& f, std::array<double, 3>& scores) {
    double no_hole = 1.0 - ramp(f.hole_ratio, 0.03, 0.12);
    double convex = ramp(f.solidity, 0.90, 0.94);
    double ring = ramp(f.hole_ratio, 0.05, 0.15) * (1.0 - ramp(f.hole_centered, 0.3, 0.6));
    double cube = no_hole * convex * ramp(f.rect_fill, 0.68, 0.74) * (1.0 - ramp(f.triangularity, 0.72, 0.80));
    double cone = no_hole * ramp(f.solidity, 0.72, 0.80) * std::max(1.0 - convex, ramp(f.triangularity, 0.66, 0.72));
    scores = {cone, cube, ring};
    size_t best = 0;  // first maximum wins, like Python's max over the dict
    for (size_t i = 1; i < 3; ++i)
        if (scores[i] > scores[best]) best = i;
    if (scores[best] < 0.3) return {"unknown", scores[best]};
    return {CLASSES[best], scores[best]};
}

// --------------------------------------------------------------- pipeline ----

struct ObjectReport {
    std::vector<Piece> pieces;
    double seconds;
    std::vector<double> hues;
};

ObjectReport detect_objects(const cv::Mat& image, const ObjectSettings& s = ObjectSettings()) {
    auto start = std::chrono::steady_clock::now();
    const int h = image.rows, w = image.cols;
    const int min_area = static_cast<int>(s.min_area_fraction * h * w);
    cv::Mat fg = foreground_mask(image, s);
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_planes;
    cv::split(hsv, hsv_planes);
    const cv::Mat& hue_plane = hsv_planes[0];
    std::vector<uchar> hues;
    for (int y = 0; y < h; ++y) {
        const uchar* f = fg.ptr<uchar>(y);
        const uchar* hp = hue_plane.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
            if (f[x]) hues.push_back(hp[x]);
    }
    std::vector<double> centers = hues.empty() ? std::vector<double>() : hue_clusters(hues);
    std::vector<Piece> pieces;
    if (!centers.empty()) {
        // Hue only has 256 values, so nearest center and distances are lookup tables.
        std::array<int, 256> nearest{};
        std::vector<std::array<double, 256>> distance(centers.size());
        for (int v = 0; v < 256; ++v) {
            for (size_t i = 0; i < centers.size(); ++i) {
                distance[i][v] = hue_distance(v, centers[i]);
                if (distance[i][v] < distance[nearest[v]][v]) nearest[v] = static_cast<int>(i);
            }
        }
        cv::Mat blurred = blurred_hsv(image), weak;
        cv::inRange(blurred, cv::Scalar(0, s.grow_saturation, s.min_value), cv::Scalar(255, 255, 255), weak);
        const int close = std::max(7, static_cast<int>(0.02 * std::hypot(h, w)) | 1);
        const cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close, close));
        for (size_t i = 0; i < centers.size(); ++i) {
            cv::Mat seeds = cv::Mat::zeros(h, w, CV_8U), grow = cv::Mat::zeros(h, w, CV_8U);
            for (int y = 0; y < h; ++y) {
                const uchar* hp = hue_plane.ptr<uchar>(y);
                const uchar* f = fg.ptr<uchar>(y);
                const uchar* wk = weak.ptr<uchar>(y);
                uchar* sd = seeds.ptr<uchar>(y);
                uchar* gr = grow.ptr<uchar>(y);
                for (int x = 0; x < w; ++x) {
                    bool mine = nearest[hp[x]] == static_cast<int>(i);
                    sd[x] = f[x] && mine;
                    gr[x] = sd[x] || (wk[x] && mine && distance[i][hp[x]] <= s.hue_tolerance);
                }
            }
            // Keep only the grown components that contain a seed.
            cv::Mat grow_labels;
            int n_grow = cv::connectedComponents(grow, grow_labels, 8, CV_32S);
            std::vector<char> seeded(n_grow, 0);
            for (int y = 0; y < h; ++y) {
                const uchar* sd = seeds.ptr<uchar>(y);
                const int* l = grow_labels.ptr<int>(y);
                for (int x = 0; x < w; ++x)
                    if (sd[x]) seeded[l[x]] = 1;
            }
            cv::Mat layer = cv::Mat::zeros(h, w, CV_8U);
            for (int y = 0; y < h; ++y) {
                const int* l = grow_labels.ptr<int>(y);
                uchar* d = layer.ptr<uchar>(y);
                for (int x = 0; x < w; ++x) d[x] = l[x] > 0 && seeded[l[x]];
            }
            // A wide closing bridges printed logos and glare that notch the outline.
            cv::morphologyEx(layer, layer, cv::MORPH_CLOSE, close_kernel);
            cv::morphologyEx(layer, layer, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));
            std::vector<cv::Mat> parts;
            if (s.split_touching)
                parts = split_touching(layer, min_area);
            else if (cv::countNonZero(layer) >= min_area)
                parts.push_back(layer);
            for (cv::Mat part : parts) {
                // Close small holes (printed logos, glare) but keep a ring's real hole.
                part = close_small_holes(part, 0.02);
                if (cv::countNonZero(part) < min_area) continue;
                Piece piece;
                piece.features = describe(part);
                std::tie(piece.label, piece.confidence) = classify(piece.features, piece.scores);
                piece.mask = part;
                piece.bbox = cv::boundingRect(part);
                cv::Moments m = cv::moments(part, true);
                piece.center = {m.m10 / m.m00, m.m01 / m.m00};
                piece.hue = centers[i];
                pieces.push_back(std::move(piece));
            }
        }
    }
    std::stable_sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) {
        return a.label != b.label ? a.label < b.label : a.center.y < b.center.y;
    });
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return {std::move(pieces), seconds, centers};
}

// ------------------------------------------------------------------ output ----

cv::Mat annotate(const cv::Mat& image, const std::vector<Piece>& pieces) {
    cv::Mat overlay = image.clone(), out;
    for (const auto& p : pieces) overlay.setTo(class_bgr(p.label), p.mask);
    cv::addWeighted(overlay, 0.35, image, 0.65, 0, out);
    for (const auto& p : pieces) {
        cv::Scalar color = class_bgr(p.label);
        std::vector<Contour> contours;
        cv::findContours(p.mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
        cv::drawContours(out, contours, -1, color, 2, cv::LINE_AA);
        const cv::Rect& b = p.bbox;
        cv::rectangle(out, cv::Point(b.x, b.y), cv::Point(b.x + b.width, b.y + b.height), color, 1, cv::LINE_AA);
        char text[64];
        std::snprintf(text, sizeof(text), "%s %.2f", p.label.c_str(), p.confidence);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 0.6, 1, &baseline);
        int ty = b.y - ts.height - 10 > 0 ? b.y - 6 : b.y + b.height + ts.height + 6;
        cv::rectangle(out, cv::Point(b.x, ty - ts.height - 4), cv::Point(b.x + ts.width + 8, ty + 4), color, -1);
        cv::putText(out, text, cv::Point(b.x + 4, ty), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
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

// round(v, digits) as in Python.
double round_to(double v, int digits) {
    double f = std::pow(10.0, digits);
    return std::nearbyint(v * f) / f;
}

std::string to_json(const ObjectReport& report) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < report.pieces.size(); ++i) {
        const Piece& p = report.pieces[i];
        const Features& f = p.features;
        o << (i ? "," : "") << "\n  {\"label\": \"" << p.label << "\", \"confidence\": " << json_number(round_to(p.confidence, 3))
          << ", \"bbox\": [" << p.bbox.x << ", " << p.bbox.y << ", " << p.bbox.width << ", " << p.bbox.height << "]"
          << ", \"center\": [" << json_number(round_to(p.center.x, 1)) << ", " << json_number(round_to(p.center.y, 1)) << "]"
          << ", \"area\": " << static_cast<long>(f.area) << ", \"hue\": " << json_number(round_to(p.hue, 1))
          << ",\n   \"features\": {\"area\": " << json_number(f.area) << ", \"hole_ratio\": " << json_number(round_to(f.hole_ratio, 4))
          << ", \"hole_centered\": " << json_number(round_to(f.hole_centered, 4)) << ", \"rect_fill\": " << json_number(round_to(f.rect_fill, 4))
          << ", \"triangle_fill\": " << json_number(round_to(f.triangle_fill, 4))
          << ", \"triangularity\": " << json_number(round_to(f.triangularity, 4)) << ", \"solidity\": " << json_number(round_to(f.solidity, 4))
          << ", \"elongation\": " << json_number(round_to(f.elongation, 4)) << "}"
          << ",\n   \"scores\": {\"cone\": " << json_number(round_to(p.scores[0], 3)) << ", \"cube\": " << json_number(round_to(p.scores[1], 3))
          << ", \"ring\": " << json_number(round_to(p.scores[2], 3)) << "}}";
    }
    o << (report.pieces.empty() ? "]\n" : "\n]\n");
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
    std::fprintf(stderr, "usage: %s IMAGE [--out FILE] [--json FILE] [--no-show]\n", prog);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path, out_path, json_path;
    bool show = true;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--out") out_path = value();
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

    ObjectReport report = detect_objects(image);
    std::map<std::string, int> counts;
    for (const auto& p : report.pieces) ++counts[p.label];
    std::printf("%s: %dx%d px, %zu pieces (%.3f s), hue clusters:", image_path.c_str(), image.cols, image.rows,
                report.pieces.size(), report.seconds);
    for (double hue : report.hues) std::printf(" %.1f", hue);
    std::printf("\n");
    for (const auto& p : report.pieces)
        std::printf("  %-7s %.2f  bbox (%d, %d, %d, %d)  center (%.1f, %.1f)\n", p.label.c_str(), p.confidence, p.bbox.x,
                    p.bbox.y, p.bbox.width, p.bbox.height, p.center.x, p.center.y);

    cv::Mat annotated = annotate(image, report.pieces);
    if (!out_path.empty()) {
        cv::imwrite(out_path, annotated);
        std::printf("wrote %s\n", out_path.c_str());
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << to_json(report);
        std::printf("wrote %s\n", json_path.c_str());
    }
    if (show && has_display()) {
        cv::imshow("objects", annotated);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
