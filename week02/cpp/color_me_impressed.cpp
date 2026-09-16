// Color Me Impressed (C++17 port of week02/colors.py "named" mode and week02/shapes.py).
//
// Splits an image into one layer per named HSV color band, finds every
// connected object of each color, reports its center (mean of its pixel
// coordinates) and classifies its shape by IoU template matching.
//
// Usage: color_me_impressed IMAGE [--out DIR] [--json FILE] [--no-show] [--min-coverage F]

#include <opencv2/opencv.hpp>
// OpenCV 5 moved moments, contourArea and minAreaRect out of imgproc into the
// new geometry module, which opencv.hpp does not include. OpenCV 4 has no such
// header, so the include is guarded.
#if CV_VERSION_MAJOR >= 5 && __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

const double kPi = 3.14159265358979323846;

// ------------------------------------------------------------ color bands ----

// OpenCV stores hue as 0..179 (degrees / 2), saturation and value as 0..255.
const int BLACK_MAX_V = 50;
const int ACHROMATIC_MAX_S = 50;
const int WHITE_MIN_V = 190;
const int BROWN_MAX_V = 150;

struct HsvRange {
    cv::Scalar lower, upper;
};

struct ColorBand {
    std::string name;
    std::vector<HsvRange> ranges;
    cv::Scalar swatch_bgr;
};

HsvRange hue(int lo, int hi, int v_lo = BLACK_MAX_V + 1, int v_hi = 255) {
    return {cv::Scalar(lo, ACHROMATIC_MAX_S + 1, v_lo), cv::Scalar(hi, 255, v_hi)};
}

// Same bands, same order as COLOR_BANDS in colors.py. Together they partition
// HSV space, so every pixel lands in exactly one layer.
const std::vector<ColorBand>& color_bands() {
    static const std::vector<ColorBand> bands = {
        // Red wraps around the hue circle, so it needs two ranges.
        {"red", {hue(0, 8), hue(170, 179)}, cv::Scalar(40, 40, 220)},
        // Brown is dark orange / dark yellow, so it is carved out by brightness.
        {"brown", {hue(9, 30, BLACK_MAX_V + 1, BROWN_MAX_V)}, cv::Scalar(40, 90, 140)},
        {"orange", {hue(9, 20, BROWN_MAX_V + 1)}, cv::Scalar(20, 140, 255)},
        {"yellow", {hue(21, 30, BROWN_MAX_V + 1), hue(31, 34)}, cv::Scalar(40, 220, 240)},
        {"green", {hue(35, 85)}, cv::Scalar(60, 190, 60)},
        {"cyan", {hue(86, 100)}, cv::Scalar(220, 210, 40)},
        {"blue", {hue(101, 130)}, cv::Scalar(220, 110, 30)},
        {"purple", {hue(131, 150)}, cv::Scalar(200, 60, 140)},
        {"pink", {hue(151, 169)}, cv::Scalar(180, 110, 250)},
        {"black", {{cv::Scalar(0, 0, 0), cv::Scalar(179, 255, BLACK_MAX_V)}}, cv::Scalar(30, 30, 30)},
        {"gray",
         {{cv::Scalar(0, 0, BLACK_MAX_V + 1), cv::Scalar(179, ACHROMATIC_MAX_S, WHITE_MIN_V - 1)}},
         cv::Scalar(140, 140, 140)},
        {"white", {{cv::Scalar(0, 0, WHITE_MIN_V), cv::Scalar(179, ACHROMATIC_MAX_S, 255)}}, cv::Scalar(245, 245, 245)},
    };
    return bands;
}

// ------------------------------------------------------------------ shapes ----

const int CANVAS = 128;
const double REFERENCE_RADIUS = 44.0;
const double REFERENCE_AREA = kPi * REFERENCE_RADIUS * REFERENCE_RADIUS;
const double MIN_CONFIDENCE = 0.80;
const int ROTATION_STEPS = 36;

using Polygon = std::vector<cv::Point2d>;

Polygon regular(int sides) {
    Polygon p;
    for (int i = 0; i < sides; ++i) {
        double a = static_cast<double>(i * 2) * kPi / sides;
        p.emplace_back(std::cos(a), std::sin(a));
    }
    return p;
}

Polygon star(int points = 5, double inner = 0.45) {
    Polygon p;
    for (int i = 0; i < points * 2; ++i) {
        double a = static_cast<double>(i) * kPi / points;
        double r = (i % 2 == 0) ? 1.0 : inner;
        p.emplace_back(r * std::cos(a), r * std::sin(a));
    }
    return p;
}

Polygon cross(double a = 0.36) {
    return {{-a, -1}, {a, -1}, {a, -a}, {1, -a}, {1, a}, {a, a}, {a, 1}, {-a, 1}, {-a, a}, {-1, a}, {-1, -a}, {-a, -a}};
}

// np.linspace(start, stop, num): step multiples plus an exact endpoint.
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> v(num);
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) v[i] = i * step + start;
    v[num - 1] = stop;
    return v;
}

Polygon semicircle() {
    Polygon p;
    for (double a : linspace(0, kPi, 64)) p.emplace_back(std::cos(a), -std::sin(a));
    return p;
}

Polygon quarter_circle() {
    Polygon p{{0.0, 0.0}};
    for (double a : linspace(0, kPi / 2, 48)) p.emplace_back(std::cos(a), std::sin(a));
    return p;
}

Polygon trapezoid() { return {{-0.5, -0.6}, {0.5, -0.6}, {1.0, 0.6}, {-1.0, 0.6}}; }

Polygon rectangle(double aspect) { return {{-aspect, -1}, {aspect, -1}, {aspect, 1}, {-aspect, 1}}; }

struct Template {
    std::string name;
    Polygon points;
    double period;  // rotational symmetry period in radians
};

const std::vector<Template>& templates() {
    static const std::vector<Template> t = {
        {"circle", regular(96), 2 * kPi / 96},
        {"semicircle", semicircle(), 2 * kPi},
        {"quarter circle", quarter_circle(), 2 * kPi},
        {"triangle", regular(3), 2 * kPi / 3},
        {"square", regular(4), 2 * kPi / 4},
        {"trapezoid", trapezoid(), 2 * kPi},
        {"pentagon", regular(5), 2 * kPi / 5},
        {"hexagon", regular(6), 2 * kPi / 6},
        {"heptagon", regular(7), 2 * kPi / 7},
        {"octagon", regular(8), 2 * kPi / 8},
        {"star", star(), 2 * kPi / 5},
        {"cross", cross(), 2 * kPi / 4},
    };
    return t;
}

// Center a polygon on its centroid and scale it to the reference area, so
// templates and blobs are compared independent of position and size.
Polygon normalize_polygon(const Polygon& points) {
    std::vector<cv::Point2f> pts32(points.begin(), points.end());
    cv::Moments m = cv::moments(pts32);
    double area = std::abs(m.m00);
    double cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    double s = std::sqrt(REFERENCE_AREA / area);
    Polygon out;
    for (const auto& p : points) out.emplace_back((p.x - cx) * s, (p.y - cy) * s);
    return out;
}

cv::Mat rasterize(const Polygon& points, double angle) {
    double c = std::cos(angle), s = std::sin(angle);
    std::vector<cv::Point> fixed;
    fixed.reserve(points.size());
    for (const auto& p : points) {
        double x = p.x * c - p.y * s + CANVAS / 2.0;
        double y = p.x * s + p.y * c + CANVAS / 2.0;
        // nearbyint rounds half to even like np.round.
        fixed.emplace_back(static_cast<int>(std::nearbyint(x * 16)), static_cast<int>(std::nearbyint(y * 16)));
    }
    cv::Mat canvas = cv::Mat::zeros(CANVAS, CANVAS, CV_8U);
    std::vector<std::vector<cv::Point>> polys{fixed};
    cv::fillPoly(canvas, polys, cv::Scalar(1), cv::LINE_8, 4);
    return canvas;
}

struct Bank {
    std::vector<cv::Mat> rasters;  // 0/1 CV_8U, CANVAS x CANVAS, continuous
    std::vector<double> sums;
    std::vector<double> angles;
};

Bank make_bank(const Polygon& normalized, double period, int steps) {
    Bank bank;
    for (int i = 0; i < steps; ++i) {
        double a = static_cast<double>(i) * period / steps;
        cv::Mat r = rasterize(normalized, a);
        bank.sums.push_back(cv::countNonZero(r));
        bank.rasters.push_back(r);
        bank.angles.push_back(a);
    }
    return bank;
}

// Rasterized template banks are fixed, so build them once.
const std::vector<Bank>& template_banks() {
    static const std::vector<Bank> banks = [] {
        std::vector<Bank> b;
        for (const auto& t : templates()) b.push_back(make_bank(normalize_polygon(t.points), t.period, ROTATION_STEPS));
        return b;
    }();
    return banks;
}

std::pair<double, double> best_iou(const cv::Mat& blob, const Bank& bank) {
    const uchar* bp = blob.ptr<uchar>(0);
    const int n = CANVAS * CANVAS;
    double blob_sum = cv::countNonZero(blob);
    double best = -1.0, best_angle = 0.0;
    for (size_t k = 0; k < bank.rasters.size(); ++k) {
        const uchar* tp = bank.rasters[k].ptr<uchar>(0);
        long inter = 0;
        for (int i = 0; i < n; ++i) inter += bp[i] & tp[i];
        double uni = bank.sums[k] + blob_sum - inter;
        double iou = inter / std::max(uni, 1.0);
        if (iou > best) {  // strict: first maximum wins like np.argmax
            best = iou;
            best_angle = bank.angles[k];
        }
    }
    return {best, best_angle};
}

// Warp a binary blob into the canonical canvas (centroid centered, reference area).
std::optional<cv::Mat> normalize_blob(const cv::Mat& mask) {
    cv::Moments m = cv::moments(mask, true);
    if (m.m00 < 1) return std::nullopt;
    double cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    double scale = std::sqrt(REFERENCE_AREA / m.m00);
    cv::Matx23d M(scale, 0, CANVAS / 2.0 - scale * cx, 0, scale, CANVAS / 2.0 - scale * cy);
    cv::Mat f, warped;
    mask.convertTo(f, CV_32F);
    int interpolation = scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::warpAffine(f, warped, M, cv::Size(CANVAS, CANVAS), interpolation);
    cv::Mat blob = (warped > 127) / 255;
    return blob;
}

struct ShapeMatch {
    std::string shape;
    double confidence;
};

double round_to(double v, int digits) {
    double f = std::pow(10.0, digits);
    return std::nearbyint(v * f) / f;
}

// Classify a single 0/255 blob. Holes (such as the target letter) are ignored.
std::optional<ShapeMatch> classify(const cv::Mat& mask_in, int min_area) {
    cv::Mat mask = (mask_in > 0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return std::nullopt;
    size_t best_idx = 0;
    double best_area = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > best_area) {
            best_area = a;
            best_idx = i;
        }
    }
    if (best_area < min_area) return std::nullopt;
    const auto& contour = contours[best_idx];
    cv::Mat solid = cv::Mat::zeros(mask.size(), CV_8U);
    cv::drawContours(solid, contours, static_cast<int>(best_idx), cv::Scalar(255), cv::FILLED);
    auto blob = normalize_blob(solid);
    if (!blob) return std::nullopt;

    struct Score {
        double iou;
        std::string name;
        double angle;
    };
    std::vector<Score> scores;
    const auto& tpl = templates();
    const auto& banks = template_banks();
    for (size_t t = 0; t < tpl.size(); ++t) {
        auto [iou, angle] = best_iou(*blob, banks[t]);
        scores.push_back({iou, tpl[t].name, angle});
    }

    // Real rectangles come in many proportions, so the template uses the
    // blob's own aspect ratio from its minimum area rectangle.
    cv::RotatedRect rect = cv::minAreaRect(contour);
    double rw = rect.size.width, rh = rect.size.height;
    double aspect = std::min(rw, rh) / std::max({rw, rh, 1e-6});
    if (aspect < 0.85) {
        Bank bank = make_bank(normalize_polygon(rectangle(aspect)), kPi, 90);
        auto [iou, angle] = best_iou(*blob, bank);
        scores.push_back({iou, "rectangle", angle});
    }

    // Python sorts (iou, name, angle) tuples in reverse; mirror the tie breaks.
    std::sort(scores.begin(), scores.end(), [](const Score& a, const Score& b) {
        return std::tie(a.iou, a.name, a.angle) > std::tie(b.iou, b.name, b.angle);
    });
    std::string name = scores[0].iou < MIN_CONFIDENCE ? "irregular" : scores[0].name;
    return ShapeMatch{name, round_to(scores[0].iou, 4)};
}

// ------------------------------------------------------------------ layers ----

struct ColorObject {
    cv::Point2d center;
    int area;
    cv::Rect bbox;
    std::optional<std::string> shape;
    double shape_confidence = 0.0;
};

struct ColorLayer {
    std::string name;
    cv::Mat mask;
    int pixels;
    double coverage;
    std::optional<cv::Point2d> center;
    std::vector<ColorObject> objects;
};

// Center of mass via the mean of pixel coordinates (np.where + np.mean).
std::optional<cv::Point2d> mask_center(const cv::Mat& mask) {
    int64_t sx = 0, sy = 0, n = 0;
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x]) {
                sx += x;
                sy += y;
                ++n;
            }
        }
    }
    if (n == 0) return std::nullopt;
    return cv::Point2d(static_cast<double>(sx) / n, static_cast<double>(sy) / n);
}

std::vector<ColorObject> find_objects(const cv::Mat& mask, int min_area, int min_shape_area = 150) {
    // A small opening removes speckle (JPEG noise, anti aliased edges) so a
    // single sign is reported as one object, not fifty.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat cleaned;
    cv::morphologyEx(mask, cleaned, cv::MORPH_OPEN, kernel);

    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8, CV_32S);

    // One pass accumulates exact integer coordinate sums for every label.
    std::vector<int64_t> sum_x(count, 0), sum_y(count, 0), sum_n(count, 0);
    for (int y = 0; y < labels.rows; ++y) {
        const int* row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int l = row[x];
            if (l > 0) {
                sum_x[l] += x;
                sum_y[l] += y;
                ++sum_n[l];
            }
        }
    }

    std::vector<ColorObject> objects;
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area) continue;
        int x = stats.at<int>(label, cv::CC_STAT_LEFT), y = stats.at<int>(label, cv::CC_STAT_TOP);
        int w = stats.at<int>(label, cv::CC_STAT_WIDTH), h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        ColorObject obj;
        obj.center = {static_cast<double>(sum_x[label]) / sum_n[label], static_cast<double>(sum_y[label]) / sum_n[label]};
        obj.area = area;
        obj.bbox = {x, y, w, h};
        bool touches_border = x == 0 || y == 0 || x + w >= mask.cols || y + h >= mask.rows;
        // A region cut off by the frame edge has no meaningful shape.
        if (area >= min_shape_area && !touches_border) {
            const int pad = 4;
            int x0 = std::max(0, x - pad), y0 = std::max(0, y - pad);
            int x1 = std::min(mask.cols, x + w + pad), y1 = std::min(mask.rows, y + h + pad);
            cv::Mat crop = labels(cv::Rect(x0, y0, x1 - x0, y1 - y0)) == label;
            auto match = classify(crop, min_shape_area);
            if (match && match->shape != "irregular") {
                obj.shape = match->shape;
                obj.shape_confidence = match->confidence;
            }
        }
        objects.push_back(obj);
    }
    std::stable_sort(objects.begin(), objects.end(), [](const ColorObject& a, const ColorObject& b) { return a.area > b.area; });
    return objects;
}

std::vector<ColorLayer> split_colors(const cv::Mat& image, double min_coverage, double min_object_fraction = 0.0008) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    const double total = static_cast<double>(image.rows) * image.cols;
    int min_area = std::max(1, static_cast<int>(std::nearbyint(total * min_object_fraction)));

    std::vector<ColorLayer> layers;
    for (const auto& band : color_bands()) {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
        for (const auto& r : band.ranges) {
            cv::Mat part;
            cv::inRange(hsv, r.lower, r.upper, part);
            mask |= part;
        }
        int pixels = cv::countNonZero(mask);
        double coverage = pixels / total;
        if (pixels < min_area) continue;
        auto objects = find_objects(mask, min_area);
        // Keep small colors when they form a real object (such as a letter).
        if (coverage < min_coverage && objects.empty()) continue;
        layers.push_back({band.name, mask, pixels, coverage, mask_center(mask), std::move(objects)});
    }
    std::stable_sort(layers.begin(), layers.end(), [](const ColorLayer& a, const ColorLayer& b) { return a.pixels > b.pixels; });
    return layers;
}

// ------------------------------------------------------------------ output ----

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    return buffer;
}

std::string format_report(const std::vector<ColorLayer>& layers) {
    std::string out;
    for (size_t li = 0; li < layers.size(); ++li) {
        const auto& layer = layers[li];
        if (li) out += "\n";
        out += format("%-7s %5.1f%% of pixels", layer.name.c_str(), layer.coverage * 100);
        if (layer.center) out += format(", overall center (x=%.1f, y=%.1f)", layer.center->x, layer.center->y);
        if (layer.objects.empty()) out += "\n        no objects above the size threshold";
        for (size_t i = 0; i < layer.objects.size(); ++i) {
            const auto& obj = layer.objects[i];
            out += format("\n        object %zu: center (x=%.1f, y=%.1f), area %d px", i + 1, obj.center.x, obj.center.y, obj.area);
            if (obj.shape) out += format(", %s (%.2f IoU)", obj.shape->c_str(), obj.shape_confidence);
        }
    }
    return out;
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

std::string to_json(const std::vector<ColorLayer>& layers) {
    std::ostringstream o;
    o << "[";
    for (size_t li = 0; li < layers.size(); ++li) {
        const auto& l = layers[li];
        o << (li ? ",\n" : "\n") << "  {\n";
        o << "    \"name\": " << json_string(l.name) << ",\n";
        o << "    \"pixels\": " << l.pixels << ",\n";
        o << "    \"coverage\": " << json_number(l.coverage) << ",\n";
        o << "    \"center\": ";
        if (l.center)
            o << "[" << json_number(l.center->x) << ", " << json_number(l.center->y) << "]";
        else
            o << "null";
        o << ",\n    \"objects\": [";
        for (size_t i = 0; i < l.objects.size(); ++i) {
            const auto& obj = l.objects[i];
            o << (i ? "," : "") << "\n      {\"center\": [" << json_number(obj.center.x) << ", " << json_number(obj.center.y)
              << "], \"area\": " << obj.area << ", \"bbox\": [" << obj.bbox.x << ", " << obj.bbox.y << ", " << obj.bbox.width
              << ", " << obj.bbox.height << "], \"shape\": " << (obj.shape ? json_string(*obj.shape) : "null")
              << ", \"shape_confidence\": " << (obj.shape ? json_number(obj.shape_confidence) : "null") << "}";
        }
        o << (l.objects.empty() ? "]" : "\n    ]") << "\n  }";
    }
    o << (layers.empty() ? "]\n" : "\n]\n");
    return o.str();
}

cv::Scalar swatch_for(const std::string& name) {
    for (const auto& band : color_bands())
        if (band.name == name) return band.swatch_bgr;
    return cv::Scalar(0, 255, 0);
}

// Crosshair and label on every object center.
cv::Mat annotate(const cv::Mat& image, const std::vector<ColorLayer>& layers) {
    cv::Mat out = image.clone();
    double scale = std::max(image.rows, image.cols) / 800.0;
    int thickness = std::max(1, static_cast<int>(std::nearbyint(2 * scale)));
    for (const auto& layer : layers) {
        cv::Scalar swatch = swatch_for(layer.name);
        for (const auto& obj : layer.objects) {
            int cx = static_cast<int>(std::nearbyint(obj.center.x)), cy = static_cast<int>(std::nearbyint(obj.center.y));
            int size = static_cast<int>(14 * scale) + 4;
            cv::drawMarker(out, {cx, cy}, cv::Scalar(0, 0, 0), cv::MARKER_CROSS, size + 4, thickness + 2);
            cv::drawMarker(out, {cx, cy}, swatch, cv::MARKER_CROSS, size, thickness);
            std::string label = obj.shape ? format("%s %s (%d,%d)", layer.name.c_str(), obj.shape->c_str(), cx, cy)
                                          : format("%s (%d,%d)", layer.name.c_str(), cx, cy);
            cv::Point org(cx + size / 2 + 4, cy - size / 2);
            cv::putText(out, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
            cv::putText(out, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }
    return out;
}

bool has_display() {
#if defined(__APPLE__) || defined(_WIN32)
    return true;
#else
    return std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY");
#endif
}

int usage(const char* prog) {
    std::fprintf(stderr, "usage: %s IMAGE [--out DIR] [--json FILE] [--no-show] [--min-coverage F]\n", prog);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path, out_dir, json_path;
    bool show = true;
    double min_coverage = 0.01;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--out") out_dir = value();
            else if (arg == "--json") json_path = value();
            else if (arg == "--no-show") show = false;
            else if (arg == "--min-coverage") min_coverage = std::stod(value());
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

    auto layers = split_colors(image, min_coverage);
    std::printf("%s: %dx%d px, %zu colors (named mode)\n\n", image_path.c_str(), image.cols, image.rows, layers.size());
    std::printf("%s\n", format_report(layers).c_str());

    if (!out_dir.empty()) {
        fs::create_directories(out_dir);
        size_t written = 0;
        for (const auto& layer : layers) {
            // BGRA cut out: alpha is the color mask.
            cv::Mat bgra;
            cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
            cv::Mat channels[] = {cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat()};
            cv::split(bgra, channels);
            channels[3] = layer.mask;
            cv::merge(channels, 4, bgra);
            cv::imwrite((fs::path(out_dir) / ("layer_" + layer.name + ".png")).string(), bgra);
            ++written;
        }
        cv::imwrite((fs::path(out_dir) / "centers.png").string(), annotate(image, layers));
        ++written;
        std::printf("\nwrote %zu files to %s\n", written, out_dir.c_str());
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << to_json(layers);
        std::printf("wrote %s\n", json_path.c_str());
    }
    if (show && has_display()) {
        cv::imshow("original", image);
        cv::imshow("object centers", annotate(image, layers));
        for (const auto& layer : layers) {
            cv::Mat isolated;
            cv::bitwise_and(image, image, isolated, layer.mask);
            cv::imshow("color: " + layer.name, isolated);
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
