/*
 * Color Me Impressed, browser port of week02/colors.py ("named" mode).
 *
 * Every step mirrors the Python/OpenCV implementation:
 *   1. RGB to HSV with OpenCV's 8-bit fixed point formula (H 0..179, S and V 0..255).
 *   2. One mask per named band, using the exact inclusive ranges of COLOR_BANDS.
 *   3. Opening with the 3x3 elliptical kernel (a plus shape), OpenCV border rules.
 *   4. 8-connected components, centers as the mean pixel coordinates.
 *   5. min_object_fraction 0.0008 and min_coverage 0.01 filtering.
 * Shape classification (shapes.py) is not ported.
 *
 * Works as a browser global (window.HsvSegment) and as a CommonJS module.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.HsvSegment = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const BLACK_MAX_V = 50;
  const ACHROMATIC_MAX_S = 50;
  const WHITE_MIN_V = 190;
  const BROWN_MAX_V = 150;

  // [[hLo, sLo, vLo], [hHi, sHi, vHi]], inclusive, like cv2.inRange.
  function hue(lo, hi, vLo = BLACK_MAX_V + 1, vHi = 255) {
    return [[lo, ACHROMATIC_MAX_S + 1, vLo], [hi, 255, vHi]];
  }

  // Same order, ranges and swatches (as RGB) as week02/colors.py COLOR_BANDS.
  const COLOR_BANDS = [
    { name: "red", ranges: [hue(0, 8), hue(170, 179)], swatch: [220, 40, 40] },
    { name: "brown", ranges: [hue(9, 30, BLACK_MAX_V + 1, BROWN_MAX_V)], swatch: [140, 90, 40] },
    { name: "orange", ranges: [hue(9, 20, BROWN_MAX_V + 1)], swatch: [255, 140, 20] },
    { name: "yellow", ranges: [hue(21, 30, BROWN_MAX_V + 1), hue(31, 34)], swatch: [240, 220, 40] },
    { name: "green", ranges: [hue(35, 85)], swatch: [60, 190, 60] },
    { name: "cyan", ranges: [hue(86, 100)], swatch: [40, 210, 220] },
    { name: "blue", ranges: [hue(101, 130)], swatch: [30, 110, 220] },
    { name: "purple", ranges: [hue(131, 150)], swatch: [140, 60, 200] },
    { name: "pink", ranges: [hue(151, 169)], swatch: [250, 110, 180] },
    { name: "black", ranges: [[[0, 0, 0], [179, 255, BLACK_MAX_V]]], swatch: [30, 30, 30] },
    { name: "gray", ranges: [[[0, 0, BLACK_MAX_V + 1], [179, ACHROMATIC_MAX_S, WHITE_MIN_V - 1]]], swatch: [140, 140, 140] },
    { name: "white", ranges: [[[0, 0, WHITE_MIN_V], [179, ACHROMATIC_MAX_S, 255]]], swatch: [245, 245, 245] },
  ];

  // ------------------------------------------------------------ HSV ----
  // OpenCV RGB2HSV_b: fixed point with hsv_shift = 12 and per value division
  // tables. OpenCV's saturation table is one lower than a rounded division
  // for six values; the corrections below make this function agree with
  // cv2.cvtColor(COLOR_BGR2HSV) on all 16,777,216 RGB colors for the macOS
  // arm64 wheels (OpenCV 4.14 and 5.0). Linux x86 builds round a few thousand
  // saturations differently, by one step at most.
  const HSV_SHIFT = 12;
  const SDIV = new Int32Array(256);
  const HDIV = new Int32Array(256);
  for (let i = 1; i < 256; i++) {
    SDIV[i] = Math.round((255 << HSV_SHIFT) / i);
    HDIV[i] = Math.round((180 << HSV_SHIFT) / (6 * i));
  }
  for (const i of [2, 4, 14, 18, 118, 209]) SDIV[i] -= 1;
  const HALF = 1 << (HSV_SHIFT - 1);

  function rgbToHsv(r, g, b) {
    let v = b;
    let vmin = b;
    if (g > v) v = g;
    if (r > v) v = r;
    if (g < vmin) vmin = g;
    if (r < vmin) vmin = r;
    const diff = v - vmin;
    const vr = v === r ? -1 : 0;
    const vg = v === g ? -1 : 0;
    const s = (diff * SDIV[v] + HALF) >> HSV_SHIFT;
    let h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))));
    h = (h * HDIV[diff] + HALF) >> HSV_SHIFT;
    if (h < 0) h += 180;
    return [h, s, v];
  }

  function inBand(band, h, s, v) {
    for (const [lo, hi] of band.ranges) {
      if (h >= lo[0] && h <= hi[0] && s >= lo[1] && s <= hi[1] && v >= lo[2] && v <= hi[2]) return true;
    }
    return false;
  }

  // Band index for every (h, s, v), 180 * 256 * 256 entries built once by
  // filling each inclusive range box. Bands are painted last to first so the
  // first matching band wins; the default bands are an exact partition anyway.
  let lookup = null;
  function bandLookup() {
    if (lookup) return lookup;
    lookup = new Uint8Array(180 * 256 * 256).fill(255);
    for (let k = COLOR_BANDS.length - 1; k >= 0; k--) {
      for (const [lo, hi] of COLOR_BANDS[k].ranges) {
        for (let h = lo[0]; h <= hi[0]; h++) {
          for (let s = lo[1]; s <= hi[1]; s++) {
            const base = (h * 256 + s) * 256;
            lookup.fill(k, base + lo[2], base + hi[2] + 1);
          }
        }
      }
    }
    return lookup;
  }

  function bandIndexOf(h, s, v) {
    return bandLookup()[(h * 256 + s) * 256 + v];
  }

  // Per pixel band label from RGBA bytes.
  function labelPixels(rgba, width, height) {
    const table = bandLookup();
    const n = width * height;
    const labels = new Uint8Array(n);
    // Same arithmetic as rgbToHsv, inlined to avoid an allocation per pixel.
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      let v = b;
      let vmin = b;
      if (g > v) v = g;
      if (r > v) v = r;
      if (g < vmin) vmin = g;
      if (r < vmin) vmin = r;
      const diff = v - vmin;
      const vr = v === r ? -1 : 0;
      const vg = v === g ? -1 : 0;
      const s = (diff * SDIV[v] + HALF) >> HSV_SHIFT;
      let h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))));
      h = (h * HDIV[diff] + HALF) >> HSV_SHIFT;
      if (h < 0) h += 180;
      labels[i] = table[(h * 256 + s) * 256 + v];
    }
    return labels;
  }

  // ------------------------------------------------------ morphology ----
  // Kernel getStructuringElement(MORPH_ELLIPSE, (3, 3)) is a plus shape.
  // OpenCV's default morphology border ignores pixels outside the image.
  function erodePlus(src, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (!src[i]) continue;
        if (x > 0 && !src[i - 1]) continue;
        if (x < w - 1 && !src[i + 1]) continue;
        if (y > 0 && !src[i - w]) continue;
        if (y < h - 1 && !src[i + w]) continue;
        out[i] = 1;
      }
    }
    return out;
  }

  function dilatePlus(src, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (
          src[i] ||
          (x > 0 && src[i - 1]) ||
          (x < w - 1 && src[i + 1]) ||
          (y > 0 && src[i - w]) ||
          (y < h - 1 && src[i + w])
        ) {
          out[i] = 1;
        }
      }
    }
    return out;
  }

  // ------------------------------------------- connected components ----
  function connectedComponents(mask, w, h) {
    const n = w * h;
    const labels = new Int32Array(n);
    const parent = [0];
    const find = (a) => {
      while (parent[a] !== a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
      }
      return a;
    };
    const union = (a, b) => {
      a = find(a);
      b = find(b);
      if (a === b) return a;
      if (a < b) {
        parent[b] = a;
        return a;
      }
      parent[a] = b;
      return b;
    };
    let next = 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        // Already visited 8-neighbors: W, NW, N, NE.
        let label = x > 0 ? labels[i - 1] : 0;
        if (y > 0) {
          const nw = x > 0 ? labels[i - w - 1] : 0;
          const up = labels[i - w];
          const ne = x < w - 1 ? labels[i - w + 1] : 0;
          if (nw) label = label ? union(label, nw) : nw;
          if (up) label = label ? union(label, up) : up;
          if (ne) label = label ? union(label, ne) : ne;
        }
        if (!label) {
          label = next++;
          parent.push(label);
        }
        labels[i] = label;
      }
    }

    // Resolve roots and renumber in raster order (like OpenCV's labeling).
    const remap = new Int32Array(next);
    const stats = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!labels[i]) continue;
        const root = find(labels[i]);
        let id = remap[root];
        if (!id) {
          stats.push({ area: 0, sumX: 0, sumY: 0, minX: x, minY: y, maxX: x, maxY: y });
          id = remap[root] = stats.length;
        }
        labels[i] = id;
        const s = stats[id - 1];
        s.area++;
        s.sumX += x;
        s.sumY += y;
        if (x < s.minX) s.minX = x;
        if (x > s.maxX) s.maxX = x;
        if (y < s.minY) s.minY = y;
        if (y > s.maxY) s.maxY = y;
      }
    }
    return { labels, stats };
  }

  // Python's round(): halves go to the even neighbor.
  function roundHalfEven(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  function findObjects(mask, w, h, minArea) {
    const cleaned = dilatePlus(erodePlus(mask, w, h), w, h);
    const { stats } = connectedComponents(cleaned, w, h);
    const objects = [];
    for (const s of stats) {
      if (s.area < minArea) continue;
      const bw = s.maxX - s.minX + 1;
      const bh = s.maxY - s.minY + 1;
      objects.push({
        center: [s.sumX / s.area, s.sumY / s.area],
        area: s.area,
        bbox: [s.minX, s.minY, bw, bh],
        touchesBorder: s.minX === 0 || s.minY === 0 || s.minX + bw >= w || s.minY + bh >= h,
      });
    }
    // Stable sort, largest first, as in colors.find_objects.
    return objects.map((o, i) => [o, i]).sort((a, b) => b[0].area - a[0].area || a[1] - b[1]).map((p) => p[0]);
  }

  /**
   * Split an RGBA image into color layers, largest first.
   * Returns { width, height, minArea, labels, layers: [{ name, index, swatch, mask,
   * pixels, coverage, center, objects }] }.
   */
  function splitColors(rgba, width, height, options = {}) {
    const minCoverage = options.minCoverage ?? 0.01;
    const minObjectFraction = options.minObjectFraction ?? 0.0008;
    const total = width * height;
    const minArea = Math.max(1, roundHalfEven(total * minObjectFraction));
    const labels = labelPixels(rgba, width, height);

    const counts = new Float64Array(COLOR_BANDS.length);
    const sumX = new Float64Array(COLOR_BANDS.length);
    const sumY = new Float64Array(COLOR_BANDS.length);
    for (let y = 0, i = 0; y < height; y++) {
      for (let x = 0; x < width; x++, i++) {
        const k = labels[i];
        counts[k]++;
        sumX[k] += x;
        sumY[k] += y;
      }
    }

    const layers = [];
    COLOR_BANDS.forEach((band, k) => {
      const pixels = counts[k];
      if (pixels < minArea) return;
      const mask = new Uint8Array(total);
      for (let i = 0; i < total; i++) if (labels[i] === k) mask[i] = 1;
      const objects = findObjects(mask, width, height, minArea);
      const coverage = pixels / total;
      if (coverage < minCoverage && objects.length === 0) return;
      layers.push({
        name: band.name,
        index: k,
        swatch: band.swatch,
        mask,
        pixels,
        coverage,
        center: [sumX[k] / pixels, sumY[k] / pixels],
        objects,
      });
    });
    layers.sort((a, b) => b.pixels - a.pixels || a.index - b.index);
    return { width, height, minArea, labels, layers };
  }

  return { COLOR_BANDS, rgbToHsv, bandIndexOf, labelPixels, erodePlus, dilatePlus, connectedComponents, splitColors };
});
