/* Interactive Color Me Impressed demo. Algorithm lives in hsv-segment.js. */
(function () {
  "use strict";

  const S = window.HsvSegment;
  const root = document.getElementById("demo");
  if (!S || !root) return;

  const MAX_UPLOAD_SIDE = 1024;
  const THUMB_WIDTH = 360;
  const DEMOS = {
    targets: { src: "assets/demo/targets.jpg", label: "SUAS targets" },
    stop_sign: { src: "assets/demo/stop_sign.jpg", label: "Stop sign" },
    apple: { src: "assets/demo/apple.jpg", label: "Apple" },
  };

  const stage = document.getElementById("stage");
  const view = document.getElementById("view");
  const marks = document.getElementById("marks");
  const statusEl = document.getElementById("status");
  const layersEl = document.getElementById("layers");
  const objectsTable = document.getElementById("objects");
  const objectsHead = document.getElementById("objects-head");
  const objectsHint = document.getElementById("objects-hint");
  const viewLabel = document.getElementById("view-label");
  const showAll = document.getElementById("show-all");
  const upload = document.getElementById("upload");
  const demoButtons = [...root.querySelectorAll("[data-demo]")];

  let references = {};
  try {
    references = JSON.parse(document.getElementById("reference-reports").textContent);
  } catch (error) {
    references = {};
  }

  const state = { rgba: null, width: 0, height: 0, result: null, selected: null, reference: null, token: 0 };
  const fmt = new Intl.NumberFormat("en-US");
  const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

  // Python round(): used for the integer crosshair labels, like annotate().
  function roundHalfEven(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d !== 0.5) return Math.round(x);
    return f % 2 === 0 ? f : f + 1;
  }

  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

  // ------------------------------------------------------------ loading ----

  async function decode(blob, maxSide) {
    let source;
    try {
      source = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none", imageOrientation: "from-image" });
    } catch (error) {
      source = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("This file could not be decoded as an image."));
        img.src = URL.createObjectURL(blob);
      });
    }
    const w0 = source.width || source.naturalWidth;
    const h0 = source.height || source.naturalHeight;
    if (!w0 || !h0) throw new Error("The image has no pixels.");
    const scale = maxSide ? Math.min(1, maxSide / Math.max(w0, h0)) : 1;
    const width = Math.max(1, Math.round(w0 * scale));
    const height = Math.max(1, Math.round(h0 * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, 0, 0, width, height);
    if (source.close) source.close();
    return { rgba: ctx.getImageData(0, 0, width, height).data, width, height, scaled: scale < 1, originalWidth: w0, originalHeight: h0 };
  }

  async function run(loader, label, reference) {
    const token = ++state.token;
    stage.setAttribute("aria-busy", "true");
    statusEl.textContent = `Loading ${label}`;
    try {
      const image = await loader();
      if (token !== state.token) return;
      statusEl.textContent = `Splitting ${fmt.format(image.width * image.height)} px`;
      await nextFrame();
      const t0 = performance.now();
      const result = S.splitColors(image.rgba, image.width, image.height);
      const ms = performance.now() - t0;
      if (token !== state.token) return;
      Object.assign(state, { rgba: image.rgba, width: image.width, height: image.height, result, selected: null, reference });
      const count = result.layers.reduce((n, layer) => n + layer.objects.length, 0);
      const size = image.scaled
        ? `${image.originalWidth}x${image.originalHeight} scaled to ${image.width}x${image.height}`
        : `${image.width}x${image.height}`;
      statusEl.textContent = `${size}, ${result.layers.length} colors, ${count} objects, ${Math.round(ms)} ms`;
      renderLayers();
      renderView();
      renderObjects();
    } catch (error) {
      if (token === state.token) statusEl.textContent = error.message || "Something went wrong reading that image.";
    } finally {
      if (token === state.token) stage.setAttribute("aria-busy", "false");
    }
  }

  function loadDemo(name) {
    const demo = DEMOS[name];
    demoButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.demo === name)));
    run(
      async () => {
        const response = await fetch(demo.src);
        if (!response.ok) throw new Error(`Could not load ${demo.src}`);
        return decode(await response.blob(), 0);
      },
      demo.label,
      references[name] || null,
    );
  }

  function loadFile(file) {
    if (!file) return;
    if (file.type && !file.type.startsWith("image/")) {
      statusEl.textContent = "That file is not an image.";
      return;
    }
    demoButtons.forEach((b) => b.setAttribute("aria-pressed", "false"));
    run(() => decode(file, MAX_UPLOAD_SIDE), file.name || "your image", null);
  }

  // ---------------------------------------------------------- rendering ----

  function checkerboard(data, width, height, cell) {
    for (let y = 0, p = 0; y < height; y++) {
      for (let x = 0; x < width; x++, p += 4) {
        const v = ((Math.floor(y / cell) + Math.floor(x / cell)) % 2) === 1 ? 58 : 44;
        data[p] = data[p + 1] = data[p + 2] = v;
        data[p + 3] = 255;
      }
    }
  }

  // Layer on a checkerboard, like colors.render_layer.
  function layerImage(layer) {
    const { width, height, rgba } = state;
    const image = new ImageData(width, height);
    const out = image.data;
    checkerboard(out, width, height, Math.max(6, Math.round(Math.max(width, height) / 90)));
    const mask = layer.mask;
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
      if (mask[i]) {
        out[p] = rgba[p];
        out[p + 1] = rgba[p + 1];
        out[p + 2] = rgba[p + 2];
      }
    }
    return image;
  }

  function renderView() {
    const { width, height, result, selected } = state;
    view.width = width;
    view.height = height;
    const ctx = view.getContext("2d");
    const layer = selected === null ? null : result.layers[selected];
    ctx.putImageData(layer ? layerImage(layer) : new ImageData(new Uint8ClampedArray(state.rgba), width, height), 0, 0);
    showAll.hidden = !layer;
    viewLabel.textContent = layer
      ? `Showing only ${layer.name}: ${(layer.coverage * 100).toFixed(1)}% of pixels.`
      : "Showing the original with object centers. Drop an image here to try it.";
    drawMarks();
  }

  function drawMarks() {
    const { width, height, result, selected } = state;
    if (!result) return;
    marks.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const displayed = view.getBoundingClientRect().width || width;
    const unit = width / displayed; // image pixels per CSS pixel
    const fontSize = 11.5 * unit;
    const arm = 9 * unit;
    const layers = selected === null ? result.layers : [result.layers[selected]];
    const compact = displayed < 520; // narrow screens: crosshairs only, centers stay in the table
    const items = [];
    for (const layer of layers) {
      for (const obj of layer.objects) {
        if (obj.touchesBorder) continue; // background regions, as in annotate()
        items.push({ layer, obj, cx: obj.center[0], cy: obj.center[1] });
      }
    }

    // Greedy label placement: nudge down until it clears earlier labels.
    const placed = [];
    const ns = "http://www.w3.org/2000/svg";
    const frag = document.createDocumentFragment();
    items.sort((a, b) => a.cy - b.cy);
    for (const item of items) {
      const text = `${item.layer.name} (${roundHalfEven(item.cx)}, ${roundHalfEven(item.cy)})`;
      const w = text.length * fontSize * 0.66 + fontSize * 0.9;
      const h = fontSize * 1.7;
      let x = item.cx + arm + 4 * unit;
      if (x + w > width) x = item.cx - arm - 4 * unit - w;
      let y = item.cy - h / 2;
      for (let guard = 0; guard < 40; guard++) {
        const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
        if (!hit) break;
        y = hit.y + hit.h + 2 * unit;
      }
      y = Math.min(Math.max(0, y), height - h);
      placed.push({ x, y, w, h });

      const g = document.createElementNS(ns, "g");
      const cross = `M${item.cx - arm} ${item.cy}H${item.cx + arm}M${item.cx} ${item.cy - arm}V${item.cy + arm}`;
      const shadow = document.createElementNS(ns, "path");
      shadow.setAttribute("d", cross);
      shadow.setAttribute("class", "cross-shadow");
      shadow.setAttribute("stroke-width", String(4 * unit));
      const line = document.createElementNS(ns, "path");
      line.setAttribute("d", cross);
      line.setAttribute("class", "cross");
      line.style.stroke = rgb(item.layer.swatch);
      line.setAttribute("stroke-width", String(1.8 * unit));
      const bg = document.createElementNS(ns, "rect");
      bg.setAttribute("class", "label-bg");
      Object.entries({ x, y, width: w, height: h, rx: 3 * unit }).forEach(([k, v]) => bg.setAttribute(k, String(v)));
      const dot = document.createElementNS(ns, "circle");
      Object.entries({ cx: x + fontSize * 0.75, cy: y + h / 2, r: fontSize * 0.32 }).forEach(([k, v]) => dot.setAttribute(k, String(v)));
      dot.setAttribute("fill", rgb(item.layer.swatch));
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(x + fontSize * 1.35));
      label.setAttribute("y", String(y + h / 2));
      label.setAttribute("font-size", String(fontSize));
      label.textContent = text;
      g.append(shadow, line);
      if (!compact) g.append(bg, dot, label);
      frag.append(g);
    }
    marks.replaceChildren(frag);
  }

  function renderLayers() {
    const { result } = state;
    layersEl.replaceChildren();
    result.layers.forEach((layer, index) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "layer-button";
      button.setAttribute("aria-pressed", "false");
      const thumb = document.createElement("canvas");
      const scale = Math.min(1, THUMB_WIDTH / state.width);
      thumb.width = Math.round(state.width * scale);
      thumb.height = Math.round(state.height * scale);
      thumb.setAttribute("aria-hidden", "true");
      const full = document.createElement("canvas");
      full.width = state.width;
      full.height = state.height;
      full.getContext("2d").putImageData(layerImage(layer), 0, 0);
      const tctx = thumb.getContext("2d");
      tctx.imageSmoothingQuality = "high";
      tctx.drawImage(full, 0, 0, thumb.width, thumb.height);
      const meta = document.createElement("span");
      meta.className = "layer-meta";
      const name = document.createElement("b");
      const swatch = document.createElement("i");
      swatch.className = "swatch";
      swatch.style.background = rgb(layer.swatch);
      name.append(swatch, layer.name);
      const info = document.createElement("span");
      const objects = layer.objects.length;
      info.textContent = `${(layer.coverage * 100).toFixed(1)}%, ${objects} object${objects === 1 ? "" : "s"}`;
      meta.append(name, info);
      button.setAttribute("aria-label", `${layer.name} layer, ${(layer.coverage * 100).toFixed(1)} percent of pixels, ${objects} objects`);
      button.append(thumb, meta);
      button.addEventListener("click", () => select(state.selected === index ? null : index));
      li.append(button);
      layersEl.append(li);
    });
  }

  function select(index) {
    state.selected = index;
    [...layersEl.querySelectorAll(".layer-button")].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
    renderView();
    renderObjects();
  }

  // Match each browser object to the Python report object of the same color
  // with the nearest center.
  function referenceFor(layerName, obj) {
    const ref = state.reference && state.reference.layers.find((l) => l.name === layerName);
    if (!ref) return null;
    let best = null;
    for (const candidate of ref.objects) {
      const d = Math.hypot(candidate.center[0] - obj.center[0], candidate.center[1] - obj.center[1]);
      if (!best || d < best.d) best = { d, candidate };
    }
    return best;
  }

  function cell(tag, text, className) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function renderObjects() {
    const { result, selected, reference } = state;
    const tbody = objectsTable.tBodies[0];
    tbody.replaceChildren();
    const head = document.createElement("tr");
    const columns = reference
      ? [["Color", ""], ["Center (JS)", "num"], ["Python", "num py"], ["Offset", "num"]]
      : [["Color", ""], ["Center", "num"], ["Area (px)", "num"]];
    columns.forEach(([label, className]) => {
      const th = cell("th", label, className);
      th.scope = "col";
      head.append(th);
    });
    objectsHead.replaceChildren(head);
    objectsHint.textContent = reference
      ? "Largest first per color. Python: the committed colors.py report for this image, including its shape labels."
      : "Largest first per color, like format_report. No Python reference for uploads.";

    const layers = selected === null ? result.layers : [result.layers[selected]];
    let worst = 0;
    for (const layer of layers) {
      layer.objects.forEach((obj) => {
        const tr = document.createElement("tr");
        const name = cell("td");
        const swatch = cell("i", undefined, "swatch");
        swatch.style.background = rgb(layer.swatch);
        name.append(swatch, layer.name);
        tr.append(name, cell("td", `${obj.center[0].toFixed(1)}, ${obj.center[1].toFixed(1)}`, "num"));
        if (reference) {
          const match = referenceFor(layer.name, obj);
          if (match) {
            worst = Math.max(worst, match.d);
            if (match.candidate.shape) name.append(cell("small", match.candidate.shape, "shape"));
            tr.append(
              cell("td", `${match.candidate.center[0].toFixed(1)}, ${match.candidate.center[1].toFixed(1)}`, "num py"),
              cell("td", `${match.d.toFixed(2)} px`, "num" + (match.d < 1 ? " delta-good" : "")),
            );
          } else {
            tr.append(cell("td", "none", "num py"), cell("td", "", "num"));
          }
        } else {
          tr.append(cell("td", fmt.format(obj.area), "num"));
        }
        if (obj.touchesBorder) tr.title = "Touches the image border (background region, not marked)";
        tbody.append(tr);
      });
    }
    if (!tbody.children.length) {
      const tr = document.createElement("tr");
      const td = cell("td", "No objects above the size threshold.");
      td.colSpan = columns.length;
      tr.append(td);
      tbody.append(tr);
    }
    root.dataset.maxOffset = reference ? worst.toFixed(3) : "";
  }

  // --------------------------------------------------------- band maps ----

  function drawBandMaps() {
    const hv = document.getElementById("band-map-hv");
    const sv = document.getElementById("band-map-sv");
    const legend = document.getElementById("band-legend");
    if (!hv || !sv || !legend) return;
    const paint = (canvas, fn) => {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(canvas.width, canvas.height);
      for (let y = 0, p = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++, p += 4) {
          const c = S.COLOR_BANDS[fn(x, y)].swatch;
          img.data[p] = c[0];
          img.data[p + 1] = c[1];
          img.data[p + 2] = c[2];
          img.data[p + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    };
    const valueAt = (y, h) => Math.round(255 - (y * 255) / (h - 1));
    paint(hv, (x, y) => S.bandIndexOf(x, 255, valueAt(y, hv.height)));
    paint(sv, (x, y) => S.bandIndexOf(60, Math.round((x * 255) / (sv.width - 1)), valueAt(y, sv.height)));

    const items = S.COLOR_BANDS.map((band) => {
      const li = document.createElement("li");
      const swatch = cell("i", undefined, "swatch");
      swatch.style.background = rgb(band.swatch);
      const ranges = band.ranges
        .map(([lo, hi]) => {
          const part = (i, key) => (lo[i] === hi[i] ? `${key} ${lo[i]}` : `${key} ${lo[i]}..${hi[i]}`);
          return [part(0, "H"), part(1, "S"), part(2, "V")].join(" ");
        })
        .join(" or ");
      li.append(swatch, band.name + " ", cell("code", ranges));
      return li;
    });
    legend.replaceChildren(...items);
  }

  // ------------------------------------------------------------- events ----

  demoButtons.forEach((button) => button.addEventListener("click", () => loadDemo(button.dataset.demo)));
  upload.addEventListener("change", () => {
    loadFile(upload.files && upload.files[0]);
    upload.value = "";
  });
  showAll.addEventListener("click", () => select(null));

  ["dragenter", "dragover"].forEach((type) =>
    stage.addEventListener(type, (event) => {
      event.preventDefault();
      stage.classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((type) =>
    stage.addEventListener(type, (event) => {
      event.preventDefault();
      stage.classList.remove("dragging");
    }),
  );
  stage.addEventListener("drop", (event) => loadFile(event.dataTransfer && event.dataTransfer.files[0]));

  if ("ResizeObserver" in window) {
    let pending = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(drawMarks);
    }).observe(view);
  }

  drawBandMaps();
  window.ColorDemo = { state, loadDemo };

  // Start processing once the demo is near the viewport.
  const start = () => loadDemo("targets");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          start();
        }
      },
      { rootMargin: "600px 0px" },
    );
    io.observe(root);
  } else {
    start();
  }
})();
