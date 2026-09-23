/* Week 3 detector explorer. Loads the precomputed detections for one photo
   (written by `python -m week03 docs`) and draws them over the image as SVG
   rings, colored by how the detection scored against the annotations. */
(function () {
  "use strict";

  const app = document.getElementById("explorer-app");
  if (!app) return;

  const NAMES = {
    gray: "SimpleBlobDetector on grayscale",
    simple: "SimpleBlobDetector per palette color",
    contrast: "SimpleBlobDetector on background contrast",
    contour: "Contour filtering on color edges",
    log: "Laplacian of Gaussian",
    dog: "Difference of Gaussians",
    doh: "Determinant of Hessian",
  };
  const BLURBS = {
    gray: "The one line baseline: cv2.SimpleBlobDetector on the grayscale image. Blind to dots as bright as their background.",
    simple: "k-means learns the photo's palette in CIELAB; SimpleBlobDetector runs once per color on a smoothed similarity map. Only as good as the palette.",
    contrast: "SimpleBlobDetector on the Delta E between each pixel and a median filtered background. Fast and palette free.",
    contour: "Canny on CIELAB, then every region enclosed by color edges, kept if round, convex and compact.",
    log: "Scale normalized Laplacian of Gaussian on CIELAB, searched on the color magnitude and on each signed channel.",
    dog: "The difference of two Gaussian blurs: the cheap approximation of LoG that SIFT uses.",
    doh: "Determinant of the Hessian. Near zero on straight edges, so card borders do not fire.",
  };

  const stage = document.getElementById("explorer-stage");
  const image = document.getElementById("explorer-image");
  const svg = document.getElementById("explorer-rings");
  const tip = document.getElementById("explorer-tip");
  const title = document.getElementById("explorer-title-method");
  const blurb = document.getElementById("explorer-blurb");
  const colors = document.getElementById("explorer-colors");
  const field = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  const cache = new Map();
  let photo = "polka_dots_1";
  let method = "log";
  let token = 0;

  function load(name) {
    if (!cache.has(name)) {
      cache.set(
        name,
        fetch(`assets/week3/${name}.json`).then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
      );
    }
    return cache.get(name);
  }

  function ring(x, y, r, cls, data) {
    // A white halo under a colored ring reads on light and dark backgrounds alike.
    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("class", cls);
    g.append(circle(x, y, r, "halo"), circle(x, y, r, "line", data));
    return g;
  }

  function circle(x, y, r, cls, data) {
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", x);
    c.setAttribute("cy", y);
    c.setAttribute("r", Math.max(r, 1.5));
    c.setAttribute("class", cls);
    if (data) {
      c.dataset.info = JSON.stringify(data);
      c.setAttribute("tabindex", "-1");
    }
    return c;
  }

  function pct(v) {
    return `${(100 * v).toFixed(1)}%`;
  }

  function render(record) {
    const result = record.methods[method];
    svg.setAttribute("viewBox", `0 0 ${record.width} ${record.height}`);
    svg.replaceChildren();
    const stroke = Math.max(record.width, record.height) / 360;
    svg.style.setProperty("--stroke", stroke.toFixed(2));
    for (const m of result.missed) svg.append(ring(m.x, m.y, m.r, "miss"));
    for (const d of result.dots) svg.append(ring(d.x, d.y, d.r, d.status, d));

    title.textContent = NAMES[method];
    blurb.textContent = BLURBS[method];
    field("s-f1").textContent = result.f1.toFixed(3);
    field("s-p").textContent = result.tp + result.fp ? pct(result.precision) : "n/a";
    field("s-r").textContent = `${result.tp} / ${result.tp + result.fn}`;
    field("s-t").textContent = `${result.seconds.toFixed(2)} s`;

    const counts = new Map();
    for (const d of result.dots) {
      const key = d.color || "unknown";
      const entry = counts.get(key) || { n: 0, rgb: d.rgb };
      entry.n += 1;
      counts.set(key, entry);
    }
    colors.replaceChildren();
    [...counts.entries()]
      .sort((a, b) => b[1].n - a[1].n)
      .forEach(([name, { n, rgb }]) => {
        const li = document.createElement("li");
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        if (rgb) swatch.style.background = `rgb(${rgb.join(",")})`;
        li.append(swatch, document.createTextNode(name));
        const b = document.createElement("b");
        b.textContent = n;
        li.append(b);
        colors.append(li);
      });
    if (!counts.size) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No dots found.";
      colors.append(li);
    }
    stage.setAttribute("aria-busy", "false");
  }

  async function update() {
    const mine = ++token;
    stage.setAttribute("aria-busy", "true");
    tip.hidden = true;
    try {
      const record = await load(photo);
      if (mine !== token) return;
      const src = `assets/week3/${photo}.webp`;
      if (!image.src.endsWith(src)) {
        image.width = record.width;
        image.height = record.height;
        image.src = src;
      }
      render(record);
    } catch (error) {
      blurb.textContent = `Could not load the detections (${error.message}).`;
      stage.setAttribute("aria-busy", "false");
    }
  }

  function press(group, button) {
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  }

  field("photo-picker").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-photo]");
    if (!button) return;
    photo = button.dataset.photo;
    press(field("photo-picker"), button);
    update();
  });

  field("method-picker").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-method]");
    if (!button) return;
    method = button.dataset.method;
    press(field("method-picker"), button);
    update();
  });

  function showTip(target, clientX, clientY) {
    const d = JSON.parse(target.dataset.info);
    const label = { tp: "found", fp: "false positive", ignored: "not scored" }[d.status] || d.status;
    tip.replaceChildren();
    const head = document.createElement("b");
    head.textContent = `${d.color || "dot"}, ${label}`;
    const body = document.createElement("span");
    body.textContent = `(${d.x.toFixed(1)}, ${d.y.toFixed(1)})  r ${d.r.toFixed(1)} px  ΔE ${d.contrast.toFixed(0)}`;
    tip.append(head, body);
    const box = stage.getBoundingClientRect();
    tip.hidden = false;
    const x = Math.min(clientX - box.left + 14, box.width - tip.offsetWidth - 6);
    const y = Math.max(clientY - box.top - tip.offsetHeight - 10, 6);
    tip.style.transform = `translate(${x}px, ${y}px)`;
  }

  svg.addEventListener("pointermove", (event) => {
    const target = event.target.closest("circle[data-info]");
    if (!target) {
      tip.hidden = true;
      return;
    }
    showTip(target, event.clientX, event.clientY);
  });
  svg.addEventListener("pointerleave", () => {
    tip.hidden = true;
  });

  // Shareable state: week3?photo=polka_dots_2&method=gray#explorer
  const params = new URLSearchParams(location.search);
  const wantPhoto = app.querySelector(`button[data-photo="${params.get("photo")}"]`);
  const wantMethod = app.querySelector(`button[data-method="${params.get("method")}"]`);
  if (wantPhoto) {
    photo = wantPhoto.dataset.photo;
    press(field("photo-picker"), wantPhoto);
  }
  if (wantMethod) {
    method = wantMethod.dataset.method;
    press(field("method-picker"), wantMethod);
  }

  function remember() {
    const url = new URL(location.href);
    url.searchParams.set("photo", photo);
    url.searchParams.set("method", method);
    history.replaceState(null, "", url);
  }
  field("photo-picker").addEventListener("click", remember);
  field("method-picker").addEventListener("click", remember);

  update();
})();
