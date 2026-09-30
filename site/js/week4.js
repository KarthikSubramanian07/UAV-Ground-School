/* Week 4 page: the wiring explorer, the logic analyser, the RTK journey and
   the check report. All data comes from docs/week04 (copied to assets/week4
   by scripts/build_site.py). */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const get = (name) =>
    fetch(`assets/week4/${name}.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    for (const c of children || []) node.append(c);
    return node;
  }

  function finding(f) {
    return h("li", { class: `finding ${f[0]}` }, [h("b", { text: f[0] === "ok" ? "pass" : f[0] }), h("span", { text: `${f[1]}: ${f[2]}` })]);
  }

  // --------------------------------------------------------- explorer ----
  const wiring = document.querySelector("#wiring-app svg");
  const panel = $("wiring-panel");
  if (wiring && panel) {
    get("site")
      .then((data) => {
        function clear() {
          wiring.querySelectorAll(".on").forEach((n) => n.classList.remove("on"));
          wiring.classList.remove("focus");
        }
        function showLink(id) {
          const d = data.links[id];
          if (!d) return;
          clear();
          wiring.classList.add("focus");
          wiring.querySelectorAll(`[data-link="${CSS.escape(id)}"]`).forEach((n) => n.classList.add("on"));
          const rows = d.wires.map((w) => h("tr", {}, [h("td", { text: w[0] }), h("td", { text: w[1] }), h("td", { class: "arrow", text: "to" }), h("td", { text: w[2] }), h("td", { text: w[3] })]));
          panel.replaceChildren(
            h("p", { class: "kicker", text: d.protocol }),
            h("h3", { text: `${d.from} to ${d.to}` }),
            h("p", { text: d.purpose }),
            d.connectors && d.connectors.length ? h("p", { class: "mono small", text: `${d.connectors[0]}  to  ${d.connectors[1]}` }) : h("span"),
            rows.length
              ? h("table", { class: "pins" }, [h("thead", {}, [h("tr", {}, ["Pin", "Signal", "", "Pin", "Signal"].map((t) => h("th", { text: t })))]), h("tbody", {}, rows)])
              : h("span"),
            ...(d.notes || []).map((n) => h("p", { class: "note", text: n })),
            h("ul", { class: "findings" }, d.findings.map(finding))
          );
        }
        function showPart(id) {
          const d = data.parts[id];
          if (!d) return;
          clear();
          wiring.classList.add("focus");
          wiring.querySelectorAll(`[data-part="${CSS.escape(id)}"]`).forEach((n) => n.classList.add("on"));
          const facts = [];
          if (d.price_usd !== null && d.price_usd !== undefined) facts.push(["Price", `$${d.price_usd}`]);
          if (d.mass_g !== null && d.mass_g !== undefined) facts.push(["Mass", `${d.mass_g} g${d.estimates.mass_g ? " (estimate)" : ""}`]);
          if (d.supply && d.supply.min_v) facts.push(["Supply", `${d.supply.min_v} to ${d.supply.max_v} V`]);
          if (d.logic_v) facts.push(["Logic", `${d.logic_v} V`]);
          const est = Object.entries(d.estimates || {}).map(([k, v]) => h("li", {}, [h("b", { text: k }), h("span", { text: v })]));
          panel.replaceChildren(
            h("p", { class: "kicker", text: `${d.role}${d.count > 1 ? `, ${d.count} of them` : ""}` }),
            h("h3", { text: d.name }),
            h("dl", { class: "facts" }, facts.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })])),
            d.url ? h("p", {}, [h("a", { href: d.url, rel: "noopener", text: "Source" })]) : h("span"),
            est.length ? h("details", {}, [h("summary", { text: "Estimates and why" }), h("ul", { class: "estimates" }, est)]) : h("span"),
            h("ul", { class: "findings" }, d.findings.map(finding))
          );
        }
        wiring.addEventListener("click", (event) => {
          const link = event.target.closest("[data-link]");
          const part = event.target.closest("[data-part]");
          if (link) showLink(link.dataset.link);
          else if (part) showPart(part.dataset.part);
        });
        wiring.querySelectorAll("[data-link], [data-part]").forEach((n) => {
          n.setAttribute("tabindex", "0");
          n.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              n.dataset.link ? showLink(n.dataset.link) : showPart(n.dataset.part);
            }
          });
        });
        function overview() {
          clear();
          const items = Object.entries(data.links)
            .filter(([id]) => !id.startsWith("power:"))
            .map(([id, d]) => {
              const b = h("button", { type: "button" }, [h("span", { text: `${d.from.split(".")[0]} to ${d.to.split(".")[0]}` }), h("span", { class: "mono", text: d.label })]);
              b.addEventListener("click", () => showLink(id));
              return h("li", {}, [b]);
            });
          panel.replaceChildren(
            h("p", { class: "kicker", text: "Protocol Pro" }),
            h("h3", { text: `${items.length} links, one autopilot` }),
            h("p", { text: "Pick a wire or a part in the diagram, or a link below." }),
            h("ul", { class: "overview" }, items)
          );
        }
        wiring.addEventListener("dblclick", overview);
        overview();
      })
      .catch((e) => (panel.textContent = `Could not load the design (${e.message}).`));
  }

  // ------------------------------------------------------------ scope ----
  const scopeSvg = $("scope-plot");
  if (scopeSvg) {
    get("scope").then((data) => {
      const title = $("scope-title");
      const unit = $("scope-unit");
      function trace(levels, y0, hgt, per) {
        let d = "";
        levels.forEach((v, i) => {
          const x = i * per;
          const yy = y0 + (v ? 0 : hgt);
          d += (i ? `L${x},${yy}` : `M${x},${yy}`) + `H${x + per}`;
        });
        return d;
      }
      function draw(key) {
        const s = data[key];
        scopeSvg.replaceChildren();
        title.textContent = s.title;
        unit.textContent = s.unit || "";
        const W = 960;
        if (key === "crsf") {
          const bytes = s.bytes.match(/../g);
          const per = W / bytes.length;
          bytes.forEach((b, i) => {
            scopeSvg.append(el("rect", { x: i * per + 1, y: 30, width: per - 2, height: 34, rx: 3, class: "byte" }));
            scopeSvg.append(el("text", { x: i * per + per / 2, y: 52, class: "hex", "text-anchor": "middle" }, b));
          });
          s.fields.forEach(([a, b, label], j) => bracket(a * per, b * per, 84, label, j));
          scopeSvg.setAttribute("viewBox", `0 0 ${W} 130`);
          return;
        }
        if (key === "i2c") {
          const per = W / s.sda.length;
          scopeSvg.append(el("text", { x: 0, y: 16, class: "lane" }, "SCL"));
          scopeSvg.append(el("path", { d: trace(s.scl, 24, 30, per), class: "wave" }));
          scopeSvg.append(el("text", { x: 0, y: 86, class: "lane" }, "SDA"));
          scopeSvg.append(el("path", { d: trace(s.sda, 94, 30, per), class: "wave sda" }));
          s.segments.forEach(([a, b, label], j) => bracket(a * per, b * per, 146, label, j));
          scopeSvg.setAttribute("viewBox", `0 0 ${W} 190`);
          return;
        }
        const n = s.levels.length;
        const per = W / n;
        scopeSvg.append(el("path", { d: trace(s.levels, 24, 40, per), class: "wave" }));
        if (s.stuff_bits) {
          s.stuff_bits.forEach((i) => scopeSvg.append(el("rect", { x: i * per, y: 18, width: per, height: 52, class: "stuff" })));
        }
        const spb = s.samples_per_bit || 1;
        const fieldScale = s.fields_are_unstuffed ? unstuffedToWire(s) : (i) => i;
        s.fields.forEach(([a, b, label], j) => bracket(fieldScale(a) * per, fieldScale(b) * per, 96, label, j));
        if (spb === 1 && n <= 150) {
          s.levels.forEach((v, i) => scopeSvg.append(el("text", { x: i * per + per / 2, y: 84, class: "bit", "text-anchor": "middle" }, String(v))));
        }
        scopeSvg.setAttribute("viewBox", `0 0 ${W} 140`);
      }
      function unstuffedToWire(s) {
        // map an index in the unstuffed frame to the wire index, skipping stuff bits
        const stuff = new Set(s.stuff_bits || []);
        const map = [];
        for (let w = 0; w < s.levels.length; w++) if (!stuff.has(w)) map.push(w);
        return (i) => (i < map.length ? map[i] : s.levels.length);
      }
      function bracket(x1, x2, yy, label, j) {
        const lift = (j % 2) * 16;
        scopeSvg.append(el("path", { d: `M${x1 + 1},${yy - 8 + lift}V${yy - 2 + lift}H${x2 - 1}V${yy - 8 + lift}`, class: "bracket" }));
        scopeSvg.append(el("text", { x: (x1 + x2) / 2, y: yy + 12 + lift, class: "field", "text-anchor": "middle" }, label));
      }
      document.querySelectorAll("#scope-picker button").forEach((b) =>
        b.addEventListener("click", () => {
          document.querySelectorAll("#scope-picker button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
          draw(b.dataset.proto);
        })
      );
      draw("uart");
    });
  }

  // ---------------------------------------------------------- journey ----
  const journeyList = $("journey-hops");
  if (journeyList) {
    get("journey").then((data) => {
      const detail = $("journey-detail");
      function show(i) {
        const hop = data.hops[i];
        journeyList.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-current", String(i === j)));
        const s = hop.sample;
        const rows = [
          ["Protocol", hop.protocol],
          ["Medium", hop.medium],
          ["Units", String(hop.units)],
          ["Bytes on the wire", `${hop.wire_bytes} for ${hop.payload_bytes} bytes of RTCM`],
          ["Overhead", `${(100 * hop.overhead).toFixed(1)} percent`],
        ];
        if (hop.seconds) rows.push(["Time on the wire", `${(1000 * hop.seconds).toFixed(1)} ms`]);
        const extra = [];
        if (s.hex) extra.push(h("pre", { class: "hexdump", text: s.hex.match(/.{1,2}/g).join(" ").replace(/(.{47}) /g, "$1\n") }));
        if (s.fields) extra.push(h("p", { class: "mono small", text: `CAN id ${s.id}: priority ${s.fields.priority}, type ${s.fields.type_id} (RTCMStream), node ${s.fields.source_node}` }));
        if (s.bits) extra.push(h("pre", { class: "bits", text: s.bits.join("").replace(/(.{60})/g, "$1\n") }));
        if (s.stuffed !== undefined) extra.push(h("p", { class: "small muted", text: `${s.stuffed} stuff bits in this frame keep the receivers' clocks locked.` }));
        if (s.fragments) extra.push(h("p", { class: "small muted", text: `${s.fragments} fragments, flags 0x${s.flags.toString(16)}: fragmented, fragment 0, sequence ${s.flags >> 3}.` }));
        detail.replaceChildren(h("h3", { text: hop.name }), h("dl", { class: "facts" }, rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })])), ...extra);
      }
      data.hops.forEach((hop, i) => {
        const b = h("button", { type: "button" }, [h("span", { class: "n", text: String(i + 1) }), h("span", { text: hop.protocol })]);
        b.addEventListener("click", () => show(i));
        journeyList.append(h("li", {}, [b]));
      });
      $("journey-summary").textContent = `${data.rtcm_bytes} bytes of RTCM (${data.rtcm_frames.map((f) => f.message).join(", ")}) arrive intact; the base decodes to ${data.base_position.lat}, ${data.base_position.lon}.`;
      show(0);
    });
  }

  // ------------------------------------------------------------ check ----
  const checkBody = $("check-rows");
  if (checkBody) {
    get("check").then((rows) => {
      const order = { error: 0, warning: 1, ok: 2 };
      rows.sort((a, b) => order[a.level] - order[b.level]);
      function render(level) {
        checkBody.replaceChildren(
          ...rows
            .filter((r) => level === "all" || r.level === level)
            .map((r) => h("tr", { class: r.level }, [h("td", {}, [h("span", { class: `pill ${r.level}`, text: r.level === "ok" ? "pass" : r.level })]), h("td", { text: r.rule }), h("td", { class: "mono", text: r.where }), h("td", { text: r.message })]))
        );
      }
      document.querySelectorAll("#check-filter button").forEach((b) =>
        b.addEventListener("click", () => {
          document.querySelectorAll("#check-filter button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
          render(b.dataset.level);
        })
      );
      render("warning");
    });
  }
})();
