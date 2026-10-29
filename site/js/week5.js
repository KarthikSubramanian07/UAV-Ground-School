/* Week 5 page: the mission replay, the quiz, the QoS playground, the
   executor timeline and the packet inspector. Data comes from docs/week05
   (copied to assets/week5 by scripts/build_site.py).

   checkCompatible() is a line by line port of week05/qos.py check_compatible
   (itself a port of rmw_dds_common's qos_profile_check_compatible);
   tests/test_site.py runs both on the same profiles and requires identical
   verdicts and reason strings. The UI part only runs in a browser. */
(function (root) {
  "use strict";

  // ------------------------------------------------------------- QoS ----
  const OK = "ok";
  const WARNING = "warning";
  const ERROR = "error";

  function profile(changes) {
    return Object.assign(
      { reliability: "reliable", durability: "volatile", history: "keep_last", depth: 10, deadline_ns: 0, lifespan_ns: 0, liveliness: "system_default", lease_ns: 0 },
      changes || {}
    );
  }

  const PRESETS = {
    default: profile(),
    sensor_data: profile({ reliability: "best_effort", depth: 5 }),
    system_default: profile({ reliability: "system_default", durability: "system_default", history: "system_default", depth: 0 }),
    best_available: profile({ reliability: "best_available", durability: "best_available", liveliness: "best_available" }),
    rosout: profile({ durability: "transient_local", depth: 1000, lifespan_ns: 10e9 }),
  };

  function unknown(value) {
    return value === "system_default" || value === "unknown";
  }

  function checkCompatible(pub, sub) {
    let level = OK;
    const reasons = [];
    const error = (text) => {
      level = ERROR;
      reasons.push("ERROR: " + text + ";");
    };
    const warning = (text) => {
      level = WARNING;
      reasons.push("WARNING: " + text + ";");
    };

    if (pub.reliability === "best_effort" && sub.reliability === "reliable") error("Best effort publisher and reliable subscription");
    if (pub.durability === "volatile" && sub.durability === "transient_local") error("Volatile publisher and transient local subscription");
    if (pub.deadline_ns === 0 && sub.deadline_ns !== 0) error("Subscription has a deadline, but publisher does not");
    if (pub.deadline_ns !== 0 && sub.deadline_ns !== 0 && sub.deadline_ns < pub.deadline_ns) error("Subscription deadline is less than publisher deadline");
    if (pub.liveliness === "automatic" && sub.liveliness === "manual_by_topic") error("Publisher's liveliness is automatic and subscription's is manual by topic");
    if (pub.lease_ns === 0 && sub.lease_ns !== 0) error("Subscription has a liveliness lease duration, but publisher does not");
    if (pub.lease_ns !== 0 && sub.lease_ns !== 0 && sub.lease_ns < pub.lease_ns) error("Subscription liveliness lease duration is less than publisher");

    if (level === OK) {
      const pr = pub.reliability;
      const sr = sub.reliability;
      if (unknown(pr) && unknown(sr)) warning(`Publisher reliability is ${pr} and subscription reliability is ${sr}`);
      else if (unknown(pr) && sr === "reliable") warning(`Reliable subscription, but publisher is ${pr}`);
      else if (pr === "best_effort" && unknown(sr)) warning(`Best effort publisher, but subscription is ${sr}`);
      const pd = pub.durability;
      const sd = sub.durability;
      // (sic) rmw spells it "durabilty"
      if (unknown(pd) && unknown(sd)) warning(`Publisher durabilty is ${pd} and subscription durability is ${sd}`);
      else if (unknown(pd) && sd === "transient_local") warning(`Transient local subscription, but publisher is ${pd}`);
      else if (pd === "volatile" && unknown(sd)) warning(`Volatile publisher, but subscription is ${sd}`);
      const pl = pub.liveliness;
      const sl = sub.liveliness;
      if (unknown(pl) && unknown(sl)) warning(`Publisher liveliness is ${pl} and subscription liveliness is ${sl}`);
      else if (unknown(pl) && sl === "manual_by_topic") warning(`Subscription's liveliness is manual by topic, but publisher's is ${pl}`);
      else if (pl === "automatic" && unknown(sl)) warning(`Publisher's liveliness is automatic, but subscription's is ${sl}`);
    }
    return [level, reasons.join("")];
  }

  const api = { checkCompatible, PRESETS, profile };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }
  root.QoS = api;

  // -------------------------------------------------------- helpers ----
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const motion = document.documentElement.classList.contains("motion");
  const get = (name) =>
    fetch(`assets/week5/${name}.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

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

  function lastAtOrBefore(rows, t) {
    // rows sorted by row[0]; index of the last row with row[0] <= t, or -1
    let lo = 0;
    let hi = rows.length - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (rows[mid][0] <= t) {
        best = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return best;
  }

  // -------------------------------------------------------- mission ----
  const missionApp = $("mission-app");
  if (missionApp) {
    get("mission")
      .then(startMission)
      .catch((e) => ($("mission-log").replaceChildren(h("li", { text: `Could not load the mission (${e.message}).` }))));
  }

  function startMission(m) {
    // Events are timed from the bridge's start; the recorder's rows from its
    // own clock. Align them on the first /fly_to goal, logged by both.
    const firstGoal = m.events.find((e) => e.text.startsWith("/fly_to ("));
    const offset = firstGoal ? m.status[0][0] - firstGoal.t : 0;
    const shift = (rows) => rows.map((r) => [r[0] - offset, ...r.slice(1)]);
    const odom = shift(m.odometry);
    const dets = shift(m.detections);
    const feedback = shift(m.feedback);
    const status = shift(m.status);
    const battery = shift(m.battery || []);
    const events = m.events.slice().sort((a, b) => a.t - b.t);
    const end = Math.max(odom[odom.length - 1][0], events[events.length - 1].t) + 0.2;
    const frames = [];
    for (let t = odom[0][0]; t <= odom[odom.length - 1][0]; t += 1 / 15) frames.push([t]);

    // goal ids to the planner's labels, in order
    const goalLabels = events.filter((e) => /^goal [^:]+:/.test(e.text)).map((e) => e.text.match(/^goal ([^:]+):/)[1]);
    const goalCoords = events.filter((e) => /^goal [^:]+:/.test(e.text)).map((e) => e.text.match(/\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)/).slice(1).map(Number));
    const goalIds = [];
    status.forEach((s) => goalIds.includes(s[1]) || goalIds.push(s[1]));
    const goalName = (id) => goalLabels[goalIds.indexOf(id)] || id;
    const goalAt = (id) => goalCoords[goalIds.indexOf(id)];
    const confirmed = events.find((e) => e.text.startsWith("target confirmed"));
    const confirmedXY = confirmed ? confirmed.text.match(/\(([-\d.]+), ([-\d.]+)\)/).slice(1).map(Number) : null;

    // ---- map
    const map = $("mission-map");
    const X0 = -3;
    const Y1 = 11;
    const S = 400 / 20;
    const sx = (x) => (x - X0) * S;
    const sy = (y) => (Y1 - y) * S;
    const grid = el("g", { class: "map-grid" });
    for (let v = -2; v <= 16; v += 2) {
      grid.append(el("line", { x1: sx(v), y1: 0, x2: sx(v), y2: 400, class: v === 0 ? "axis" : "" }));
      grid.append(el("text", { x: sx(v) + 3, y: 394, class: "tick" }, String(v)));
    }
    for (let v = -8; v <= 10; v += 2) {
      grid.append(el("line", { x1: 0, y1: sy(v), x2: 400, y2: sy(v), class: v === 0 ? "axis" : "" }));
      grid.append(el("text", { x: 3, y: sy(v) - 3, class: "tick" }, String(v)));
    }
    map.append(grid);
    map.append(el("polyline", { points: [[0, 0], [12, 6], [12, -6]].map(([x, y]) => `${sx(x)},${sy(y)}`).join(" "), class: "planned" }));
    [[0, 0], [12, 6], [12, -6]].forEach(([x, y], i) => {
      map.append(el("circle", { cx: sx(x), cy: sy(y), r: 3.5, class: "wp" }));
      map.append(el("text", { x: sx(x) + 8, y: sy(y) + (i === 1 ? 4 : 14), class: "wp-label" }, i === 0 ? "takeoff" : `waypoint ${i + 1}`));
    });
    map.append(el("circle", { cx: sx(11), cy: sy(7), r: 0.8 * S, class: "target" }));
    map.append(el("circle", { cx: sx(11), cy: sy(7), r: 2.5, class: "target-dot" }));
    map.append(el("text", { x: sx(11) - 0.8 * S - 6, y: sy(7) - 6, "text-anchor": "end", class: "target-label" }, "target"));
    const goalMark = el("g", { class: "goal-mark", visibility: "hidden" }, undefined);
    goalMark.append(el("circle", { r: 7 }), el("path", { d: "M-11 0H-4M4 0H11M0 -11V-4M0 4V11" }));
    map.append(goalMark);
    const confirmMark = el("g", { class: "confirm-mark", visibility: "hidden" });
    if (confirmedXY) {
      confirmMark.setAttribute("transform", `translate(${sx(confirmedXY[0])} ${sy(confirmedXY[1])})`);
      confirmMark.append(el("path", { d: "M-6 -6L6 6M6 -6L-6 6" }), el("text", { x: -14, y: 26, "text-anchor": "end" }, "confirmed"));
    }
    map.append(confirmMark);
    const trail = el("polyline", { class: "trail", points: "" });
    map.append(trail);
    const drone = el("g", { class: "drone" });
    drone.append(el("circle", { r: 13, class: "shadow" }));
    [[-7, -7], [7, -7], [-7, 7], [7, 7]].forEach(([x, y]) => drone.append(el("circle", { cx: x, cy: y, r: 4.2, class: "rotor" })));
    drone.append(el("rect", { x: -4, y: -4, width: 8, height: 8, rx: 2, class: "body" }));
    map.append(drone);

    // ---- camera view
    const cam = $("cam-view");
    cam.append(el("rect", { x: 0.5, y: 0.5, width: 319, height: 239, class: "frame" }));
    cam.append(el("path", { d: "M160 110V130M150 120H170", class: "cross" }));
    const camHit = el("g", { class: "cam-hit", visibility: "hidden" });
    const camBox = el("rect", { width: 34, height: 34, rx: 3 });
    const camText = el("text", { class: "cam-text" }, "");
    camHit.append(camBox, camText);
    const camNone = el("text", { x: 160, y: 200, "text-anchor": "middle", class: "cam-none" }, "no target in view");
    cam.append(camHit, camNone);

    // ---- graph pulses
    const pulseLayer = $("pulses");
    const paths = {};
    ["e-odom-cam", "e-image", "e-det", "e-odom-plan", "e-srv", "e-act"].forEach((id) => {
      const p = $(id);
      paths[id] = { el: p, len: p.getTotalLength() };
    });
    const live = [];
    function pulse(edge, reverse, delay, kind) {
      if (!motion) return;
      const dot = el("circle", { r: kind === "big" ? 5.5 : 4, class: `pulse ${edge}` });
      pulseLayer.append(dot);
      live.push({ dot, edge, reverse, start: performance.now() + (delay || 0), dur: 520 });
    }
    function flash(node) {
      const n = $(`n-${node}`);
      if (!n) return;
      n.classList.remove("hot");
      void n.getBBox();
      n.classList.add("hot");
    }
    function animatePulses(now) {
      for (let i = live.length - 1; i >= 0; i--) {
        const p = live[i];
        const k = (now - p.start) / p.dur;
        if (k < 0) {
          p.dot.setAttribute("visibility", "hidden");
          continue;
        }
        if (k >= 1) {
          p.dot.remove();
          live.splice(i, 1);
          continue;
        }
        const path = paths[p.edge];
        const pt = path.el.getPointAtLength(path.len * (p.reverse ? 1 - k : k));
        p.dot.setAttribute("visibility", "visible");
        p.dot.setAttribute("cx", pt.x);
        p.dot.setAttribute("cy", pt.y);
      }
    }

    function emitBetween(a, b) {
      const inRange = (rows, fn) => {
        for (let i = lastAtOrBefore(rows, a) + 1; i < rows.length && rows[i][0] <= b; i++) fn(rows[i]);
      };
      inRange(odom, () => {
        pulse("e-odom-cam");
        pulse("e-odom-plan");
        flash("px4_bridge");
      });
      inRange(frames, () => pulse("e-image"));
      inRange(dets, () => {
        pulse("e-det", false, 0, "big");
        flash("detector");
      });
      inRange(feedback, () => pulse("e-act", true));
      inRange(status, () => pulse("e-act", true, 0, "big"));
      events.forEach((e) => {
        if (e.t <= a || e.t > b) return;
        flash(e.node);
        if (e.node !== "px4_bridge") return;
        if (e.text.startsWith("/set_mode") || e.text.startsWith("/arm")) {
          pulse("e-srv", false, 0, "big");
          pulse("e-srv", true, 260, "big");
        } else if (e.text.startsWith("/fly_to (") || e.text.includes("cancel requested")) {
          pulse("e-act", false, 0, "big");
        } else if (e.text.startsWith("/fly_to")) {
          pulse("e-act", true, 0, "big");
        }
      });
    }

    // ---- state at time t
    const log = $("mission-log");
    const scrub = $("mission-scrub");
    const clock = $("mission-clock");
    const play = $("mission-play");
    let shown = -1;
    function state(t) {
      // odometry, interpolated
      let x = 0;
      let y = 0;
      let z = 0;
      let v = 0;
      const i = lastAtOrBefore(odom, t);
      if (i >= 0) {
        const r0 = odom[i];
        const r1 = odom[Math.min(i + 1, odom.length - 1)];
        const k = r1[0] > r0[0] ? Math.min(1, (t - r0[0]) / (r1[0] - r0[0])) : 0;
        [x, y, z, v] = [1, 2, 3, 4].map((j) => r0[j] + (r1[j] - r0[j]) * k);
      }
      drone.setAttribute("transform", `translate(${sx(x)} ${sy(y)}) scale(${0.8 + z / 12})`);
      drone.classList.toggle("landed", z < 0.05);
      const pts = odom.slice(0, i + 1).map((r) => `${sx(r[1]).toFixed(1)},${sy(r[2]).toFixed(1)}`);
      if (i >= 0) pts.push(`${sx(x).toFixed(1)},${sy(y).toFixed(1)}`);
      trail.setAttribute("points", pts.join(" "));
      $("ro-alt").textContent = `${z.toFixed(1)} m`;
      $("ro-speed").textContent = `${v.toFixed(1)} m/s`;

      // planner phase
      const phases = events.filter((e) => e.t <= t && e.node === "planner" && /^\[.*\]$/.test(e.text));
      const done = events.find((e) => e.t <= t && e.text.startsWith("MISSION COMPLETE"));
      $("ro-phase").textContent = done ? "complete" : phases.length ? phases[phases.length - 1].text.slice(1, -1) : "waiting";

      // action goal and feedback
      const si = lastAtOrBefore(status, t);
      const goalEl = $("ro-goal");
      const distEl = $("ro-dist");
      if (si >= 0) {
        const s = status[si];
        goalEl.textContent = `${goalName(s[1])}: ${s[2]}`;
        goalEl.dataset.status = s[2];
        const active = s[2] === "executing" || s[2] === "accepted" || s[2] === "canceling";
        const fi = lastAtOrBefore(feedback, t);
        distEl.textContent = active && fi >= 0 && feedback[fi][1] === s[1] ? `${feedback[fi][2].toFixed(2)} m` : "";
        const at = goalAt(s[1]);
        if (active && at) {
          goalMark.setAttribute("transform", `translate(${sx(at[0])} ${sy(at[1])})`);
          goalMark.setAttribute("visibility", "visible");
        } else goalMark.setAttribute("visibility", "hidden");
      } else {
        goalEl.textContent = "none";
        delete goalEl.dataset.status;
        distEl.textContent = "";
        goalMark.setAttribute("visibility", "hidden");
      }
      const bi = lastAtOrBefore(battery, t);
      $("ro-batt").textContent = bi >= 0 ? `${(100 * battery[bi][1]).toFixed(2)}%` : "";
      confirmMark.setAttribute("visibility", confirmed && confirmed.t <= t ? "visible" : "hidden");

      // camera
      const di = lastAtOrBefore(dets, t);
      if (di >= 0 && t - dets[di][0] < 0.35) {
        const [, u, vv, score] = dets[di];
        camBox.setAttribute("x", u - 17);
        camBox.setAttribute("y", vv - 17);
        camText.setAttribute("x", Math.min(u + 21, 250));
        camText.setAttribute("y", Math.max(vv - 8, 14));
        camText.textContent = `score ${score.toFixed(2)}`;
        camHit.setAttribute("visibility", "visible");
        camNone.setAttribute("visibility", "hidden");
      } else {
        camHit.setAttribute("visibility", "hidden");
        camNone.setAttribute("visibility", "visible");
      }

      // log
      let n = 0;
      while (n < events.length && events[n].t <= t) n++;
      if (n !== shown) {
        shown = n;
        log.replaceChildren(
          ...events.slice(0, n).map((e) =>
            h("li", { class: e.text.startsWith("MISSION") || e.text.startsWith("target") ? "hl" : "" }, [
              h("span", { class: "t", text: e.t.toFixed(2) }),
              h("span", { class: `who ${e.node}`, text: e.node }),
              h("span", { class: "msg", text: e.text }),
            ])
          )
        );
        if (!n) log.append(h("li", { class: "empty", text: "Press play. The log fills as the nodes speak." }));
        log.scrollTop = log.scrollHeight;
      }
      clock.textContent = `${Math.max(0, t).toFixed(1)} s`;
      scrub.value = String(Math.round((1000 * t) / end));
    }

    let t = 0;
    let playing = false;
    let speed = 1;
    let last = 0;
    let touched = false;
    function frame(now) {
      if (playing) {
        const next = Math.min(end, t + ((now - last) / 1000) * speed);
        emitBetween(t, next);
        t = next;
        last = now;
        state(t);
        if (t >= end) setPlaying(false);
      }
      animatePulses(now);
      if (playing || live.length) requestAnimationFrame(frame);
    }
    function setPlaying(on) {
      playing = on;
      play.setAttribute("aria-pressed", String(on));
      play.textContent = on ? "Pause" : t >= end ? "Replay" : "Play";
      if (on) {
        if (t >= end) {
          t = 0;
          shown = -1;
        }
        last = performance.now();
        requestAnimationFrame(frame);
      }
    }
    play.addEventListener("click", () => {
      touched = true;
      setPlaying(!playing);
    });
    scrub.addEventListener("input", () => {
      touched = true;
      t = (Number(scrub.value) / 1000) * end;
      state(t);
      if (!playing) play.textContent = t >= end ? "Replay" : "Play";
    });
    document.querySelectorAll(".speed button").forEach((b) =>
      b.addEventListener("click", () => {
        speed = Number(b.dataset.speed);
        document.querySelectorAll(".speed button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      })
    );

    if (motion) {
      state(0);
      if ("IntersectionObserver" in window) {
        const io = new IntersectionObserver(
          (entries) => {
            if (entries.some((e) => e.isIntersecting) && !touched) {
              io.disconnect();
              setPlaying(true);
            }
          },
          { threshold: 0.35 }
        );
        io.observe(missionApp);
      }
    } else {
      t = end;
      state(end);
      play.textContent = "Replay";
    }
  }

  // ----------------------------------------------------------- quiz ----
  const quiz = $("quiz-app");
  if (quiz) {
    const items = [...quiz.querySelectorAll("li")];
    const score = $("quiz-score");
    function update() {
      const answered = items.filter((li) => li.dataset.state);
      const right = items.filter((li) => li.dataset.state === "right").length;
      score.textContent = answered.length === items.length ? `${right} of ${items.length} right.${right === items.length ? " Ready for the lab." : " Try the others again."}` : `${answered.length} of ${items.length} answered`;
    }
    items.forEach((li, i) => {
      li.prepend(h("span", { class: "qn", text: String(i + 1) }));
      const why = h("p", { class: "why" });
      const choices = h(
        "div",
        { class: "choices", role: "group", "aria-label": `Question ${i + 1}` },
        ["topic", "service", "action"].map((kind) => {
          const b = h("button", { class: "chip plain", type: "button", "aria-pressed": "false", text: kind });
          b.addEventListener("click", () => {
            const right = kind === li.dataset.answer;
            li.dataset.state = right ? "right" : "wrong";
            choices.querySelectorAll("button").forEach((o) => {
              o.setAttribute("aria-pressed", String(o === b));
              o.classList.toggle("answer", o.textContent === li.dataset.answer);
            });
            why.textContent = `${right ? "Right" : `Not quite: ${li.dataset.answer}`}. ${li.dataset.why}`;
            update();
          });
          return b;
        })
      );
      li.append(choices, why);
    });
    update();
  }

  // ------------------------------------------------------ QoS playground ----
  const qosApp = $("qos-app");
  if (qosApp) {
    const OPTIONS = {
      reliability: ["reliable", "best_effort", "system_default", "best_available"],
      durability: ["volatile", "transient_local", "system_default", "best_available"],
      liveliness: ["automatic", "manual_by_topic", "system_default", "best_available"],
    };
    const DURATIONS = [
      [0, "not set"],
      [100e6, "100 ms"],
      [500e6, "500 ms"],
      [1e9, "1 s"],
      [2e9, "2 s"],
    ];
    const PRESET_NAMES = ["default", "sensor_data", "system_default", "best_available"];
    const sides = { pub: Object.assign({}, PRESETS.sensor_data), sub: Object.assign({}, PRESETS.default) };
    const controls = {};
    qosApp.querySelectorAll(".qos-side").forEach((fs) => {
      const side = fs.dataset.side;
      const presetRow = h("div", { class: "toolbar presets", role: "group", "aria-label": "Presets" });
      PRESET_NAMES.forEach((name) => {
        const b = h("button", { class: "chip plain", type: "button", "aria-pressed": "false", "data-preset": name, text: name });
        b.addEventListener("click", () => {
          sides[side] = Object.assign({}, PRESETS[name]);
          sync();
        });
        presetRow.append(b);
      });
      fs.append(presetRow);
      controls[side] = {};
      const grid = h("div", { class: "qos-fields" });
      const add = (key, label, options) => {
        const select = h("select", { id: `qos-${side}-${key}` });
        options.forEach(([value, text]) => select.append(h("option", { value: String(value), text })));
        select.addEventListener("change", () => {
          sides[side][key] = key.endsWith("_ns") ? Number(select.value) : select.value;
          sync();
        });
        controls[side][key] = select;
        grid.append(h("label", { for: select.id, text: label }), select);
      };
      add("reliability", "Reliability", OPTIONS.reliability.map((v) => [v, v.replace(/_/g, " ")]));
      add("durability", "Durability", OPTIONS.durability.map((v) => [v, v.replace(/_/g, " ")]));
      add("liveliness", "Liveliness", OPTIONS.liveliness.map((v) => [v, v.replace(/_/g, " ")]));
      add("deadline_ns", "Deadline", DURATIONS);
      add("lease_ns", "Lease", DURATIONS);
      fs.append(grid);
    });
    const verdict = $("qos-verdict");
    const HEADLINES = {
      ok: "They will match.",
      warning: "They may match: it depends on the middleware.",
      error: "They will never match.",
    };
    function sync() {
      for (const side of ["pub", "sub"]) {
        const p = sides[side];
        for (const key in controls[side]) controls[side][key].value = String(p[key]);
        qosApp.querySelectorAll(`.qos-side[data-side="${side}"] [data-preset]`).forEach((b) => {
          const q = PRESETS[b.dataset.preset];
          const same = ["reliability", "durability", "liveliness", "deadline_ns", "lease_ns"].every((k) => q[k] === p[k]);
          b.setAttribute("aria-pressed", String(same));
        });
      }
      const [level, reason] = checkCompatible(sides.pub, sides.sub);
      verdict.dataset.level = level;
      verdict.replaceChildren(
        h("p", { class: "verdict-line" }, [h("span", { class: `pill ${level}`, text: level }), h("strong", { text: HEADLINES[level] })]),
        h("p", { class: "reason-label small muted", text: reason ? "Reason, exactly as rmw words it:" : "rmw gives no reason: nothing to warn about." }),
        reason ? h("code", { class: "reason", text: reason }) : h("span")
      );
    }
    sync();
  }

  // ------------------------------------------------ executor timeline ----
  const timeline = $("exec-timeline");
  if (timeline) {
    get("lab")
      .then((lab) => drawTimeline(lab.blocking))
      .catch(() => drawTimeline(null));
  }

  function drawTimeline(measured) {
    const svg = timeline;
    const X = 90;
    const W = 850;
    const MS = 3000;
    const LANE = 108;
    const px = (ms) => X + (ms / MS) * W;
    const lanes = [
      { name: "SingleThreadedExecutor", separate: false },
      { name: "MultiThreadedExecutor, one group", separate: false },
      { name: "MultiThreadedExecutor, separate groups", separate: true },
    ];
    const bottom = 12 + lanes.length * LANE;
    svg.setAttribute("viewBox", `0 0 960 ${bottom + 26}`);
    for (let ms = 0; ms <= MS; ms += 500) {
      svg.append(el("line", { x1: px(ms), y1: 30, x2: px(ms), y2: bottom, class: "grid" }));
      svg.append(el("text", { x: px(ms), y: bottom + 18, "text-anchor": ms === MS ? "end" : ms ? "middle" : "start", class: "tick" }, `${ms} ms`));
    }
    lanes.forEach((lane, li) => {
      const y = 12 + li * LANE;
      const m = measured && measured.find((r) => r.executor === lane.name);
      const title = el("text", { x: 0, y: y + 12, class: "lane-name" }, lane.name);
      if (m) title.append(el("tspan", { class: "lane-stat", dx: 12 }, `${m.ticks} ticks, worst gap ${m.max_gap_ms} ms (measured)`));
      svg.append(title);
      svg.append(el("text", { x: X - 12, y: y + 38, "text-anchor": "end", class: "row-label" }, "timer"));
      svg.append(el("text", { x: X - 12, y: y + 66, "text-anchor": "end", class: "row-label" }, lane.separate ? "thread 2" : "callback"));
      const ticks = [];
      if (lane.separate) {
        for (let ms = 0; ms < MS; ms += 50) ticks.push(ms);
      } else {
        // one thread for both: a tick due while the callback runs fires when it returns
        for (let c = 0; c < MS; c += 500) ticks.push(c, c + 300, c + 350, c + 400, c + 450);
      }
      ticks.forEach((ms, i) => {
        const late = !lane.separate && i % 5 === 1;
        svg.append(el("line", { x1: px(ms), y1: y + 24, x2: px(ms), y2: y + 44, class: late ? "tickmark late" : "tickmark" }));
      });
      for (let c = 0; c < MS; c += 500) {
        svg.append(el("rect", { x: px(c), y: y + 54, width: px(c + 300) - px(c), height: 16, rx: 3, class: "busy" }));
      }
      if (!lane.separate) {
        svg.append(el("path", { d: `M${px(0) + 2} ${y + 78}V${y + 84}H${px(300) - 2}V${y + 78}`, class: "gap" }));
        svg.append(el("text", { x: px(310), y: y + 86, class: "gap-label" }, "timer starved for 300 ms, then one late tick"));
      } else {
        svg.append(el("text", { x: px(310), y: y + 86, class: "gap-label ok" }, "the timer keeps its 50 ms on another thread"));
      }
    });
  }

  // --------------------------------------------------- packet inspector ----
  const wireApp = $("wire-app");
  if (wireApp) {
    get("wire")
      .then(startInspector)
      .catch((e) => ($("packet-head").textContent = `Could not load the packets (${e.message}).`));
  }

  function startInspector(packets) {
    const list = $("packet-list");
    const head = $("packet-head");
    const grid = $("hexgrid");
    const tree = $("field-tree");
    const readout = $("field-readout");
    let bytes = [];
    let owner = [];
    let spans = [];
    let pinned = -1;
    let lit = -1;

    function crumbs(k) {
      // the submessage (or header) a field belongs to: the nearest depth 0 span before it
      if (spans[k].depth === 0) return [];
      for (let j = k - 1; j >= 0; j--) if (spans[j].depth === 0) return [spans[j].label.replace(/ header$/, "")];
      return [];
    }

    function light(k) {
      if (k === lit) return;
      if (lit >= 0) {
        const s = spans[lit];
        for (let i = s.start; i < s.end && i < bytes.length; i++) bytes[i].forEach((n) => n.classList.remove("on"));
        const row = tree.children[lit];
        if (row) row.classList.remove("on");
      }
      lit = k;
      if (k < 0) {
        readout.replaceChildren(h("span", { class: "muted", text: "Hover a byte, or pick a field." }));
        return;
      }
      const s = spans[k];
      for (let i = s.start; i < s.end && i < bytes.length; i++) bytes[i].forEach((n) => n.classList.add("on"));
      const row = tree.children[k];
      if (row) row.classList.add("on");
      const path = crumbs(k);
      readout.replaceChildren(
        h("span", { class: "crumbs", text: path.length ? path.join(" / ") + " /" : "" }),
        h("b", { text: s.label }),
        h("span", { class: "val", text: s.value || "" }),
        h("span", { class: "range mono", text: `bytes ${s.start} to ${s.end - 1}, ${s.end - s.start} long` })
      );
    }

    function show(index) {
      const p = packets[index];
      list.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-current", String(j === index)));
      head.replaceChildren(
        h("h3", { text: p.title }),
        h("p", { text: p.text }),
        h("dl", { class: "facts" }, [
          h("dt", { text: "from" }),
          h("dd", { class: "mono", text: p.from }),
          h("dt", { text: "to" }),
          h("dd", { class: "mono", text: `${p.to}, ${p.port}` }),
          h("dt", { text: "decoded" }),
          h("dd", { class: "mono", text: p.summary.length > 1 ? `${p.summary[p.summary.length - 1]} (and ${p.summary.length - 1} more)` : p.summary[0] }),
        ])
      );
      const raw = p.hex.match(/../g) || [];
      spans = p.spans;
      owner = new Array(raw.length).fill(-1);
      spans.forEach((s, k) => {
        for (let i = s.start; i < s.end && i < raw.length; i++) {
          if (owner[i] < 0 || spans[owner[i]].depth <= s.depth) owner[i] = k;
        }
      });
      bytes = raw.map(() => []);
      lit = -1;
      pinned = -1;
      const frag = document.createDocumentFragment();
      for (let r = 0; r < raw.length; r += 16) {
        const line = h("div", { class: "hexrow" });
        line.append(h("span", { class: "off", text: r.toString(16).padStart(4, "0") }));
        const hexes = h("span", { class: "hexes" });
        const ascii = h("span", { class: "ascii" });
        for (let i = r; i < Math.min(r + 16, raw.length); i++) {
          const k = owner[i];
          const cls = k < 0 ? "b none" : `b d${Math.min(spans[k].depth, 2)} f${k % 2}`;
          const b = h("span", { class: cls, "data-i": String(i), text: raw[i] });
          const code = parseInt(raw[i], 16);
          const c = h("span", { class: cls, "data-i": String(i), text: code >= 32 && code < 127 ? String.fromCharCode(code) : "." });
          bytes[i].push(b, c);
          hexes.append(b);
          ascii.append(c);
        }
        line.append(hexes, ascii);
        frag.append(line);
      }
      grid.replaceChildren(frag);
      tree.replaceChildren(
        ...spans.map((s, k) => {
          const b = h("button", { type: "button", style: `--depth: ${s.depth}` }, [h("span", { class: "lbl", text: s.label }), h("span", { class: "val", text: s.value || "" })]);
          b.addEventListener("click", () => {
            pinned = k;
            light(k);
            const first = bytes[s.start] && bytes[s.start][0];
            if (first) grid.scrollTo({ top: first.offsetTop - grid.offsetTop - 40, behavior: motion ? "smooth" : "auto" });
          });
          return h("li", {}, [b]);
        })
      );
      const payload = spans.findIndex((s) => s.label.startsWith("serialized payload"));
      pinned = payload >= 0 ? payload : -1;
      light(pinned);
    }

    function byteAt(event) {
      const t = event.target.closest("[data-i]");
      return t ? owner[Number(t.dataset.i)] : -2;
    }
    grid.addEventListener("mouseover", (e) => {
      const k = byteAt(e);
      if (k !== -2) light(k);
    });
    grid.addEventListener("mouseleave", () => light(pinned));
    grid.addEventListener("click", (e) => {
      const k = byteAt(e);
      if (k === -2) return;
      pinned = k;
      light(k);
    });

    packets.forEach((p, i) => {
      const b = h("button", { type: "button" }, [h("span", { class: "n", text: String(i + 1) }), h("span", { class: "pt" }, [h("b", { text: p.title }), h("span", { class: "mono", text: p.summary[p.summary.length - 1] })])]);
      b.addEventListener("click", () => show(i));
      list.append(h("li", {}, [b]));
    });
    const hello = packets.findIndex((p) => p.id === "hello");
    show(hello >= 0 ? hello : 0);
  }
})(typeof window !== "undefined" ? window : globalThis);
