/* Week 4 PID playground. simulate() and metrics() are a line by line port of
   week04/pid.py; tests/test_week04_site.py runs both on the same gains and
   requires identical traces. The UI part only runs in a browser. */
(function (root) {
  "use strict";

  const PLANT = { inertia: 0.021, damping: 0.004, tau_max: 1.9, motor_lag: 0.035, disturbance: 0.05 };
  const SCENARIO = { target: (10 * Math.PI) / 180, duration: 3.0, rate_hz: 400, substeps: 4, disturbance_at: 1.5, initial: 0.0, gyro_noise: 0.0, angle_noise: 0.0, seed: 2463534242 };
  const GAINS = { kp: 1.5, ki: 0.0, kd: 0.0, imax: 1.0, d_cutoff_hz: 20.0, mode: "single", angle_p: 4.5, max_rate: (200 * Math.PI) / 180 };

  function xorshift32(state) {
    state ^= (state << 13) >>> 0;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0;
    return state >>> 0;
  }

  function makeNoise(seed) {
    let state = (seed >>> 0) || 1;
    return function () {
      state = xorshift32(state);
      const a = state / 4294967296.0;
      state = xorshift32(state);
      const b = state / 4294967296.0;
      return a + b - 1.0;
    };
  }

  function simulate(gainsIn, plantIn, scenarioIn) {
    const g = Object.assign({}, GAINS, gainsIn);
    const plant = Object.assign({}, PLANT, plantIn);
    const sc = Object.assign({}, SCENARIO, scenarioIn);
    const dt = 1.0 / sc.rate_hz;
    const h = dt / sc.substeps;
    const alpha = dt / (dt + 1.0 / (2.0 * Math.PI * g.d_cutoff_hz));
    let theta = sc.initial, omega = 0.0, tau = 0.0;
    let integ = 0.0, dFilt = 0.0, prevGyro = 0.0;
    const noise = makeNoise(sc.seed);
    const out = { t: [], angle: [], rate: [], torque: [], p: [], i: [], d: [], command: [] };
    const steps = Math.round(sc.duration * sc.rate_hz);
    for (let k = 0; k <= steps; k++) {
      const t = k * dt;
      const gyro = omega + sc.gyro_noise * noise();
      const angle = theta + sc.angle_noise * noise();
      let err, rawD;
      if (g.mode === "cascade") {
        const rateTarget = Math.max(-g.max_rate, Math.min(g.max_rate, g.angle_p * (sc.target - angle)));
        err = rateTarget - gyro;
        rawD = k > 0 ? -(gyro - prevGyro) / dt : 0.0;
      } else {
        err = sc.target - angle;
        rawD = -gyro;
      }
      prevGyro = gyro;
      const pTerm = g.kp * err;
      integ = Math.max(-g.imax, Math.min(g.imax, integ + g.ki * err * dt));
      dFilt += alpha * (rawD - dFilt);
      const dTerm = g.kd * dFilt;
      const u = Math.max(-plant.tau_max, Math.min(plant.tau_max, pTerm + integ + dTerm));
      out.t.push(t);
      out.angle.push(theta);
      out.rate.push(omega);
      out.torque.push(tau);
      out.p.push(pTerm);
      out.i.push(integ);
      out.d.push(dTerm);
      out.command.push(u);
      const dist = t >= sc.disturbance_at ? plant.disturbance : 0.0;
      for (let s = 0; s < sc.substeps; s++) {
        tau += ((u - tau) * h) / plant.motor_lag;
        omega += ((tau - plant.damping * omega + dist) / plant.inertia) * h;
        theta += omega * h;
      }
    }
    return out;
  }

  function metrics(tr, scenarioIn) {
    const sc = Object.assign({}, SCENARIO, scenarioIn);
    const step = sc.target - sc.initial;
    const t = tr.t, y = tr.angle;
    const before = [];
    for (let i = 0; i < t.length; i++) if (t[i] < sc.disturbance_at) before.push(i);
    const frac = (i) => (y[i] - sc.initial) / step;
    let t10 = null, t90 = null;
    for (const i of before) {
      if (t10 === null && frac(i) >= 0.1) t10 = t[i];
      if (t90 === null && frac(i) >= 0.9) t90 = t[i];
    }
    const rise = t10 !== null && t90 !== null ? t90 - t10 : null;
    let peak = 0.0;
    if (before.length) peak = Math.max(...before.map(frac));
    const band = 0.02 * Math.abs(step);
    let settle = 0.0;
    for (let n = before.length - 1; n >= 0; n--) {
      const i = before[n];
      if (Math.abs(y[i] - sc.target) > band) {
        settle = n + 1 < before.length ? t[i + 1] : null;
        break;
      }
    }
    const window = Math.max(1, Math.round(0.25 * sc.rate_hz));
    let sse = 0.0;
    for (const i of before.slice(-window)) sse += sc.target - y[i];
    sse /= window;
    let dse = 0.0;
    for (const v of y.slice(-window)) dse += sc.target - v;
    dse /= window;
    let itae = 0.0;
    for (let i = 0; i < t.length; i++) itae += t[i] * Math.abs(sc.target - y[i]);
    itae /= sc.rate_hz;
    let chatter = 0.0;
    const u = tr.command;
    for (let i = 1; i < u.length; i++) chatter += Math.abs(u[i] - u[i - 1]);
    chatter /= Math.max(1, u.length - 1);
    return {
      rise_time: rise,
      overshoot: Math.max(0.0, peak - 1.0),
      settling_time: settle,
      steady_state_error: sse,
      disturbance_error: dse,
      peak_torque: Math.max(...tr.torque.map(Math.abs)),
      itae: itae,
      chatter: chatter,
    };
  }

  const api = { simulate, metrics, PLANT, SCENARIO, GAINS };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }
  root.PID = api;

  // ------------------------------------------------------------------ UI --
  const app = document.getElementById("pid-app");
  if (!app) return;
  const svg = document.getElementById("pid-plot");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const sliders = {
    kp: document.getElementById("pid-kp"),
    ki: document.getElementById("pid-ki"),
    kd: document.getElementById("pid-kd"),
  };
  const readout = (id) => document.getElementById(id);
  const W = 720, H = 300, PAD = { l: 44, r: 12, t: 16, b: 30 };
  const LO = -5, HI = 25;
  let data = null;

  function x(t) { return PAD.l + ((W - PAD.l - PAD.r) * t) / SCENARIO.duration; }
  function y(deg) { return H - PAD.b - ((H - PAD.t - PAD.b) * (Math.max(LO, Math.min(HI, deg)) - LO)) / (HI - LO); }

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function axes() {
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    for (let v = LO; v <= HI; v += 5) {
      svg.append(el("line", { x1: PAD.l, x2: W - PAD.r, y1: y(v), y2: y(v), class: v === 10 ? "target" : "grid" }));
      svg.append(el("text", { x: PAD.l - 8, y: y(v) + 4, class: "tick", "text-anchor": "end" }, String(v)));
    }
    for (let s = 0; s <= SCENARIO.duration; s += 0.5) {
      svg.append(el("text", { x: x(s), y: H - 8, class: "tick", "text-anchor": "middle" }, `${s} s`));
    }
    svg.append(el("line", { x1: x(SCENARIO.disturbance_at), x2: x(SCENARIO.disturbance_at), y1: PAD.t, y2: H - PAD.b, class: "dist" }));
    svg.append(el("text", { x: x(SCENARIO.disturbance_at) + 6, y: PAD.t + 10, class: "tick" }, "gust"));
  }

  function path(ts, degs, cls) {
    let d = "";
    for (let i = 0; i < ts.length; i += 2) d += (i ? "L" : "M") + x(ts[i]).toFixed(1) + "," + y(degs[i]).toFixed(1);
    svg.append(el("path", { d, class: cls }));
  }

  function fmt(v, unit, scale) {
    if (v === null || v === undefined || !isFinite(v)) return "never";
    return (v * (scale || 1)).toFixed(scale === 100 ? 0 : 2) + unit;
  }

  function update() {
    const g = { kp: +sliders.kp.value, ki: +sliders.ki.value, kd: +sliders.kd.value };
    readout("pid-kp-v").textContent = g.kp.toFixed(2);
    readout("pid-ki-v").textContent = g.ki.toFixed(2);
    readout("pid-kd-v").textContent = g.kd.toFixed(3);
    const tr = simulate(g, PLANT, SCENARIO);
    const m = metrics(tr, SCENARIO);
    axes();
    if (data && app.dataset.ghost) {
      const p = data.presets[app.dataset.ghost];
      if (p) path(data.t, p.angle_deg, "ghost");
    }
    path(tr.t, tr.angle.map((a) => (a * 180) / Math.PI), "trace");
    readout("pid-rise").textContent = fmt(m.rise_time, " s");
    readout("pid-over").textContent = fmt(m.overshoot, "%", 100);
    readout("pid-settle").textContent = m.settling_time === null ? "not before the gust" : fmt(m.settling_time, " s");
    readout("pid-offset").textContent = (Math.abs(m.disturbance_error) * 180 / Math.PI).toFixed(2) + " deg";
    const unstable = tr.angle.some((a) => Math.abs(a) > 1.5) || m.overshoot > 1.5;
    app.classList.toggle("unstable", unstable);
  }

  function setGains(g, ghost) {
    sliders.kp.value = g.kp;
    sliders.ki.value = g.ki;
    sliders.kd.value = g.kd;
    if (ghost) app.dataset.ghost = ghost;
    update();
  }

  app.querySelectorAll("button[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      app.querySelectorAll("button[data-preset]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      const name = b.dataset.preset;
      if (name === "tuned" && data) setGains(data.tuned.gains, "");
      else if (data && data.presets[name]) setGains(data.presets[name].gains, "");
    })
  );
  Object.values(sliders).forEach((s) => s.addEventListener("input", () => {
    app.querySelectorAll("button[data-preset]").forEach((o) => o.setAttribute("aria-pressed", "false"));
    update();
  }));

  fetch("assets/week4/pid.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((json) => {
      data = json;
      setGains(json.presets.all_three.gains, "");
      drawSitl(json.airframe);
    })
    .catch(() => update());

  function drawSitl(a) {
    const box = document.getElementById("sitl-plot");
    if (!box || !a.sitl_step_t) return;
    const w = 1200, h = 320, pad = { l: 44, r: 36, t: 14, b: 28 };
    const T = 2.5;
    const sx = (t) => pad.l + ((w - pad.l - pad.r) * t) / T;
    const sy = (d) => h - pad.b - ((h - pad.t - pad.b) * (d + 1)) / 13;
    box.setAttribute("viewBox", `0 0 ${w} ${h}`);
    box.replaceChildren();
    for (let v = 0; v <= 12; v += 2) {
      box.append(el("line", { x1: pad.l, x2: w - pad.r, y1: sy(v), y2: sy(v), class: v === 10 ? "target" : "grid" }));
      box.append(el("text", { x: pad.l - 8, y: sy(v) + 4, class: "tick", "text-anchor": "end" }, String(v)));
    }
    for (let s = 0; s <= T; s += 0.5) box.append(el("text", { x: sx(s), y: h - 8, class: "tick", "text-anchor": "middle" }, `${s} s`));
    const line = (ts, ds, cls) => {
      let d = "";
      ts.forEach((t, i) => { if (t <= T) d += (d ? "L" : "M") + sx(t).toFixed(1) + "," + sy(ds[i]).toFixed(1); });
      box.append(el("path", { d, class: cls }));
    };
    line(a.sim_step_t, a.sim_step_deg, "trace");
    line(a.sitl_step_t, a.sitl_step_deg, "sitl");
  }
})(typeof window !== "undefined" ? window : globalThis);
