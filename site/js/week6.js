/* Week 6 page: the takeoff lab, the SITL flight replay and the XRCE inspector.
   Data comes from docs/week06 (copied to assets/week6 by scripts/build_site.py).

   The lab runs a line by line port of ros2_ws/src/ugs_px4/ugs_px4/model.py (the
   fake PX4) and takeoff.py (the action server's logic) in the browser;
   tests/test_site.py runs both on the same scenarios and requires the same
   phases, commands and altitudes. The UI part only runs in a browser. */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- PX4 ----
  const ARMED = 2;
  const DISARMED = 1;
  const NAV = { POSCTL: 2, AUTO_LOITER: 4, OFFBOARD: 14, AUTO_TAKEOFF: 17, AUTO_LAND: 18 };
  const NAV_NAMES = { 0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "AUTO_LOITER", 5: "AUTO_RTL", 12: "DESCEND", 14: "OFFBOARD", 15: "STAB", 17: "AUTO_TAKEOFF", 18: "AUTO_LAND" };
  const CMD = { NAV_LAND: 21, NAV_TAKEOFF: 22, DO_SET_MODE: 176, COMPONENT_ARM_DISARM: 400 };
  const CMD_NAMES = { 21: "NAV_LAND", 22: "NAV_TAKEOFF", 176: "DO_SET_MODE", 400: "COMPONENT_ARM_DISARM" };
  const ACK_NAMES = { 0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED", 4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED" };
  const navName = (s) => NAV_NAMES[s] || `NAV_STATE_${s}`;
  const ackName = (r) => ACK_NAMES[r] || `RESULT_${r}`;
  const cmdName = (c) => CMD_NAMES[c] || `CMD_${c}`;

  const command = {
    arm: () => ({ command: CMD.COMPONENT_ARM_DISARM, params: [1, 0, 0, 0, 0, 0, 0] }),
    offboard: () => ({ command: CMD.DO_SET_MODE, params: [1, 6, 0, 0, 0, 0, 0] }),
    hold: () => ({ command: CMD.DO_SET_MODE, params: [1, 4, 3, 0, 0, 0, 0] }),
    takeoff: (amsl) => ({ command: CMD.NAV_TAKEOFF, params: [0, 0, 0, NaN, NaN, NaN, amsl] }),
    land: () => ({ command: CMD.NAV_LAND, params: [0, 0, 0, NaN, NaN, NaN, NaN] }),
  };

  // model.py constants (PX4 v1.17 defaults)
  const EKF_DELAY = 3.0;
  const PREFLIGHT_DELAY = 5.0;
  const COM_OF_LOSS_T = 1.0;
  const MPC_TKO_SPEED = 1.5;
  const MPC_LAND_SPEED = 0.7;
  const MPC_Z_P = 1.0;
  const MPC_Z_VEL_MAX_UP = 3.0;
  const MPC_Z_VEL_MAX_DN = 1.5;
  const MPC_ACC_UP_MAX = 4.0;
  const VELOCITY_TAU = 0.25;
  const COM_SPOOLUP_TIME = 1.0;
  const MPC_TKO_RAMP_T = 3.0;
  const RAMP_DEAD = 1.7;
  const COM_DISARM_LAND = 2.0;
  const COM_DISARM_PRFLT = 10.0;
  const REF_ALT = 488.0;

  class Px4Model {
    constructor(faults) {
      this.faults = {};
      for (const f of (faults || "").replace(/,/g, " ").split(/\s+/).filter(Boolean)) {
        const [k, v] = f.split("=");
        this.faults[k] = v === undefined ? "" : v;
      }
      this.t = 0.0;
      this.z = 0.0;
      this.vz = 0.0;
      this.arming_state = DISARMED;
      this.nav_state = NAV.AUTO_LOITER;
      this.offboard_seen = -Infinity;
      this.offboard_since = null;
      this.setpoint_z = null;
      this.setpoint_vz = 0.0;
      this.takeoff_z = null;
      this.hold_z = 0.0;
      this.armed_at = null;
      this.touchdown_at = null;
      this.climb_started = null;
      this.took_over = false;
      this.ramp_start = null;
      this.flew = false;
      this.acks = [];
    }

    has(f) {
      return Object.prototype.hasOwnProperty.call(this.faults, f);
    }

    offboardControlMode() {
      if (this.t - this.offboard_seen > COM_OF_LOSS_T) this.offboard_since = this.t;
      this.offboard_seen = this.t;
    }

    trajectorySetpoint(z, vz) {
      this.setpoint_z = z;
      this.setpoint_vz = Number.isNaN(vz) ? 0.0 : vz;
    }

    vehicleCommand(cmd, p) {
      const result = this.handle(cmd, p);
      if (!this.has("mute")) this.acks.push([this.t, cmd, result]);
    }

    handle(cmd, p) {
      const valid = this.t >= EKF_DELAY;
      if (cmd === CMD.COMPONENT_ARM_DISARM) {
        if (p[0] < 0.5) {
          if (this.landed) {
            this.disarm();
            return 0;
          }
          return 2;
        }
        if (this.has("deny_arm") || this.has("no_preflight") || this.t < PREFLIGHT_DELAY) return 2;
        if (this.arming_state !== ARMED) {
          this.arming_state = ARMED;
          this.armed_at = this.t;
          this.climb_started = null;
        }
        return 0;
      }
      if (cmd === CMD.DO_SET_MODE) {
        const main = Math.trunc(p[1]);
        const sub = Math.trunc(p[2]);
        if (main === 6) {
          const streaming = this.offboard_since !== null && this.t - this.offboard_seen < COM_OF_LOSS_T && this.t - this.offboard_since > 0.5;
          if (this.has("no_offboard") || !streaming || !valid) return 1;
          this.nav_state = NAV.OFFBOARD;
          return 0;
        }
        if (main === 4 && sub === 3 && valid) {
          this.nav_state = NAV.AUTO_LOITER;
          this.hold_z = this.z;
          return 0;
        }
        return 1;
      }
      if (cmd === CMD.NAV_TAKEOFF) {
        if (!valid) return 1;
        this.takeoff_z = !Number.isNaN(p[6]) ? -(p[6] - REF_ALT) : -2.5;
        this.nav_state = NAV.AUTO_TAKEOFF;
        return 0;
      }
      if (cmd === CMD.NAV_LAND) {
        if (!valid || this.arming_state !== ARMED) return 1;
        this.nav_state = NAV.AUTO_LAND;
        return 0;
      }
      return 3;
    }

    disarm() {
      this.arming_state = DISARMED;
      this.armed_at = null;
      this.touchdown_at = null;
      this.ramp_start = null;
      this.flew = false;
    }

    get landed() {
      return this.z > -0.05 && Math.abs(this.vz) < 0.1;
    }

    step(dt) {
      this.t += dt;
      const armed = this.arming_state === ARMED;
      if (this.nav_state === NAV.OFFBOARD && this.t - this.offboard_seen > COM_OF_LOSS_T) {
        this.nav_state = NAV.AUTO_LOITER;
        this.hold_z = this.z;
      }
      const takeover = this.faults.takeover_at;
      if (takeover && !this.took_over && this.climb_started !== null && this.t - this.climb_started > parseFloat(takeover)) {
        this.took_over = true;
        this.nav_state = NAV.POSCTL;
        this.hold_z = this.z;
      }
      let target = this.z;
      let ff = 0.0;
      let up = MPC_Z_VEL_MAX_UP;
      if (armed && this.nav_state === NAV.OFFBOARD && this.setpoint_z !== null) {
        target = this.setpoint_z;
        ff = this.setpoint_vz;
      } else if (armed && this.nav_state === NAV.AUTO_TAKEOFF && this.takeoff_z !== null) {
        target = this.takeoff_z;
        up = MPC_TKO_SPEED;
        if (Math.abs(this.z - this.takeoff_z) < 0.1) {
          this.nav_state = NAV.AUTO_LOITER;
          this.hold_z = this.takeoff_z;
        }
      } else if (armed && (this.nav_state === NAV.AUTO_LOITER || this.nav_state === NAV.POSCTL)) {
        target = this.hold_z;
      }
      if (armed && target < -0.2 && this.climb_started === null) this.climb_started = this.t;
      if (armed && this.landed && this.climb_started !== null && this.ramp_start === null && this.armed_at !== null && this.t - this.armed_at >= COM_SPOOLUP_TIME) this.ramp_start = this.t;
      if (armed && this.ramp_start === null && this.landed) up = 0.0;
      else if (armed && this.ramp_start !== null) {
        const ramp = (this.t - this.ramp_start - RAMP_DEAD) / (MPC_TKO_RAMP_T - RAMP_DEAD);
        up = up * Math.max(0.0, Math.min(1.0, ramp));
      }
      let desired = armed ? ff + MPC_Z_P * (target - this.z) : 0.0;
      if (armed && this.nav_state === NAV.AUTO_LAND) desired = MPC_LAND_SPEED;
      desired = Math.max(-up, Math.min(MPC_Z_VEL_MAX_DN, desired));
      let dv = (desired - this.vz) * Math.min(1.0, dt / VELOCITY_TAU);
      dv = Math.max(-MPC_ACC_UP_MAX * dt, Math.min(MPC_ACC_UP_MAX * dt, dv));
      this.vz += dv;
      this.z += this.vz * dt;
      if (this.z > 0.0) {
        this.z = 0.0;
        this.vz = Math.min(0.0, this.vz);
      }
      if (this.z < -0.3) this.flew = true;
      if (armed && this.landed && (this.nav_state === NAV.AUTO_LAND || this.flew)) {
        if (this.touchdown_at === null) this.touchdown_at = this.t;
        else if (this.t - this.touchdown_at > COM_DISARM_LAND) {
          this.disarm();
          this.nav_state = NAV.AUTO_LOITER;
        }
      } else if (armed && !this.landed) this.touchdown_at = null;
      if (armed && this.climb_started === null && this.armed_at !== null && this.t - this.armed_at > COM_DISARM_PRFLT) this.disarm();
    }

    snapshot(now) {
      const valid = this.t >= EKF_DELAY;
      const acks = this.acks;
      this.acks = [];
      return {
        status_time: now,
        position_time: now,
        arming_state: this.arming_state,
        nav_state: this.nav_state,
        preflight_ok: this.t >= PREFLIGHT_DELAY && !this.has("no_preflight"),
        xy_valid: valid,
        z_valid: valid,
        x: 0.0,
        y: 0.0,
        z: this.z,
        vz: this.vz,
        heading: 0.0,
        ref_alt: REF_ALT,
        landed: this.landed,
        acks,
        get armed() {
          return this.arming_state === ARMED;
        },
      };
    }
  }

  // ------------------------------------------------------ takeoff logic ----
  const STATUS_MAX_AGE = 5.0;
  const fresh = (s, now, maxAge) => s.status_time !== null && s.position_time !== null && now - s.position_time < maxAge && now - s.status_time < Math.max(maxAge, STATUS_MAX_AGE);

  function limits(changes) {
    return Object.assign(
      { climb_rate: 1.0, accel: 1.0, tolerance: 0.15, settled_speed: 0.15, settle_time: 1.0, max_lead: 0.4, stream_before_switch: 1.0, retry_every: 1.0, max_tries: 5, telemetry_timeout: 2.0, preflight_timeout: 60.0, timeout: 90.0 },
      changes || {}
    );
  }

  class Profile {
    constructor(z0, z1, v, a) {
      this.z0 = z0;
      this.z1 = z1;
      this.sign = z1 >= z0 ? 1.0 : -1.0;
      const d = Math.abs(z1 - z0);
      this.a = a;
      this.v = Math.min(v, Math.sqrt(d * a));
      this.t_acc = a > 0 ? this.v / a : 0.0;
      const dAcc = 0.5 * a * this.t_acc ** 2;
      this.t_cruise = this.v > 0 ? (d - 2 * dAcc) / this.v : 0.0;
      this.duration = 2 * this.t_acc + this.t_cruise;
    }

    sample(t) {
      if (t <= 0) return [this.z0, 0.0];
      if (t >= this.duration) return [this.z1, 0.0];
      const { a, v, t_acc: ta, t_cruise: tc } = this;
      let s;
      let speed;
      if (t < ta) {
        s = 0.5 * a * t * t;
        speed = a * t;
      } else if (t < ta + tc) {
        s = 0.5 * a * ta * ta + v * (t - ta);
        speed = v;
      } else {
        const tb = this.duration - t;
        s = Math.abs(this.z1 - this.z0) - 0.5 * a * tb * tb;
        speed = a * tb;
      }
      return [this.z0 + this.sign * s, this.sign * speed];
    }
  }

  class Commander {
    constructor(cmd, lim) {
      this.command = cmd;
      this.limits = lim;
      this.sent_at = null;
      this.tries = 0;
      this.last_ack = null;
    }

    poll(now, acks) {
      for (const [, c, result] of acks) if (c === this.command.command) this.last_ack = result;
      if (this.sent_at === null || now - this.sent_at >= this.limits.retry_every) {
        if (this.tries >= this.limits.max_tries) {
          const why = this.last_ack !== null ? `last ack ${ackName(this.last_ack)}` : "no ack";
          return [null, `PX4 did not act on ${cmdName(this.command.command)} after ${this.tries} tries (${why})`];
        }
        this.sent_at = now;
        this.tries += 1;
        return [this.command, null];
      }
      return [null, null];
    }
  }

  const output = (phase, setpoint, commands, done, message) => ({ phase, setpoint: setpoint || null, commands: commands || [], done: done === undefined ? null : done, message: message || "" });
  const sp = (x, y, z, vz, yaw) => ({ x, y, z, vz: vz === undefined ? NaN : vz, yaw: yaw === undefined ? NaN : yaw });

  class Takeoff {
    constructor(altitude, now, groundZ, method, lim) {
      this.altitude = altitude;
      this.method = method || "offboard";
      this.limits = lim || limits();
      this.ground_z = groundZ === undefined ? null : groundZ;
      this.start = now;
      this.phase = "WAIT_FOR_PX4";
      this.phase_since = now;
      this.hold = null;
      this.profile = null;
      this.climb_start = null;
      this.tau = 0.0;
      this.last_update = null;
      this.left_ground = null;
      this.settled_since = null;
      this.commander = null;
      this.result = null;
      this.climb_time = NaN;
    }

    enter(phase, now) {
      this.phase = phase;
      this.phase_since = now;
      this.commander = null;
    }

    targetZ() {
      return this.ground_z - this.altitude;
    }

    altitudeOf(s) {
      return (this.ground_z !== null ? this.ground_z : s.z) - s.z;
    }

    finish(done, message, setpoint) {
      this.result = output(this.phase, setpoint, [], done, message);
      return this.result;
    }

    cmd(c, now, s) {
      if (this.commander === null || this.commander.command.command !== c.command) this.commander = new Commander(c, this.limits);
      return this.commander.poll(now, s.acks);
    }

    update(now, s) {
      if (this.result !== null) return output(this.phase, this.result.setpoint, [], this.result.done, this.result.message);
      if (now - this.start > this.limits.timeout) return this.finish(false, `timed out after ${this.limits.timeout.toFixed(0)} s in ${this.phase}`, this.method === "offboard" ? this.hold : null);
      if (this.phase !== "WAIT_FOR_PX4" && !fresh(s, now, this.limits.telemetry_timeout)) return this.finish(false, "lost PX4 telemetry", null);

      if (this.phase === "WAIT_FOR_PX4") {
        if (!(fresh(s, now, this.limits.telemetry_timeout) && s.xy_valid && s.z_valid)) return output(this.phase);
        if (this.ground_z === null || s.landed) this.ground_z = s.z;
        this.hold = sp(s.x, s.y, s.z, NaN, s.heading);
        if (s.armed && !s.landed) return this.airborne(now, s);
        this.enter("PREFLIGHT", now);
      }
      const stream = this.method === "offboard" ? this.hold : null;

      if (this.phase === "PREFLIGHT") {
        if (s.preflight_ok || s.armed) this.enter(this.method === "offboard" ? "OFFBOARD" : "TAKEOFF_MODE", now);
        else if (now - this.phase_since > this.limits.preflight_timeout) return this.finish(false, `PX4's preflight checks did not pass within ${this.limits.preflight_timeout.toFixed(0)} s`, stream);
        else return output(this.phase, stream);
      }
      if (this.phase === "OFFBOARD") {
        if (s.nav_state === NAV.OFFBOARD) this.enter("ARMING", now);
        else if (now - this.phase_since < this.limits.stream_before_switch) return output(this.phase, stream);
        else {
          const [c, error] = this.cmd(command.offboard(), now, s);
          if (error) return this.finish(false, error, stream);
          return output(this.phase, stream, c ? [c] : []);
        }
      }
      if (this.phase === "TAKEOFF_MODE") {
        if (s.nav_state === NAV.AUTO_TAKEOFF) this.enter("ARMING", now);
        else {
          const amsl = s.ref_alt - this.ground_z + this.altitude;
          if (Number.isNaN(amsl)) return this.finish(false, "PX4 has no altitude reference (ref_alt)", null);
          const [c, error] = this.cmd(command.takeoff(amsl), now, s);
          if (error) return this.finish(false, error, null);
          return output(this.phase, null, c ? [c] : []);
        }
      }
      if (this.phase === "ARMING") {
        const wanted = this.method === "offboard" ? NAV.OFFBOARD : NAV.AUTO_TAKEOFF;
        if (s.nav_state !== wanted && !s.armed) return this.finish(false, `PX4 switched to ${navName(s.nav_state)} before arming`, stream);
        if (s.armed) return this.beginClimb(now, s);
        const [c, error] = this.cmd(command.arm(), now, s);
        if (error) return this.finish(false, error, stream);
        return output(this.phase, stream, c ? [c] : []);
      }
      return this.fly(now, s);
    }

    airborne(now, s) {
      if (this.method === "offboard" && s.nav_state !== NAV.OFFBOARD) {
        this.enter("OFFBOARD", now);
        this.phase_since = now - this.limits.stream_before_switch;
        return output(this.phase, this.hold);
      }
      return this.beginClimb(now, s);
    }

    beginClimb(now, s) {
      this.enter("CLIMB", now);
      this.climb_start = now;
      this.tau = 0.0;
      this.last_update = now;
      this.profile = new Profile(s.z, this.targetZ(), this.limits.climb_rate, this.limits.accel);
      return this.fly(now, s);
    }

    fly(now, s) {
      if (!s.armed) return this.finish(false, "PX4 disarmed during the climb", null);
      if (this.method === "offboard" && s.nav_state !== NAV.OFFBOARD) return this.finish(false, `PX4 left offboard for ${navName(s.nav_state)} (a failsafe or a pilot took over)`, null);
      if (this.method === "takeoff_mode" && s.nav_state !== NAV.AUTO_TAKEOFF && s.nav_state !== NAV.AUTO_LOITER) return this.finish(false, `PX4 left takeoff for ${navName(s.nav_state)}`, null);
      if (this.left_ground === null && (!s.landed || this.altitudeOf(s) > 0.3)) this.left_ground = now;
      let setpoint = null;
      if (this.method === "offboard") {
        // the reference waits whenever it is max_lead ahead of the vehicle
        let tau = this.tau + (now - this.last_update);
        this.last_update = now;
        let [z, vz] = this.profile.sample(tau);
        if (this.limits.max_lead > 0 && (z - s.z) * this.profile.sign > this.limits.max_lead) {
          tau = this.tau;
          [z, vz] = this.profile.sample(tau);
        }
        this.tau = tau;
        setpoint = sp(this.hold.x, this.hold.y, z, vz, this.hold.yaw);
        this.hold = sp(this.hold.x, this.hold.y, this.targetZ(), NaN, this.hold.yaw);
        if (this.tau >= this.profile.duration) this.phase = "SETTLE";
      } else if (s.nav_state === NAV.AUTO_LOITER) this.phase = "SETTLE";
      const error = Math.abs(this.altitudeOf(s) - this.altitude);
      const close = error < this.limits.tolerance && Math.abs(s.vz) < this.limits.settled_speed;
      if (this.phase === "SETTLE" && close) {
        if (this.settled_since === null) this.settled_since = now;
        if (now - this.settled_since >= this.limits.settle_time) {
          const start = this.left_ground !== null ? this.left_ground : this.climb_start;
          this.climb_time = this.settled_since - start;
          return this.finish(true, `holding ${this.altitudeOf(s).toFixed(2)} m (${this.climb_time.toFixed(1)} s climb)`, setpoint && this.hold);
        }
      } else this.settled_since = null;
      return output(this.phase, setpoint);
    }

    cancel(now, s) {
      if (this.method === "offboard") {
        if (this.hold === null) return this.finish(false, "canceled before PX4 was ready", null);
        this.hold = sp(this.hold.x, this.hold.y, s.z, NaN, this.hold.yaw);
        return this.finish(false, `canceled at ${this.altitudeOf(s).toFixed(2)} m, holding`, s.armed ? this.hold : null);
      }
      const out = this.finish(false, `canceled at ${this.altitudeOf(s).toFixed(2)} m, switching to hold`, null);
      if (s.armed) out.commands = [command.hold()];
      return out;
    }
  }

  class Land {
    constructor(now, groundZ, lim) {
      this.limits = lim || limits({ timeout: 120.0 });
      this.ground_z = groundZ === undefined ? null : groundZ;
      this.start = now;
      this.phase = "DESCEND";
      this.commander = null;
      this.touchdown = null;
      this.result = null;
    }

    altitudeOf(s) {
      return (this.ground_z !== null ? this.ground_z : s.z) - s.z;
    }

    update(now, s) {
      if (this.result !== null) return this.result;
      if (now - this.start > this.limits.timeout) return (this.result = output(this.phase, null, [], false, "timed out landing"));
      if (!fresh(s, now, this.limits.telemetry_timeout)) return (this.result = output(this.phase, null, [], false, "lost PX4 telemetry"));
      if (!s.armed) {
        this.phase = "DISARMED";
        const took = now - this.start;
        const message = took > 0.5 ? `landed and disarmed in ${took.toFixed(1)} s` : "already on the ground and disarmed";
        return (this.result = output(this.phase, null, [], s.landed, message));
      }
      if (s.landed) {
        this.phase = "TOUCHDOWN";
        if (this.touchdown === null) this.touchdown = now;
        return output(this.phase);
      }
      if (s.nav_state === NAV.AUTO_LAND) return output(this.phase);
      if (this.commander === null) this.commander = new Commander(command.land(), this.limits);
      const [c, error] = this.commander.poll(now, s.acks);
      if (error) return (this.result = output(this.phase, null, [], false, error));
      return output(this.phase, null, c ? [c] : []);
    }
  }

  /* One flight, the way the server and fake PX4 run it: logic every 20 ms,
     setpoints streamed every 5th step (10 Hz), the land goal after a hover.
     Returns a trace the lab draws and tests/test_site.py compares with Python. */
  function simulate(o) {
    const opt = Object.assign({ altitude: 5, method: "offboard", climb_rate: 1, faults: "", cancel_at: null, link_drop_at: null, hover: 3, land: true, until: 90 }, o || {});
    const DT = 0.02;
    const m = new Px4Model(opt.faults);
    const lim = limits({ climb_rate: opt.climb_rate });
    let logic = new Takeoff(opt.altitude, 0.0, null, opt.method, lim);
    let stage = "takeoff";
    let hoverUntil = null;
    let climbAt = null;
    const trace = { t: [], altitude: [], setpoint: [], phase: [], nav: [], events: [], result: null, land: null };
    let lastPhase = null;
    let stream = null;
    let t = 0.0;
    for (let i = 0; i < Math.round(opt.until / DT); i++) {
      const s = m.snapshot(t);
      for (const [, c, r] of s.acks) trace.events.push([t, "ack", `<- ${cmdName(c)}: ${ackName(r)}`]);
      let out;
      if (stage === "takeoff" && opt.cancel_at !== null && t >= opt.cancel_at && logic.result === null) out = logic.cancel(t, s);
      else out = logic.update(t, s);
      for (const c of out.commands) {
        trace.events.push([t, "cmd", `-> ${cmdName(c.command)}`]);
        m.vehicleCommand(c.command, c.params);
      }
      stream = out.setpoint;
      if (stage === "takeoff" && out.phase === "CLIMB" && climbAt === null) climbAt = t;
      const dropped = opt.link_drop_at !== null && climbAt !== null && t - climbAt >= opt.link_drop_at;
      if (stream !== null && i % 5 === 0 && !dropped) {
        m.offboardControlMode();
        m.trajectorySetpoint(stream.z, stream.vz);
      }
      if (i % 5 === 0) {
        trace.t.push(t);
        trace.altitude.push(-m.z);
        trace.setpoint.push(stream ? -stream.z : null);
        trace.phase.push(out.phase);
        trace.nav.push(m.nav_state);
      }
      if (out.phase !== lastPhase) {
        trace.events.push([t, "phase", `[${stage === "land" ? "land " : ""}${out.phase}]`]);
        lastPhase = out.phase;
      }
      if (stage === "takeoff" && out.done !== null) {
        trace.result = { t, done: out.done, message: out.message, altitude: -m.z };
        trace.events.push([t, out.done ? "ok" : "fail", (out.done ? "TAKEOFF succeeded: " : opt.cancel_at !== null && out.message.startsWith("canceled") ? "TAKEOFF canceled: " : "TAKEOFF aborted: ") + out.message]);
        if (!opt.land || (!out.done && !out.message.startsWith("canceled"))) {
          stage = "after";
          hoverUntil = t + 4;
        } else {
          stage = "hover";
          hoverUntil = t + opt.hover;
        }
      } else if (stage === "hover" && t >= hoverUntil) {
        stage = "land";
        logic = new Land(t, 0.0);
        trace.events.push([t, "cmd", "/land"]);
      } else if (stage === "land" && out.done !== null) {
        trace.land = { t, done: out.done, message: out.message };
        trace.events.push([t, out.done ? "ok" : "fail", (out.done ? "LAND succeeded: " : "LAND failed: ") + out.message]);
        break;
      } else if (stage === "after" && t >= hoverUntil) break;
      m.step(DT);
      t += DT;
    }
    return trace;
  }

  const api = { Px4Model, Takeoff, Land, Profile, simulate, limits, navName };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }
  root.Week6 = api;

  // ------------------------------------------------------------ helpers ----
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const motion = document.documentElement.classList.contains("motion");
  const get = (name) => fetch(`assets/week6/${name}.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

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

  const PHASE_COLORS = {
    WAIT_FOR_PX4: "var(--night-line)",
    PREFLIGHT: "oklch(0.45 0.05 220)",
    OFFBOARD: "var(--steel)",
    TAKEOFF_MODE: "var(--steel)",
    ARMING: "oklch(0.62 0.12 80)",
    CLIMB: "var(--signal)",
    SETTLE: "oklch(0.7 0.12 150)",
    DESCEND: "oklch(0.6 0.09 300)",
    TOUCHDOWN: "oklch(0.52 0.08 300)",
    DISARMED: "var(--night-line)",
  };

  function niceMax(v) {
    const steps = [2, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50];
    for (const s of steps) if (v <= s) return s;
    return Math.ceil(v / 10) * 10;
  }

  /* An altitude chart: phase bands along the bottom, setpoint (orange, dashed)
     and altitude (white) lines, axes in metres and seconds. Returns a function
     that moves the playhead. */
  function drawChart(svg, ts, alt, setp, phases, opts) {
    const W = Number(svg.viewBox.baseVal.width);
    const H = Number(svg.viewBox.baseVal.height);
    const m = { l: 44, r: 14, t: 14, b: 44 };
    const t1 = ts[ts.length - 1] || 1;
    const top = niceMax(Math.max(opts.target || 0, ...alt, ...setp.filter((v) => v !== null)) * 1.12);
    const x = (t) => m.l + ((W - m.l - m.r) * t) / t1;
    const y = (a) => H - m.b - ((H - m.t - m.b) * a) / top;
    svg.replaceChildren();
    const grid = el("g", { class: "w6-grid" });
    for (let a = 0; a <= top + 1e-9; a += top <= 12 ? 1 : 5) {
      grid.append(el("line", { x1: m.l, x2: W - m.r, y1: y(a), y2: y(a) }));
      grid.append(el("text", { x: m.l - 8, y: y(a) + 4, "text-anchor": "end" }, `${a}`));
    }
    const tick = t1 > 60 ? 10 : 5;
    for (let t = 0; t <= t1 + 1e-9; t += tick) grid.append(el("text", { x: x(t), y: H - m.b + 32, "text-anchor": "middle" }, `${t}s`));
    svg.append(grid);
    // phase band
    const band = el("g", { class: "w6-band" });
    let start = 0;
    for (let i = 1; i <= phases.length; i++) {
      if (i === phases.length || phases[i] !== phases[start]) {
        if (phases[start]) {
          const r = el("rect", { x: x(ts[start]), y: H - m.b + 6, width: Math.max(1, x(ts[Math.min(i, ts.length - 1)]) - x(ts[start])), height: 10, fill: PHASE_COLORS[phases[start]] || "var(--night-line)" });
          r.append(el("title", {}, phases[start]));
          band.append(r);
        }
        start = i;
      }
    }
    svg.append(band);
    if (opts.target) svg.append(el("line", { class: "w6-goal", x1: m.l, x2: W - m.r, y1: y(opts.target), y2: y(opts.target) }));
    const path = (vals) => {
      let d = "";
      let pen = false;
      vals.forEach((v, i) => {
        if (v === null || v === undefined) {
          pen = false;
          return;
        }
        d += `${pen ? "L" : "M"}${x(ts[i]).toFixed(1)} ${y(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };
    svg.append(el("path", { class: "w6-setpoint", d: path(setp) }));
    svg.append(el("path", { class: "w6-altitude", d: path(alt) }));
    if (opts.truth) svg.append(el("path", { class: "w6-truth", d: path(opts.truth) }));
    const head = el("line", { class: "w6-head", x1: m.l, x2: m.l, y1: m.t, y2: H - m.b });
    const dot = el("circle", { class: "w6-dot", r: 5, cx: m.l, cy: y(0) });
    svg.append(head, dot);
    return (i) => {
      if (i < 0) return;
      head.setAttribute("x1", x(ts[i]));
      head.setAttribute("x2", x(ts[i]));
      dot.setAttribute("cx", x(ts[i]));
      dot.setAttribute("cy", y(alt[i] || 0));
    };
  }

  function drawSide(svg, top) {
    const H = 340;
    const y = (a) => H - 34 - ((H - 64) * a) / top;
    svg.replaceChildren();
    svg.append(el("rect", { class: "w6-ground", x: 0, y: H - 34, width: 220, height: 34 }));
    for (let a = 0; a <= top; a += top <= 12 ? 1 : 5) svg.append(el("text", { class: "w6-tick", x: 8, y: y(a) + 4 }, `${a} m`));
    const goal = el("line", { class: "w6-goal", x1: 40, x2: 210, y1: 0, y2: 0 });
    const sp = el("line", { class: "w6-sp-mark", x1: 70, x2: 180, y1: 0, y2: 0 });
    const drone = el("g", { class: "w6-drone" });
    drone.append(el("rect", { x: -16, y: -5, width: 32, height: 10, rx: 3 }));
    drone.append(el("line", { x1: -34, x2: 34, y1: -7, y2: -7 }));
    for (const dx of [-34, 34]) drone.append(el("ellipse", { class: "rotor", cx: dx, cy: -10, rx: 16, ry: 2.6 }));
    drone.append(el("line", { x1: -10, x2: -14, y1: 5, y2: 13 }), el("line", { x1: 10, x2: 14, y1: 5, y2: 13 }));
    svg.append(goal, sp, drone);
    return (alt, setpoint, target, armed) => {
      goal.setAttribute("y1", y(target));
      goal.setAttribute("y2", y(target));
      sp.style.opacity = setpoint === null ? 0 : 1;
      if (setpoint !== null) {
        sp.setAttribute("y1", y(setpoint));
        sp.setAttribute("y2", y(setpoint));
      }
      drone.setAttribute("transform", `translate(125 ${y(alt) - 13})`);
      drone.classList.toggle("armed", !!armed);
    };
  }

  // ------------------------------------------------------------- replay ----
  if ($("flight-app")) {
    get("replay")
      .then(startReplay)
      .catch((e) => $("flight-log").replaceChildren(h("li", { text: `Could not load the flights (${e.message}).` })));
  }

  function fmt(v, digits, unit) {
    return v === null || v === undefined ? "n/a" : `${Number(v).toFixed(digits)}${unit || ""}`;
  }

  function startReplay(flights) {
    const pick = $("flight-pick");
    const play = $("flight-play");
    const scrub = $("flight-scrub");
    const clock = $("flight-clock");
    const log = $("flight-log");
    let speed = 4;
    let playing = false;
    let flight = null;
    let index = 0;
    let t = 0;
    let last = null;
    let move = null;
    let side = null;
    let shown = -1;

    function choose(k) {
      flight = flights[k];
      pick.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === k)));
      const rows = flight.rows;
      const ts = rows.map((r) => r[0]);
      const truth = rows.some((r) => r[7] !== null && r[7] !== undefined) ? rows.map((r) => (r[7] === undefined ? null : r[7])) : null;
      move = drawChart($("flight-chart"), ts, rows.map((r) => r[1]), rows.map((r) => r[2]), rows.map((r) => r[4]), { target: flight.target, truth });
      $("flight-legend").hidden = !truth;
      const top = niceMax(Math.max(flight.target, ...rows.map((r) => r[1])) * 1.12);
      side = drawSide($("flight-side"), top);
      const mm = flight.metrics;
      $("flight-metrics").replaceChildren(
        ...[
          ["result", mm.succeeded ? "succeeded" : mm.result ? mm.result.split(":")[0] : "n/a"],
          ["peak", fmt(mm.peak_altitude, 2, " m")],
          ["rise 10 to 90%", fmt(mm.rise_time, 2, " s")],
          ["overshoot", fmt(mm.overshoot, 2, " m")],
          ["hover RMS", fmt(mm.hover_rms_error, 3, " m")],
          ["drift", fmt(mm.max_drift, 2, " m")],
        ].map(([k, v]) => h("div", {}, [h("dt", { text: k }), h("dd", { text: v })]))
      );
      t = 0;
      shown = -1;
      log.replaceChildren();
      update();
    }

    function lastAt(rows, time) {
      let lo = 0;
      let hi = rows.length - 1;
      let best = 0;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (rows[mid][0] <= time) {
          best = mid;
          lo = mid + 1;
        } else hi = mid - 1;
      }
      return best;
    }

    function update() {
      const rows = flight.rows;
      const end = rows[rows.length - 1][0];
      index = lastAt(rows, t);
      const r = rows[index];
      move(index);
      side(r[1], r[2], flight.target, r[6] === 2);
      $("fr-phase").textContent = r[4] || "waiting";
      $("fr-nav").textContent = r[5] === null ? "" : navName(r[5]);
      $("fr-alt").textContent = `${r[1].toFixed(2)} m`;
      $("fr-sp").textContent = r[2] === null ? "none" : `${r[2].toFixed(2)} m`;
      $("fr-rate").textContent = `${r[3] >= 0 ? "+" : ""}${r[3].toFixed(2)} m/s`;
      $("fr-armed").textContent = r[6] === 2 ? "yes" : "no";
      scrub.value = String(Math.round((1000 * t) / end));
      clock.textContent = `${t.toFixed(1)} s`;
      // the log shows every event up to now, newest first
      const due = flight.events.filter((e) => e.t <= t);
      if (due.length !== shown) {
        shown = due.length;
        log.replaceChildren(
          ...due
            .slice(-40)
            .reverse()
            .map((e) => h("li", { class: /succeeded|COMPLETE/.test(e.msg) ? "ok" : /aborted|FAILED|canceled/.test(e.msg) ? "warn" : "" }, [h("span", { class: "mono t", text: `${e.t.toFixed(1)}` }), h("span", { class: "who", text: e.node.replace("takeoff_", "") }), h("span", { text: e.msg })]))
        );
      }
    }

    function frame(now) {
      if (!playing) return;
      if (last !== null) t = Math.min(flight.rows[flight.rows.length - 1][0], t + ((now - last) / 1000) * speed);
      last = now;
      update();
      if (t >= flight.rows[flight.rows.length - 1][0]) setPlaying(false);
      else requestAnimationFrame(frame);
    }

    function setPlaying(on) {
      playing = on;
      play.setAttribute("aria-pressed", String(on));
      play.textContent = on ? "Pause" : "Play";
      last = null;
      if (on) {
        if (t >= flight.rows[flight.rows.length - 1][0]) t = 0;
        requestAnimationFrame(frame);
      }
    }

    flights.forEach((f, k) => {
      const b = h("button", { class: "chip plain", type: "button", "aria-pressed": "false", text: f.title });
      b.addEventListener("click", () => {
        setPlaying(false);
        choose(k);
      });
      pick.append(b);
    });
    play.addEventListener("click", () => setPlaying(!playing));
    scrub.addEventListener("input", () => {
      t = (Number(scrub.value) / 1000) * flight.rows[flight.rows.length - 1][0];
      update();
    });
    document.querySelectorAll("#flight-app .speed button").forEach((b) =>
      b.addEventListener("click", () => {
        speed = Number(b.dataset.speed);
        document.querySelectorAll("#flight-app .speed button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      })
    );
    choose(0);
    // jump to the end of the climb so the static page shows the result
    const settled = flight.rows.findIndex((r) => r[4] === "SETTLE");
    if (settled > 0) {
      t = flight.rows[Math.min(flight.rows.length - 1, settled + 20)][0];
      update();
    }
  }

  // ---------------------------------------------------------------- lab ----
  if ($("lab-app")) startLab();

  function startLab() {
    const state = { method: "offboard", fault: "" };
    const alt = $("lab-alt");
    const rate = $("lab-rate");

    function scenario() {
      const o = { altitude: Number(alt.value), climb_rate: Number(rate.value), method: state.method };
      const f = state.fault;
      if (f === "cancel") o.cancel_at = 12;
      else if (f === "takeover") o.faults = "takeover_at=6";
      else if (f === "link_drop") o.link_drop_at = 6;
      else if (f === "no_preflight") {
        o.faults = f;
        o.until = 70;
      } else if (f) o.faults = f;
      return o;
    }

    function run() {
      $("lab-alt-out").textContent = `${alt.value} m`;
      $("lab-rate-out").textContent = `${Number(rate.value).toFixed(1)} m/s`;
      rate.disabled = state.method !== "offboard";
      const tr = simulate(scenario());
      const move = drawChart($("lab-chart"), tr.t, tr.altitude, tr.setpoint, tr.phase, { target: Number(alt.value) });
      move(tr.t.length - 1);
      const res = tr.result;
      const verdict = $("lab-verdict");
      if (!res) verdict.textContent = "Still running when the simulation stopped.";
      else {
        const landed = tr.land ? ` Then: ${tr.land.message}.` : "";
        verdict.className = `w6-verdict ${res.done ? "ok" : "warn"}`;
        verdict.textContent = `${res.done ? "Succeeded" : state.fault === "cancel" ? "Canceled" : "Aborted"} at ${res.t.toFixed(1)} s: ${res.message}.${landed}`;
      }
      $("lab-log").replaceChildren(
        ...tr.events.map((e) => h("li", { class: e[1] === "ok" ? "ok" : e[1] === "fail" ? "warn" : e[1] === "ack" ? "muted" : "" }, [h("span", { class: "mono t", text: e[0].toFixed(2) }), h("span", { text: e[2] })]))
      );
    }

    document.querySelectorAll("#lab-form [data-method]").forEach((b) =>
      b.addEventListener("click", () => {
        state.method = b.dataset.method;
        document.querySelectorAll("#lab-form [data-method]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        run();
      })
    );
    document.querySelectorAll("#lab-form [data-fault]").forEach((b) =>
      b.addEventListener("click", () => {
        state.fault = b.dataset.fault;
        document.querySelectorAll("#lab-form [data-fault]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        run();
      })
    );
    alt.addEventListener("input", run);
    rate.addEventListener("input", run);
    run();
  }

  // ---------------------------------------------------------- inspector ----
  if ($("xrce-app")) {
    get("xrce")
      .then((x) => startInspector(x.packets))
      .catch((e) => $("packet-head").replaceChildren(h("p", { text: `Could not load the capture (${e.message}).` })));
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
      readout.replaceChildren(h("b", { text: s.label }), h("span", { class: "val", text: s.value || "" }), h("span", { class: "range mono", text: `bytes ${s.start} to ${s.end - 1}, ${s.end - s.start} long` }));
    }

    function show(index) {
      const p = packets[index];
      list.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-current", String(j === index)));
      head.replaceChildren(
        h("h3", { text: p.title }),
        h("p", { text: p.text }),
        h("dl", { class: "facts" }, [h("dt", { text: "from" }), h("dd", { class: "mono", text: p.from }), h("dt", { text: "to" }), h("dd", { class: "mono", text: `${p.to}, ${p.port}` }), h("dt", { text: "decoded" }), h("dd", { class: "mono", text: p.summary.join("; ") })])
      );
      const raw = p.hex.match(/../g) || [];
      spans = p.spans;
      owner = new Array(raw.length).fill(-1);
      spans.forEach((s, k) => {
        for (let i = s.start; i < s.end && i < raw.length; i++) if (owner[i] < 0 || spans[owner[i]].depth <= s.depth) owner[i] = k;
      });
      bytes = raw.map(() => []);
      lit = -1;
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
      pinned = spans.findIndex((s) => s.depth === 2);
      light(pinned);
    }

    function byteAt(event) {
      const tgt = event.target.closest("[data-i]");
      return tgt ? owner[Number(tgt.dataset.i)] : -2;
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
    if (packets.length) show(0);
  }
})(typeof window !== "undefined" ? window : globalThis);
