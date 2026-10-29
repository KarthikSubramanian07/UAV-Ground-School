"""ROS 2 quality of service: the presets, and whether a publisher and subscription will talk.

A port of ``rmw_dds_common::qos_profile_check_compatible`` (rmw_dds_common,
Jazzy), the function behind ``rclpy.qos.qos_check_compatible`` and the
"incompatible QoS" warnings ROS 2 prints. The rule is *offered must be at least
requested*: a best effort publisher cannot satisfy a reliable subscription, a
volatile one cannot satisfy transient local, a publisher must promise a deadline
and a liveliness lease at least as tight as the subscription asks for. When a
policy is ``system_default`` or unknown the answer depends on the middleware,
and the check returns a warning instead.

Durations are integer nanoseconds and 0 means "not set" (infinite), as in rmw.
The tests compare this module with rclpy on every combination of policies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

OK, WARNING, ERROR = "ok", "warning", "error"
RELIABILITY = ("system_default", "reliable", "best_effort", "unknown", "best_available")
DURABILITY = ("system_default", "transient_local", "volatile", "unknown", "best_available")
LIVELINESS = ("system_default", "automatic", "manual_by_topic", "unknown", "best_available")
HISTORY = ("system_default", "keep_last", "keep_all", "unknown")


@dataclass(frozen=True)
class QoSProfile:
    reliability: str = "reliable"
    durability: str = "volatile"
    history: str = "keep_last"
    depth: int = 10
    deadline_ns: int = 0
    lifespan_ns: int = 0
    liveliness: str = "system_default"
    lease_ns: int = 0

    def with_(self, **changes) -> QoSProfile:
        return replace(self, **changes)

    def describe(self) -> str:
        parts = [self.reliability, self.durability, f"{self.history} {self.depth}" if self.history == "keep_last" else self.history]
        if self.deadline_ns:
            parts.append(f"deadline {self.deadline_ns / 1e9:g} s")
        if self.liveliness not in ("system_default", "automatic") or self.lease_ns:
            parts.append(f"liveliness {self.liveliness}" + (f" lease {self.lease_ns / 1e9:g} s" if self.lease_ns else ""))
        return ", ".join(parts)


# rclpy.qos presets (rmw/qos_profiles.h).
PRESETS = {
    "default": QoSProfile(),
    "sensor_data": QoSProfile(reliability="best_effort", depth=5),
    "services_default": QoSProfile(),
    "parameters": QoSProfile(depth=1000),
    "parameter_events": QoSProfile(depth=1000),
    "system_default": QoSProfile(reliability="system_default", durability="system_default", history="system_default", depth=0),
    "best_available": QoSProfile(reliability="best_available", durability="best_available", liveliness="best_available"),
    "rosout": QoSProfile(durability="transient_local", depth=1000, lifespan_ns=10 * 10**9),
}


def _unknown(value: str) -> bool:
    return value in ("system_default", "unknown")


def check_compatible(pub: QoSProfile, sub: QoSProfile) -> tuple[str, str]:
    """Return ``(compatibility, reason)`` exactly as rmw does."""
    level, reasons = OK, []

    def error(text: str) -> None:
        nonlocal level
        level = ERROR
        reasons.append("ERROR: " + text + ";")

    def warning(text: str) -> None:
        nonlocal level
        level = WARNING
        reasons.append("WARNING: " + text + ";")

    if pub.reliability == "best_effort" and sub.reliability == "reliable":
        error("Best effort publisher and reliable subscription")
    if pub.durability == "volatile" and sub.durability == "transient_local":
        error("Volatile publisher and transient local subscription")
    if pub.deadline_ns == 0 and sub.deadline_ns != 0:
        error("Subscription has a deadline, but publisher does not")
    if pub.deadline_ns != 0 and sub.deadline_ns != 0 and sub.deadline_ns < pub.deadline_ns:
        error("Subscription deadline is less than publisher deadline")
    if pub.liveliness == "automatic" and sub.liveliness == "manual_by_topic":
        error("Publisher's liveliness is automatic and subscription's is manual by topic")
    if pub.lease_ns == 0 and sub.lease_ns != 0:
        error("Subscription has a liveliness lease duration, but publisher does not")
    if pub.lease_ns != 0 and sub.lease_ns != 0 and sub.lease_ns < pub.lease_ns:
        error("Subscription liveliness lease duration is less than publisher")

    if level == OK:
        pr, sr = pub.reliability, sub.reliability
        if _unknown(pr) and _unknown(sr):
            warning(f"Publisher reliability is {pr} and subscription reliability is {sr}")
        elif _unknown(pr) and sr == "reliable":
            warning(f"Reliable subscription, but publisher is {pr}")
        elif pr == "best_effort" and _unknown(sr):
            warning(f"Best effort publisher, but subscription is {sr}")
        pd, sd = pub.durability, sub.durability
        if _unknown(pd) and _unknown(sd):
            # (sic) rmw spells it "durabilty"
            warning(f"Publisher durabilty is {pd} and subscription durability is {sd}")
        elif _unknown(pd) and sd == "transient_local":
            warning(f"Transient local subscription, but publisher is {pd}")
        elif pd == "volatile" and _unknown(sd):
            warning(f"Volatile publisher, but subscription is {sd}")
        pl, sl = pub.liveliness, sub.liveliness
        if _unknown(pl) and _unknown(sl):
            warning(f"Publisher liveliness is {pl} and subscription liveliness is {sl}")
        elif _unknown(pl) and sl == "manual_by_topic":
            warning(f"Subscription's liveliness is manual by topic, but publisher's is {pl}")
        elif pl == "automatic" and _unknown(sl):
            warning(f"Publisher's liveliness is automatic, but subscription's is {sl}")
    return level, "".join(reasons)


def matrix(names: list[str] | None = None) -> list[dict]:
    """Every preset publisher against every preset subscription (for the docs and the site)."""
    names = names or ["default", "sensor_data", "system_default", "best_available", "rosout"]
    rows = []
    for p in names:
        for s in names:
            level, reason = check_compatible(PRESETS[p], PRESETS[s])
            rows.append({"publisher": p, "subscription": s, "compatibility": level, "reason": reason})
    return rows
