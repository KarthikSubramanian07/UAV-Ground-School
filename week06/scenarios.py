"""The takeoff lab's scenarios, shared by week06.lab, the site build and the tests.

Kept free of imports so the site build (which runs from an installed package,
without ros2_ws beside it) can count them.
"""

SCENARIOS = {
    "takeoff": {},
    "takeoff_mode": {"method": "takeoff_mode"},
    "fast": {"altitude": 20, "climb_rate": 3},
    "cancel": {"altitude": 10, "cancel_at": 12},
    "deny_arm": {"faults": "deny_arm"},
    "no_offboard": {"faults": "no_offboard"},
    "takeover": {"altitude": 10, "faults": "takeover_at=6"},
    "link_drop": {"altitude": 10, "link_drop_at": 6},
    "no_preflight": {"faults": "no_preflight", "until": 70},
}
