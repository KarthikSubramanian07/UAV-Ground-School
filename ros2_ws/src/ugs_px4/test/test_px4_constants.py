# Copyright 2026 Karthik Subramanian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ugs_px4.px4 copies PX4's numbers so its logic runs without px4_msgs: check them."""

from px4_msgs.msg import VehicleCommand, VehicleCommandAck, VehicleStatus
from ugs_px4 import px4


def test_nav_states_match_px4_msgs():
    for value, name in px4.NAV_STATES.items():
        assert getattr(VehicleStatus, f'NAVIGATION_STATE_{name}') == value


def test_commands_and_states_match_px4_msgs():
    assert px4.CMD_NAV_LAND == VehicleCommand.VEHICLE_CMD_NAV_LAND
    assert px4.CMD_NAV_TAKEOFF == VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF
    assert px4.CMD_DO_SET_MODE == VehicleCommand.VEHICLE_CMD_DO_SET_MODE
    assert px4.CMD_COMPONENT_ARM_DISARM == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    assert px4.ARMING_STATE_ARMED == VehicleStatus.ARMING_STATE_ARMED
    assert px4.ARMING_STATE_DISARMED == VehicleStatus.ARMING_STATE_DISARMED


def test_ack_results_match_px4_msgs():
    for value, name in px4.ACK_RESULTS.items():
        assert getattr(VehicleCommandAck, f'VEHICLE_CMD_RESULT_{name}') == value
