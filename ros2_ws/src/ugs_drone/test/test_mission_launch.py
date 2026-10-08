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

"""Launch the whole drone and check the mission: services, action and topics together."""

import os
import sys
import unittest

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest

CANCEL = '0.0'  # no obstacle: the mission should find the target and land on it


@pytest.mark.launch_test
def generate_test_description():
    os.environ['PYTHONUNBUFFERED'] = '1'
    drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ugs_drone'), 'launch', 'drone.launch.py')),
        launch_arguments={'cancel_after_s': CANCEL}.items())
    # A domain of its own, so tests that colcon runs in parallel cannot hear each other
    isolate = SetEnvironmentVariable('ROS_DOMAIN_ID', '55')
    return LaunchDescription([isolate, drone, launch_testing.actions.ReadyToTest()])


class TestMission(unittest.TestCase):

    def test_mission_completes(self, proc_output):
        outcome = 'canceled' if float(CANCEL) > 0 else 'succeeded'
        expected = f'MISSION COMPLETE: {outcome}'
        proc_output.assertWaitFor(expected, timeout=120, stream='stderr')

    def test_used_every_pattern(self, proc_output):
        for text in ('/set_mode GUIDED: mode GUIDED', '/arm True: armed', 'feedback:',
                     'mission: 3 waypoints', '/arm False: disarmed'):
            proc_output.assertWaitFor(text, timeout=120, stream='stderr')
        if float(CANCEL) == 0:
            proc_output.assertWaitFor('target confirmed at', timeout=120, stream='stderr')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
