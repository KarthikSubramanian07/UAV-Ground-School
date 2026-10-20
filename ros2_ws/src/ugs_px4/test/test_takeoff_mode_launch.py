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

"""Take off with PX4's own takeoff mode instead of offboard setpoints."""

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

ARGS = {'fake': 'true', 'altitude': '5.0', 'method': 'takeoff_mode', 'land': 'true',
        'hover': '1.0'}
EXPECT = ['-> PX4 NAV_TAKEOFF', 'feedback: [CLIMB]', 'TAKEOFF succeeded', 'LAND succeeded',
          'FLIGHT COMPLETE']


@pytest.mark.launch_test
def generate_test_description():
    os.environ['PYTHONUNBUFFERED'] = '1'
    flight = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ugs_px4'), 'launch', 'takeoff.launch.py')),
        launch_arguments=ARGS.items())
    # a domain of its own, so tests that colcon runs in parallel cannot hear each other
    isolate = SetEnvironmentVariable('ROS_DOMAIN_ID', '62')
    return LaunchDescription([isolate, flight, launch_testing.actions.ReadyToTest()])


class TestFlight(unittest.TestCase):

    def test_flight(self, proc_output):
        for text in EXPECT:
            proc_output.assertWaitFor(text, timeout=120, stream='stderr')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
