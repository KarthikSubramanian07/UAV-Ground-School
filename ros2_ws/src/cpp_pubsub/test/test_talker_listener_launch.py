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

"""Integration test: the C++ talker and listener from the tutorial."""

import sys
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest


@pytest.mark.launch_test
def generate_test_description():
    talker = Node(package='cpp_pubsub', executable='talker', output='screen')
    listener = Node(package='cpp_pubsub', executable='listener', output='screen')
    # A domain of its own, so tests that colcon runs in parallel cannot hear each other
    isolate = SetEnvironmentVariable('ROS_DOMAIN_ID', '52')
    return LaunchDescription([isolate, talker, listener, launch_testing.actions.ReadyToTest()]), {
        'talker': talker, 'listener': listener}


class TestTalkerListener(unittest.TestCase):

    def test_talker_publishes(self, proc_output, talker):
        proc_output.assertWaitFor("Publishing: 'Hello, world! 2'", process=talker, timeout=30)

    def test_listener_hears(self, proc_output, listener):
        proc_output.assertWaitFor("I heard: 'Hello, world! ", process=listener, timeout=30)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
