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

"""
Integration test: the tutorial's talker and listener, launched together.

The listener must hear consecutive "Hello World: N" messages that the talker
published, which proves the topic, its type and the default QoS all match.
"""

import re
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
    talker = Node(package='py_pubsub', executable='talker', output='screen',
                  additional_env={'PYTHONUNBUFFERED': '1'})
    listener = Node(package='py_pubsub', executable='listener', output='screen',
                    additional_env={'PYTHONUNBUFFERED': '1'})
    # A domain of its own, so tests that colcon runs in parallel cannot hear each other
    isolate = SetEnvironmentVariable('ROS_DOMAIN_ID', '51')
    return LaunchDescription([isolate, talker, listener, launch_testing.actions.ReadyToTest()]), {
        'talker': talker, 'listener': listener}


class TestTalkerListener(unittest.TestCase):

    def test_talker_publishes(self, proc_output, talker):
        proc_output.assertWaitFor('Publishing: "Hello World: 2"', process=talker, timeout=30)

    def test_listener_hears_consecutive_messages(self, proc_output, listener):
        three = re.compile(r'(I heard: "Hello World: \d+".*?){3}', re.S)
        proc_output.assertWaitFor(expected_output=three, process=listener, timeout=30)
        text = ''.join(event.text.decode() for event in proc_output[listener])
        heard = [int(n) for n in re.findall(r'I heard: "Hello World: (\d+)"', text)]
        self.assertEqual(heard[:3], list(range(heard[0], heard[0] + 3)))


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, 1, -2, -15])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
