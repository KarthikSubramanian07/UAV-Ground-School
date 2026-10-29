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

"""Integration test: the C++ server and client from the tutorial."""

import sys
import unittest

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest


@pytest.mark.launch_test
def generate_test_description():
    server = Node(package='cpp_srvcli', executable='server', output='screen')
    # exactly as the tutorial runs it; a launch_ros Node would append --ros-args,
    # and the C++ client checks for exactly two arguments
    client = ExecuteProcess(cmd=['ros2', 'run', 'cpp_srvcli', 'client', '2', '3'], output='screen',
                            additional_env={'PYTHONUNBUFFERED': '1'})
    # A domain of its own, so tests that colcon runs in parallel cannot hear each other
    isolate = SetEnvironmentVariable('ROS_DOMAIN_ID', '54')
    return LaunchDescription([isolate, server, client, launch_testing.actions.ReadyToTest()]), {
        'server': server, 'client': client}


class TestServerClient(unittest.TestCase):

    def test_client_gets_the_sum(self, proc_output, client):
        proc_output.assertWaitFor('Sum: 5', process=client, timeout=30)

    def test_server_answers(self, proc_output, server):
        proc_output.assertWaitFor('sending back response: [5]', process=server, timeout=30)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_client_exit_code(self, proc_info, client):
        launch_testing.asserts.assertExitCodes(proc_info, process=client)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
