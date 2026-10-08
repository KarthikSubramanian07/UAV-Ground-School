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
lab: measures what the slides claim about QoS and callbacks.

Run it with ``ros2 run ugs_drone lab --json lab.json``.

QoS experiments:
  reliability   every publisher and subscription reliability pair: matched?
                messages received? which incompatible QoS event fired?
  durability    a message published once, then late subscriptions join
  depth         a fast publisher and a slow subscription, for several queue depths

Callback experiments:
  blocking      a 20 Hz timer next to a subscription callback that takes 0.3 s,
                on each executor and callback group arrangement
  deadlock      client.call() inside a callback (single threaded: hangs), against
                call_async with a done callback, and a reentrant group on a
                multi threaded executor
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.event_handler import PublisherEventCallbacks, SubscriptionEventCallbacks
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, qos_check_compatible, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Int32
from std_srvs.srv import Trigger



def policy_name(kind):
    """Turn rmw_qos_policy_kind_t.RMW_QOS_POLICY_RELIABILITY into 'reliability'."""
    return str(kind).rsplit('POLICY_', 1)[-1].lower()


def profile(reliability='reliable', durability='volatile', depth=10, keep_all=False):
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_ALL if keep_all else HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE if reliability == 'reliable'
        else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if durability == 'transient_local'
        else DurabilityPolicy.VOLATILE)


class Spinner:
    """Spin some nodes on a background executor for the length of a with block."""

    def __init__(self, *nodes, executor=None):
        self.executor = executor or SingleThreadedExecutor()
        for n in nodes:
            self.executor.add_node(n)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.executor.shutdown()
        self.thread.join(timeout=2)


def wait_matched(endpoint, attr, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(endpoint, attr)() > 0:
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------ QoS --

def reliability_pairs(topic_id):
    rows = []
    for pub_r in ('reliable', 'best_effort'):
        for sub_r in ('reliable', 'best_effort'):
            topic = f'lab/reliability_{topic_id}_{pub_r}_{sub_r}'
            events = {'publisher': [], 'subscription': []}
            node = Node(f'lab_reliability_{pub_r}_{sub_r}')
            received = []
            pub = node.create_publisher(
                Int32, topic, profile(pub_r), event_callbacks=PublisherEventCallbacks(
                    incompatible_qos=lambda e: events['publisher'].append(
                        policy_name(e.last_policy_kind))))
            node.create_subscription(
                Int32, topic, lambda m: received.append(m.data), profile(sub_r),
                event_callbacks=SubscriptionEventCallbacks(
                    incompatible_qos=lambda e: events['subscription'].append(
                        policy_name(e.last_policy_kind))))
            with Spinner(node):
                matched = wait_matched(pub, 'get_subscription_count', 1.5)
                for i in range(20):
                    pub.publish(Int32(data=i))
                    time.sleep(0.01)
                time.sleep(0.5)
            check = qos_check_compatible(profile(pub_r), profile(sub_r))
            rows.append({
                'publisher': pub_r, 'subscription': sub_r, 'matched': matched,
                'sent': 20, 'received': len(received),
                'publisher_event': sorted(set(events['publisher'])),
                'subscription_event': sorted(set(events['subscription'])),
                'rclpy_check': check[1] or 'OK',
            })
            node.destroy_node()
    return rows


def durability():
    node = Node('lab_durability')
    rows = []
    for pub_d in ('transient_local', 'volatile'):
        pub = node.create_publisher(Int32, f'lab/mission_{pub_d}',
                                    profile(durability=pub_d, depth=1))
        pub.publish(Int32(data=7))  # published once, before anyone listens
        time.sleep(0.3)
        for sub_d in ('volatile', 'transient_local'):
            got = []
            node.create_subscription(Int32, f'lab/mission_{pub_d}', lambda m: got.append(m.data),
                                     profile(durability=sub_d, depth=1))
            with Spinner(node):
                time.sleep(1.0)
            rows.append({'publisher': pub_d, 'late_subscription': sub_d, 'received': len(got),
                         'rclpy_check': qos_check_compatible(
                             profile(durability=pub_d), profile(durability=sub_d))[1] or 'OK'})
    node.destroy_node()
    return rows


def settle(received, quiet=1.0, limit=15.0):
    """Wait until nothing new has arrived for `quiet` seconds."""
    end = time.monotonic() + limit
    count, last_change = len(received), time.monotonic()
    while time.monotonic() < end and time.monotonic() - last_change < quiet:
        time.sleep(0.05)
        if len(received) != count:
            count, last_change = len(received), time.monotonic()


def depth():
    rows = []
    for depth_n, keep_all in ((1, False), (10, False), (100, False), (0, True)):
        label = 'keep_all' if keep_all else f'keep_last {depth_n}'
        for reliability in ('reliable', 'best_effort'):
            pub_node = Node('lab_depth_pub')
            sub_node = Node('lab_depth_sub')
            qos = profile(reliability, depth=max(depth_n, 1), keep_all=keep_all)
            got = []

            def slow(msg, got=got):
                got.append(msg.data)
                time.sleep(0.01)  # 100 messages per second at most

            pub = pub_node.create_publisher(Int32, 'lab/depth', qos)
            sub_node.create_subscription(Int32, 'lab/depth', slow, qos)
            with Spinner(sub_node):
                wait_matched(pub, 'get_subscription_count')
                start = time.monotonic()
                for i in range(300):  # 1000 per second for 0.3 s
                    pub.publish(Int32(data=i))
                    time.sleep(max(0.0, start + (i + 1) * 0.001 - time.monotonic()))
                settle(got)
            rows.append({'history': label, 'reliability': reliability, 'sent': 300,
                         'received': len(got), 'in_order': got == sorted(got)})
            pub_node.destroy_node()
            sub_node.destroy_node()
    return rows


# ------------------------------------------------------------ callbacks --

def blocking():
    rows = []
    setups = (
        ('SingleThreadedExecutor', False, False),
        ('MultiThreadedExecutor, one group', True, False),
        ('MultiThreadedExecutor, separate groups', True, True),
    )
    for label, multi, separate in setups:
        node = Node('lab_blocking')
        ticks = []
        timer_group = MutuallyExclusiveCallbackGroup() if separate else None
        sub_group = MutuallyExclusiveCallbackGroup() if separate else None
        node.create_timer(0.05, lambda: ticks.append(time.monotonic()), callback_group=timer_group)
        node.create_subscription(Int32, 'lab/slow', lambda m: time.sleep(0.3), 10,
                                 callback_group=sub_group)
        feeder = Node('lab_blocking_feeder')
        pub = feeder.create_publisher(Int32, 'lab/slow', 10)
        executor = MultiThreadedExecutor(num_threads=4) if multi else SingleThreadedExecutor()
        with Spinner(node, executor=executor), Spinner(feeder):
            wait_matched(pub, 'get_subscription_count')
            end = time.monotonic() + 3.0
            while time.monotonic() < end:
                pub.publish(Int32(data=1))
                time.sleep(0.5)
        gaps = [1000 * (b - a) for a, b in zip(ticks, ticks[1:])]
        rows.append({'executor': label, 'timer_period_ms': 50, 'ticks': len(ticks),
                     'mean_gap_ms': round(statistics.mean(gaps), 1),
                     'max_gap_ms': round(max(gaps), 1)})
        node.destroy_node()
        feeder.destroy_node()
    return rows


class DeadlockNode(Node):
    """A service, and a timer whose callback calls that service."""

    def __init__(self, mode):
        super().__init__('lab_deadlock')
        self.mode = mode
        reentrant = ReentrantCallbackGroup() if mode == 'multi_reentrant' else None
        self.create_service(Trigger, 'lab/ping', self.on_ping, callback_group=reentrant)
        self.cli = self.create_client(Trigger, 'lab/ping', callback_group=reentrant)
        self.timer = self.create_timer(0.2, self.on_timer, callback_group=reentrant)
        self.result = None

    def on_ping(self, request, response):
        response.success = True
        response.message = 'pong'
        return response

    def on_timer(self):
        self.timer.cancel()
        self.cli.wait_for_service(timeout_sec=2.0)
        start = time.monotonic()
        if self.mode == 'async':
            future = self.cli.call_async(Trigger.Request())
            future.add_done_callback(lambda f: self.report(f.result(), start))
        else:
            self.report(self.cli.call(Trigger.Request()), start)  # blocks this callback

    def report(self, reply, start):
        self.result = {'reply': reply.message, 'ms': round(1000 * (time.monotonic() - start), 2)}
        print(json.dumps(self.result), flush=True)


def deadlock_child(mode):
    rclpy.init()
    node = DeadlockNode(mode)
    executor = MultiThreadedExecutor() if mode == 'multi_reentrant' else SingleThreadedExecutor()
    executor.add_node(node)
    while node.result is None:
        executor.spin_once(timeout_sec=0.1)
    rclpy.shutdown()


def deadlock():
    rows = []
    for mode, label in (('sync', 'call() in a callback, SingleThreadedExecutor'),
                        ('async', 'call_async() with a done callback, SingleThreadedExecutor'),
                        ('multi_reentrant', 'call() in a callback, MultiThreadedExecutor, '
                                            'reentrant group')):
        try:
            out = subprocess.run([sys.executable, '-m', 'ugs_drone.lab', '--deadlock-child', mode],
                                 capture_output=True, text=True, timeout=6)
            result = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append({'setup': label, 'outcome': 'returned', 'ms': result['ms']})
        except subprocess.TimeoutExpired:
            rows.append({'setup': label, 'outcome': 'deadlock (killed after 6 s)', 'ms': None})
    return rows


EXPERIMENTS = {'reliability': lambda: reliability_pairs(0), 'durability': durability,
               'depth': depth, 'blocking': blocking, 'deadlock': deadlock}


def main(argv=None):
    parser = argparse.ArgumentParser(description='QoS and callback experiments')
    parser.add_argument('experiments', nargs='*', default=list(EXPERIMENTS))
    parser.add_argument('--json', help='write the results here')
    parser.add_argument('--deadlock-child', help=argparse.SUPPRESS)
    args = parser.parse_args(argv if argv is not None else rclpy.utilities.remove_ros_args()[1:])
    if args.deadlock_child:
        return deadlock_child(args.deadlock_child)
    rclpy.init()
    results = {}
    try:
        for name in args.experiments:
            print(f'== {name}', flush=True)
            results[name] = EXPERIMENTS[name]()
            for row in results[name]:
                print('  ' + json.dumps(row), flush=True)
    finally:
        rclpy.try_shutdown()
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)
            f.write('\n')


if __name__ == '__main__':
    main()
