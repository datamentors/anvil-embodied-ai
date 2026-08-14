#!/usr/bin/env python3
"""Passively prove that both isolated shadow command streams are healthy."""

from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

TOPICS = (
    "/debug/follower_l_forward_position_controller/commands",
    "/debug/follower_r_forward_position_controller/commands",
)
MIN_SAMPLES = 60
TIMEOUT_SECONDS = 20.0
MIN_RATE_HZ = 25.0
MAX_RATE_HZ = 35.0


class ShadowFlowCheck(Node):
    def __init__(self) -> None:
        super().__init__("shadow_flow_check")
        self.arrivals: dict[str, list[float]] = {topic: [] for topic in TOPICS}
        self.invalid: list[str] = []
        self._subscriptions_keepalive = [
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, topic=topic: self._on_message(topic, msg),
                10,
            )
            for topic in TOPICS
        ]

    def _on_message(self, topic: str, msg: Float64MultiArray) -> None:
        values = tuple(float(value) for value in msg.data)
        if len(values) != 8:
            self.invalid.append(f"{topic}: action dimension {len(values)} != 8")
            return
        if not all(math.isfinite(value) for value in values):
            self.invalid.append(f"{topic}: action contains NaN/Inf")
            return
        self.arrivals[topic].append(time.monotonic())


def main() -> int:
    rclpy.init(args=None)
    node = ShadowFlowCheck()
    deadline = time.monotonic() + TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.invalid:
                print(f"ERROR: {node.invalid[0]}", file=sys.stderr)
                return 1
            if all(len(node.arrivals[topic]) >= MIN_SAMPLES for topic in TOPICS):
                break

        summaries: list[str] = []
        for topic in TOPICS:
            arrivals = node.arrivals[topic]
            if len(arrivals) < MIN_SAMPLES:
                print(
                    f"ERROR: {topic} delivered only {len(arrivals)}/{MIN_SAMPLES} "
                    f"valid shadow actions in {TIMEOUT_SECONDS:.0f}s",
                    file=sys.stderr,
                )
                return 1
            duration = arrivals[-1] - arrivals[0]
            rate = (len(arrivals) - 1) / duration if duration > 0 else math.inf
            if not MIN_RATE_HZ <= rate <= MAX_RATE_HZ:
                print(
                    f"ERROR: {topic} rate {rate:.2f} Hz outside "
                    f"[{MIN_RATE_HZ:.1f}, {MAX_RATE_HZ:.1f}] Hz",
                    file=sys.stderr,
                )
                return 1
            summaries.append(f"{topic}={len(arrivals)}@{rate:.2f}Hz")

        print("SHADOW_FLOW_PASS " + " ".join(summaries))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
