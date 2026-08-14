#!/usr/bin/env python3
"""Exercise the DDS endpoint contract without a ROS graph."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
CHECKER = DEPLOY_DIR / "check_ros_publishers.sh"

FAKE_ROS2 = r"""#!/usr/bin/env python3
import os
import sys

topic = sys.argv[-1]
mode = os.environ["FAKE_GRAPH_MODE"]
live_publishers = int(os.environ.get("FAKE_LIVE_PUBLISHERS", "0"))
debug_publishers = int(os.environ.get("FAKE_DEBUG_PUBLISHERS", "0"))

sensor_publishers = {
    "/joint_states": ["/joint_state_broadcaster"],
    "/cam_chest/image_raw/compressed": [
        "/cam_chest/cam_chest", "/cam_chest/cam_chest"
    ],
    "/cam_wrist_l/image_raw/compressed": [
        "/cam_wrist_l/cam_wrist_l", "/cam_wrist_l/cam_wrist_l"
    ],
    "/cam_wrist_r/image_raw/compressed": [
        "/cam_wrist_r/cam_wrist_r", "/cam_wrist_r/cam_wrist_r"
    ],
}
live_controllers = {
    "/follower_l_forward_position_controller/commands":
        "/follower_l_forward_position_controller",
    "/follower_r_forward_position_controller/commands":
        "/follower_r_forward_position_controller",
}

publishers = list(sensor_publishers.get(topic, []))
subscribers = []
if topic in live_controllers:
    if live_publishers:
        publisher = (
            "/unexpected_policy_publisher"
            if mode == "wrong_live_publisher"
            else "/lerobot_inference"
        )
        publishers = [publisher]
    else:
        publishers = []
    if mode == "missing_controller":
        subscribers = ["/replay_buffer"]
    elif mode == "no_live_subscriber":
        subscribers = []
    else:
        subscribers = [live_controllers[topic]]
        if mode == "duplicate_controller":
            subscribers.append(live_controllers[topic])
        if mode in {"with_replay", "unknown_debug"}:
            subscribers.append("/replay_buffer")
        if mode == "duplicate_replay":
            subscribers.extend(["/replay_buffer", "/replay_buffer"])
        if mode == "unknown_live":
            subscribers.append("/unexpected_command_consumer")
elif topic.startswith("/debug/"):
    publishers = ["/lerobot_inference"] if debug_publishers else []
    if mode in {"with_replay", "unknown_live", "missing_controller"}:
        subscribers = ["/replay_buffer"]
    elif mode == "unknown_debug":
        subscribers = ["/unexpected_debug_consumer"]

def endpoint(fqn, endpoint_type):
    namespace, name = fqn.rsplit("/", 1)
    namespace = namespace or "/"
    print(f"Node name: {name}")
    print(f"Node namespace: {namespace}")
    print(f"Endpoint type: {endpoint_type}")

print(f"Publisher count: {len(publishers)}")
for node in publishers:
    endpoint(node, "PUBLISHER")
print(f"Subscription count: {len(subscribers)}")
for node in subscribers:
    endpoint(node, "SUBSCRIPTION")
"""


class DdsAuthorityContractTest(unittest.TestCase):
    def _run(
        self,
        mode: str,
        *,
        live_publishers: int = 0,
        debug_publishers: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir)
            ros2 = bin_dir / "ros2"
            ros2.write_text(FAKE_ROS2, encoding="utf-8")
            ros2.chmod(ros2.stat().st_mode | stat.S_IXUSR)

            sleep = bin_dir / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            sleep.chmod(sleep.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "FAKE_GRAPH_MODE": mode,
                    "FAKE_LIVE_PUBLISHERS": str(live_publishers),
                    "FAKE_DEBUG_PUBLISHERS": str(debug_publishers),
                }
            )
            return subprocess.run(
                [
                    "bash",
                    str(CHECKER),
                    str(live_publishers),
                    str(debug_publishers),
                ],
                cwd=DEPLOY_DIR,
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )

    def test_passes_with_passive_replay_buffer(self) -> None:
        result = self._run("with_replay", debug_publishers=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DDS_AUTHORITY_PASS", result.stdout)

    def test_passes_without_passive_replay_buffer(self) -> None:
        result = self._run("without_replay")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("controller[+optional_replay_buffer]", result.stdout)

    def test_passes_live_owner_with_controller_only(self) -> None:
        result = self._run("without_replay", live_publishers=1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_passes_shadow_owner_without_replay_buffer(self) -> None:
        result = self._run("without_replay", debug_publishers=1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_unknown_live_subscriber(self) -> None:
        result = self._run("unknown_live")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected_command_consumer", result.stderr)

    def test_rejects_unknown_debug_subscriber(self) -> None:
        result = self._run("unknown_debug")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected_debug_consumer", result.stderr)

    def test_rejects_missing_controller_even_if_replay_is_present(self) -> None:
        result = self._run("missing_controller")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected=/follower_", result.stderr)

    def test_rejects_command_topic_without_controller_or_replay(self) -> None:
        result = self._run("no_live_subscriber")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("subscription_nodes=<none>", result.stderr)

    def test_rejects_duplicate_controller(self) -> None:
        result = self._run("duplicate_controller")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("subscription_nodes=/follower_", result.stderr)

    def test_rejects_duplicate_replay_buffer(self) -> None:
        result = self._run("duplicate_replay")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("/replay_buffer,/replay_buffer", result.stderr)

    def test_rejects_wrong_live_publisher_owner(self) -> None:
        result = self._run("wrong_live_publisher", live_publishers=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected_policy_publisher", result.stderr)


if __name__ == "__main__":
    unittest.main()
