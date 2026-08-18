#!/usr/bin/env python3

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from prepare_robot_home import (
    HomingTransition,
    PoseSettler,
    evaluate_joint_sample,
    load_contract,
)

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE.parent / "configs" / "lerobot_control" / "robot_home_contract.json"


class RobotHomeGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(CONTRACT_PATH)
        cls.names = [joint.name for joint in cls.contract.joints]
        cls.positions = [joint.target for joint in cls.contract.joints]
        cls.velocities = [0.0] * len(cls.names)

    def test_contract_matches_both_eight_joint_home_targets(self) -> None:
        targets = {joint.name: joint.target for joint in self.contract.joints}
        self.assertEqual(len(targets), 16)
        self.assertEqual(targets["follower_l_joint2"], -0.174)
        self.assertEqual(targets["follower_l_joint4"], 1.5708)
        self.assertEqual(targets["follower_l_finger_joint1"], 0.045)
        self.assertEqual(targets["follower_r_joint2"], 0.174)
        self.assertEqual(targets["follower_r_joint4"], 1.5708)
        self.assertEqual(targets["follower_r_finger_joint1"], 0.045)

    def test_exact_home_pose_passes_in_any_joint_order(self) -> None:
        evaluation = evaluate_joint_sample(
            list(reversed(self.names)),
            list(reversed(self.positions)),
            list(reversed(self.velocities)),
            self.contract,
        )
        self.assertTrue(evaluation.ok, evaluation.reason)

    def test_pose_left_by_previous_live_run_is_rejected(self) -> None:
        positions = list(self.positions)
        positions[self.names.index("follower_l_joint7")] = -0.678454
        evaluation = evaluate_joint_sample(self.names, positions, self.velocities, self.contract)
        self.assertFalse(evaluation.ok)
        self.assertIn("follower_l_joint7 position error", evaluation.reason)

    def test_missing_duplicate_nonfinite_and_moving_samples_fail(self) -> None:
        cases = []
        cases.append(
            evaluate_joint_sample(
                self.names[:-1], self.positions[:-1], self.velocities[:-1], self.contract
            )
        )
        duplicate_names = list(self.names)
        duplicate_names[-1] = duplicate_names[0]
        cases.append(
            evaluate_joint_sample(duplicate_names, self.positions, self.velocities, self.contract)
        )
        nonfinite = list(self.positions)
        nonfinite[0] = float("nan")
        cases.append(evaluate_joint_sample(self.names, nonfinite, self.velocities, self.contract))
        moving = list(self.velocities)
        moving[0] = 0.2
        cases.append(evaluate_joint_sample(self.names, self.positions, moving, self.contract))
        self.assertTrue(all(not result.ok for result in cases))

    def test_stale_homed_status_cannot_satisfy_a_new_reset(self) -> None:
        transition = HomingTransition(self.contract)
        transition.observe("homed", False)
        self.assertFalse(transition.complete)
        transition.observe("homing_arms", False)
        transition.observe("homed", False)
        self.assertTrue(transition.complete)

    def test_dehome_and_failure_statuses_fail_closed(self) -> None:
        dehome = HomingTransition(self.contract)
        dehome.observe("homing_arms", True)
        self.assertIn("dehome", dehome.failure)

        failed = HomingTransition(self.contract)
        failed.observe("homing_failed", False)
        self.assertIn("failure state", failed.failure)

    def test_pose_must_remain_valid_for_duration_and_sample_count(self) -> None:
        settler = PoseSettler(self.contract)
        start = 10.0
        for index in range(self.contract.minimum_settle_samples):
            settler.observe(
                start + index * 0.01,
                self.names,
                self.positions,
                self.velocities,
            )
        self.assertFalse(settler.ready(start + 0.99))
        self.assertTrue(settler.ready(start + 1.0))

        invalid = list(self.positions)
        invalid[0] += 0.2
        settler.observe(start + 1.01, self.names, invalid, self.velocities)
        self.assertFalse(settler.ready(start + 2.0))
        self.assertEqual(settler.valid_samples, 0)

    def test_ros_gate_uses_transient_local_status_qos(self) -> None:
        import sys
        import types

        durability = types.SimpleNamespace(TRANSIENT_LOCAL="transient", VOLATILE="volatile")
        history = types.SimpleNamespace(KEEP_LAST="keep_last")
        reliability = types.SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")
        captured_qos = []

        class QoSProfile:
            def __init__(self, **kwargs):
                captured_qos.append(kwargs)

        node = MagicMock()
        node.create_client.return_value.wait_for_service.return_value = False
        rclpy = types.ModuleType("rclpy")
        rclpy.init = MagicMock()
        rclpy.shutdown = MagicMock()
        rclpy_node = types.ModuleType("rclpy.node")
        rclpy_node.Node = MagicMock(return_value=node)
        rclpy_qos = types.ModuleType("rclpy.qos")
        rclpy_qos.DurabilityPolicy = durability
        rclpy_qos.HistoryPolicy = history
        rclpy_qos.QoSProfile = QoSProfile
        rclpy_qos.ReliabilityPolicy = reliability
        anvil_msg = types.ModuleType("anvil_msgs.msg")
        anvil_msg.ArmsResetStatus = object
        anvil_srv = types.ModuleType("anvil_msgs.srv")
        anvil_srv.ResetArms = object
        sensor_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msg.JointState = object
        modules = {
            "rclpy": rclpy,
            "rclpy.node": rclpy_node,
            "rclpy.qos": rclpy_qos,
            "anvil_msgs.msg": anvil_msg,
            "anvil_msgs.srv": anvil_srv,
            "sensor_msgs.msg": sensor_msg,
        }
        with patch.dict(sys.modules, modules):
            from prepare_robot_home import run_ros_gate

            with self.assertRaisesRegex(RuntimeError, "reset service unavailable"):
                run_ros_gate(self.contract)

        self.assertEqual(captured_qos[0]["durability"], durability.TRANSIENT_LOCAL)
        self.assertEqual(captured_qos[0]["depth"], 1)


if __name__ == "__main__":
    unittest.main()
