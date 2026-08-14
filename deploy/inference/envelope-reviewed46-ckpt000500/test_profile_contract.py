"""Repository contract checks for the checkpoint-specific deployment profile."""

from __future__ import annotations

from pathlib import Path

import yaml

PROFILE = Path(__file__).resolve().parent


def test_runtime_example_contains_no_host_specific_values() -> None:
    payload = (PROFILE / "runtime.env.example").read_text()

    assert "/home/" not in payload
    assert "192.168." not in payload
    assert "100.116." not in payload
    for name in ("DDS_IFACE", "DDS_LOCAL_IP", "DDS_PEER_IP"):
        assert f"{name}=" in payload


def test_dds_template_is_multicast_free_and_parameterized() -> None:
    payload = (PROFILE / "cyclonedds_two_pc_gpu.xml.template").read_text()

    assert "<AllowMulticast>false</AllowMulticast>" in payload
    assert "<ParticipantIndex>auto</ParticipantIndex>" in payload
    assert "<MaxAutoParticipantIndex>31</MaxAutoParticipantIndex>" in payload
    assert payload.count("<Peer address=") == 2
    assert "@DDS_IFACE@" in payload
    assert "@DDS_LOCAL_IP@" in payload
    assert "@DDS_PEER_IP@" in payload


def test_shadow_and_live_topics_cannot_be_confused() -> None:
    shadow = yaml.safe_load((PROFILE / "inference_envelope_ckpt000500_shadow.yaml").read_text())
    live = yaml.safe_load((PROFILE / "inference_envelope_ckpt000500_live.yaml").read_text())

    assert all(arm["command_topic"].startswith("/debug/") for arm in shadow["arms"].values())
    assert all(not arm["command_topic"].startswith("/debug/") for arm in live["arms"].values())
    assert live["safety"]["saturate_all_raw_targets_to_joint_limits"] is True
    assert live["safety"]["allow_live_joint_limit_saturation"] is True
    assert live["safety"]["joint_position_limits"] == shadow["safety"]["joint_position_limits"]


def test_live_runner_keeps_both_operator_confirmations_and_homing() -> None:
    payload = (PROFILE / "_run_mode.sh").read_text()

    assert "LIVE_ROBOT_CONFIRM" in payload
    assert "HOME AND RUN CKPT000500 LIVE" in payload
    assert "require_robot_home" in payload
    assert 'require_graph_contract 0 0 "after homing before live startup"' in payload
