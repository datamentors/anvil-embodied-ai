"""Tests for the mcap-exclude CLI (episodes moved out of a session before conversion)."""

from __future__ import annotations

import json

import pytest

from mcap_converter.cli.mcap_exclude import EXIT_BAD_INPUT, load_manual, main
from mcap_converter.core.reader import McapReader

LEFT = "/follower_l_forward_position_controller/commands"
RIGHT = "/follower_r_forward_position_controller/commands"
CHEST = "/cam_chest/image_raw/compressed"

CONFIG = f"""\
camera_topic_mapping:
  "{CHEST}": base
  "/cam_wrist_l/image_raw/compressed": left_wrist
action_from_observation: {{afo}}
action_topics:
  "{LEFT}":
    arm: left
    joint_order: [joint1]
  "{RIGHT}":
    arm: right
    joint_order: [joint1]
"""


@pytest.fixture
def session(tmp_path, monkeypatch):
    """Session builder: {episode: (camera fps, left commands, right commands, severity)}."""

    def build(episodes: dict, *, afo: bool = False):
        root = tmp_path / "session"
        entries, fps = [], {}
        for name, (cam_fps, left, right, severity) in episodes.items():
            folder = root / name
            folder.mkdir(parents=True)
            mcap = folder / f"{name}_0.mcap"
            mcap.write_bytes(b"stub")
            fps[name] = cam_fps
            entries.append(
                {
                    "path": str(mcap.resolve()),
                    "severity": severity,
                    "duration_s": 2.0,
                    "topics": [
                        {"topic": CHEST, "message_count": 60},
                        {"topic": LEFT, "message_count": left},
                        {"topic": RIGHT, "message_count": right},
                    ],
                }
            )
        report = root / "mcap_valid_reports" / "report.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"episodes": entries}))
        config = tmp_path / "config.yaml"
        config.write_text(CONFIG.format(afo=str(afo).lower()))
        monkeypatch.setattr(
            McapReader, "estimate_fps", lambda self, _topic: fps[self.mcap_path.parent.name]
        )
        return root, report, config

    return build


def run(tmp_path, root, report, config, *extra, fps=30):
    out = tmp_path / "work" / "exclusions.json"
    code = main(
        [
            "-i",
            str(root),
            "--report",
            str(report),
            "--config",
            str(config),
            "--fps",
            str(fps),
            "--excluded-dir",
            str(tmp_path / "work" / "excluded"),
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_clean_session_excludes_nothing(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "pass"), "0002": (29.6, 50, 50, "pass")})
    code, result = run(tmp_path, root, report, config)
    assert code == 0
    assert result == {"excluded": [], "kept": 2, "total": 2}


def test_low_fps_episode_is_moved_out(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "pass"), "0002": (24.1, 50, 50, "pass")})
    code, result = run(tmp_path, root, report, config)
    assert code == 0
    assert result["excluded"] == [{"episode": "0002", "reasons": ["low_fps"], "fps": 24}]
    assert (result["kept"], result["total"]) == (1, 2)
    assert not (root / "0002").exists()
    assert (tmp_path / "work" / "excluded" / "0002" / "0002_0.mcap").is_file()


def test_unknown_fps_counts_as_low(tmp_path, session):
    root, report, config = session({"0001": (None, 50, 50, "pass")})
    _, result = run(tmp_path, root, report, config)
    assert result["excluded"][0]["reasons"] == ["low_fps"]
    assert result["excluded"][0]["fps"] is None


def test_no_commands_any_moves_one_handed_episodes(tmp_path, session):
    root, report, config = session(
        {"0001": (30.0, 0, 50, "warning"), "0002": (30.0, 50, 50, "pass")}
    )
    _, result = run(tmp_path, root, report, config, "--no-commands", "any")
    assert result["excluded"] == [{"episode": "0001", "reasons": ["no_commands_left"], "fps": 30}]


def test_no_commands_all_keeps_one_handed_episodes(tmp_path, session):
    root, report, config = session(
        {"0001": (30.0, 0, 50, "warning"), "0002": (30.0, 0, 0, "warning")}
    )
    _, result = run(tmp_path, root, report, config, "--no-commands", "all")
    assert [e["episode"] for e in result["excluded"]] == ["0002"]
    assert result["excluded"][0]["reasons"] == ["no_commands_left", "no_commands_right"]
    assert (root / "0001").is_dir()


def test_command_topic_absent_from_report_counts_as_silent(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "pass")})
    payload = json.loads(report.read_text())
    payload["episodes"][0]["topics"] = [
        t for t in payload["episodes"][0]["topics"] if t["topic"] != RIGHT
    ]
    report.write_text(json.dumps(payload))
    _, result = run(tmp_path, root, report, config)
    assert result["excluded"][0]["reasons"] == ["no_commands_right"]


def test_action_from_observation_config_ignores_commands(tmp_path, session):
    root, report, config = session({"0001": (30.0, 0, 0, "warning")}, afo=True)
    _, result = run(tmp_path, root, report, config)
    assert result["excluded"] == []


def test_manual_list_with_reason(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "pass"), "0002": (30.0, 50, 50, "pass")})
    manual = tmp_path / "manual.tsv"
    manual.write_text("# episode\treason\n\n0002\tbad grasp\n0099\tnot in this session\n")
    _, result = run(tmp_path, root, report, config, "--manual", str(manual))
    assert result["excluded"] == [
        {"episode": "0002", "reasons": ["manual"], "fps": 30, "manual_reason": "bad grasp"}
    ]


def test_reasons_accumulate(tmp_path, session):
    root, report, config = session({"0001": (15.0, 0, 50, "warning")})
    manual = tmp_path / "manual.tsv"
    manual.write_text("0001\treviewed\n")
    _, result = run(tmp_path, root, report, config, "--manual", str(manual))
    assert result["excluded"][0]["reasons"] == ["low_fps", "no_commands_left", "manual"]


def test_critical_episodes_stay_for_the_converter(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "critical")})
    _, result = run(tmp_path, root, report, config)
    assert result == {"excluded": [], "kept": 1, "total": 1}
    assert (root / "0001").is_dir()


def test_everything_excluded_still_succeeds(tmp_path, session):
    root, report, config = session({"0001": (15.0, 50, 50, "pass"), "0002": (15.0, 50, 50, "pass")})
    code, result = run(tmp_path, root, report, config)
    assert code == 0
    assert (result["kept"], result["total"]) == (0, 2)


def test_missing_report_is_bad_input(tmp_path, session):
    root, report, config = session({"0001": (30.0, 50, 50, "pass")})
    report.unlink()
    code, result = run(tmp_path, root, report, config)
    assert code == EXIT_BAD_INPUT
    assert result is None


def test_mcap_missing_from_report_is_bad_input_and_moves_nothing(tmp_path, session):
    root, report, config = session({"0001": (15.0, 50, 50, "pass"), "0002": (30.0, 50, 50, "pass")})
    payload = json.loads(report.read_text())
    payload["episodes"] = payload["episodes"][1:]
    report.write_text(json.dumps(payload))
    code, result = run(tmp_path, root, report, config)
    assert code == EXIT_BAD_INPUT
    assert result is None
    assert (root / "0001").is_dir()


def test_clash_in_excluded_dir_is_bad_input(tmp_path, session):
    root, report, config = session({"0001": (15.0, 50, 50, "pass")})
    (tmp_path / "work" / "excluded" / "0001").mkdir(parents=True)
    code, _ = run(tmp_path, root, report, config)
    assert code == EXIT_BAD_INPUT
    assert (root / "0001" / "0001_0.mcap").is_file()


def test_excluded_dir_inside_the_session_is_bad_input(tmp_path, session):
    root, report, config = session({"0001": (15.0, 50, 50, "pass")})
    code = main(
        [
            "-i",
            str(root),
            "--report",
            str(report),
            "--config",
            str(config),
            "--fps",
            "30",
            "--excluded-dir",
            str(root / "excluded"),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == EXIT_BAD_INPUT


def test_load_manual_ignores_comments_and_blank_lines(tmp_path):
    manual = tmp_path / "m.tsv"
    manual.write_text("# header\n\n0003\tshaky\n0004\n")
    assert load_manual(manual) == {"0003": "shaky", "0004": ""}
