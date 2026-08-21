from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "curate_envelope_dataset.py"
SPEC = importlib.util.spec_from_file_location("curate_envelope_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
curation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curation)


def episode(index: int, *, group: int = 0, motion: str = "left", **changes):
    value = {
        "episode_index": index,
        "duration_sec": 20.0,
        "motion_class": motion,
        "left_departure_rad": 1.0 if motion != "right" else 0.01,
        "right_departure_rad": 1.0 if motion == "right" else 0.01,
        "initial_action_state_gap": 0.0,
        "initial_left_pose_distance_rad": 0.0,
        "initial_right_pose_distance_rad": 0.0,
        "left_close_relative_5mm": motion != "right",
        "right_close_relative_5mm": motion == "right",
        "videos": {"base": {"file_index": group}},
    }
    value.update(changes)
    return value


def label(**changes):
    value = {
        "auto_decision": "keep_candidate",
        "verification_status": "agreed",
        "task_success": "yes",
        "grasp_success": "yes",
        "actual_basket": "left",
        "quality_issue": "none",
        "vlm_evidence": "places the envelope into the left basket",
    }
    value.update(changes)
    return value


def test_objective_failure_is_excluded_even_with_visual_success():
    result = curation.classify_episode(episode(1, duration_sec=2.0), label())

    assert result["decision"] == "excluded"
    assert result["decision_basis"] == "objective_trajectory_failure"
    assert "duration_lt_5s" in result["reasons"]


def test_initial_pose_outlier_is_deferred_not_deleted():
    result = curation.classify_episode(episode(1, initial_left_pose_distance_rad=0.3), label())

    assert result["decision"] == "deferred"
    assert result["decision_basis"] == "initial_pose_outlier"


def test_right_route_is_rescued_from_size_classifier_bias():
    visual = label(
        auto_decision="exclude_candidate",
        task_success="no",
        actual_basket="none",
        quality_issue="incomplete",
        vlm_evidence=(
            "Right arm places the envelope into the right basket. Final view is unclear."
        ),
    )

    result = curation.classify_episode(episode(2, motion="right"), visual)

    assert result["decision"] == "selected"
    assert result["decision_basis"] == "right_route_trajectory_rescue"


def test_two_independent_explicit_failures_are_not_rescued():
    visual = label(
        auto_decision="exclude_candidate",
        task_success="no",
        actual_basket="none",
        quality_issue="failed_attempt",
        vlm_evidence=(
            "The envelope is not placed in either basket."
            " | The envelope remains on the table and is not placed in basket."
        ),
    )

    result = curation.classify_episode(episode(3, motion="right"), visual)

    assert result["decision"] == "deferred"
    assert result["explicit_failure_evidence_sources"] == 2


def test_grouped_split_has_no_group_or_episode_leakage():
    rows = []
    for group in range(7):
        for offset, motion in enumerate(("left", "right", "bimanual_or_mixed")):
            index = group * 10 + offset
            rows.append(
                {
                    "episode_index": index,
                    "source_group": group,
                    "motion_class": motion,
                }
            )

    splits, groups, score = curation.grouped_stratified_split(rows, (5.0, 1.0, 1.0))

    assert score >= 0
    assert set(groups["train"]).isdisjoint(groups["val"])
    assert set(groups["train"]).isdisjoint(groups["test"])
    assert set(groups["val"]).isdisjoint(groups["test"])
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert set().union(*map(set, splits.values())) == {row["episode_index"] for row in rows}
