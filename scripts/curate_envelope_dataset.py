#!/usr/bin/env python3
"""Build an automated, leakage-safe envelope curation and split manifest.

The source LeRobot dataset is never modified. The script combines objective
trajectory checks with multi-pass visual labels, preserves the under-recognized
right-arm branch, and groups episodes by continuous base-camera source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POLICY_NAME = "envelope-auto-curation-v1"
TASK_PROMPT = (
    "Turn the envelope face up if needed. Place large or medium envelopes in the "
    "left basket and small envelopes in the right basket."
)
MOTION_CLASSES = ("left", "right", "bimanual_or_mixed", "inactive")
SPLIT_NAMES = ("train", "val", "test")

_POSITIVE_RIGHT_PATTERNS = (
    re.compile(r"(?:place|places|placed|drop|drops|dropped|deposit|deposits).{0,80}right basket"),
    re.compile(r"right basket.{0,80}(?:visible|contains|inside|at the end)"),
)
_EXPLICIT_FAILURE_PATTERNS = (
    re.compile(r"not placed in (?:either|any|the) basket"),
    re.compile(r"not placed in basket"),
    re.compile(r"not in either basket"),
    re.compile(r"remains? on (?:the )?table"),
    re.compile(r"placed it on (?:the )?table"),
    re.compile(r"drops? it (?:back )?onto (?:the )?table"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_labels(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {int(row["episode_index"]): row for row in rows}


def evidence_votes(evidence: str) -> tuple[int, int]:
    """Count source-level positive-right and explicit-failure descriptions."""
    positive = 0
    failure = 0
    for source in evidence.lower().split(" | "):
        if any(pattern.search(source) for pattern in _POSITIVE_RIGHT_PATTERNS):
            positive += 1
        if any(pattern.search(source) for pattern in _EXPLICIT_FAILURE_PATTERNS):
            failure += 1
    return positive, failure


def technical_reasons(episode: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (hard exclusion reasons, conservative deferral reasons)."""
    exclude: list[str] = []
    defer: list[str] = []
    duration = float(episode["duration_sec"])
    max_departure = max(
        float(episode["left_departure_rad"]),
        float(episode["right_departure_rad"]),
    )
    if duration < 5.0:
        exclude.append("duration_lt_5s")
    if duration > 60.0:
        exclude.append("duration_gt_60s")
    if max_departure < 0.2:
        exclude.append("low_motion_lt_0p2rad")
    if float(episode.get("initial_action_state_gap", 0.0)) > 0.01:
        exclude.append("initial_action_state_gap_gt_0p01")
    if (
        max(
            float(episode.get("initial_left_pose_distance_rad", 0.0)),
            float(episode.get("initial_right_pose_distance_rad", 0.0)),
        )
        > 0.2
    ):
        defer.append("initial_pose_outlier_gt_0p2rad")
    return exclude, defer


def classify_episode(episode: dict[str, Any], label: dict[str, str]) -> dict[str, Any]:
    """Classify one episode as selected, deferred or excluded."""
    excluded, deferred = technical_reasons(episode)
    positive_right, explicit_failure = evidence_votes(label.get("vlm_evidence", ""))
    motion = episode["motion_class"]

    if excluded:
        decision = "excluded"
        basis = "objective_trajectory_failure"
    elif deferred:
        decision = "deferred"
        basis = "initial_pose_outlier"
    else:
        clear_success = label.get("auto_decision") == "keep_candidate" or (
            label.get("task_success") == "yes"
            and label.get("grasp_success") == "yes"
            and label.get("actual_basket") in {"left", "right"}
        )
        right_route_rescue = (
            motion == "right"
            and bool(episode.get("right_close_relative_5mm"))
            and positive_right >= 1
            and explicit_failure < 2
            and label.get("quality_issue") != "wrong_basket"
        )
        if clear_success:
            decision = "selected"
            basis = "visual_consensus_success"
        elif right_route_rescue:
            decision = "selected"
            basis = "right_route_trajectory_rescue"
        else:
            decision = "deferred"
            basis = "unresolved_semantic_outcome"

    return {
        "episode_index": int(episode["episode_index"]),
        "decision": decision,
        "decision_basis": basis,
        "reasons": excluded + deferred,
        "source_group": int(episode["videos"]["base"]["file_index"]),
        "motion_class": motion,
        "duration_sec": float(episode["duration_sec"]),
        "auto_decision": label.get("auto_decision", ""),
        "verification_status": label.get("verification_status", ""),
        "task_success": label.get("task_success", ""),
        "actual_basket": label.get("actual_basket", ""),
        "positive_right_evidence_sources": positive_right,
        "explicit_failure_evidence_sources": explicit_failure,
    }


def _assignment_score(
    assignment: tuple[int, ...],
    groups: list[int],
    grouped_rows: dict[int, list[dict[str, Any]]],
    ratios: tuple[float, float, float],
) -> float:
    all_rows = [row for rows in grouped_rows.values() for row in rows]
    total = len(all_rows)
    global_motion = Counter(row["motion_class"] for row in all_rows)
    ratio_total = sum(ratios)
    score = 0.0
    for split_idx, ratio in enumerate(ratios):
        rows = [
            row
            for group, assigned in zip(groups, assignment, strict=True)
            if assigned == split_idx
            for row in grouped_rows[group]
        ]
        if ratio > 0 and not rows:
            return math.inf
        if ratio == 0 and rows:
            return math.inf
        if not rows:
            continue
        target_fraction = ratio / ratio_total
        observed_fraction = len(rows) / total
        score += 4.0 * ((observed_fraction - target_fraction) / target_fraction) ** 2
        motion = Counter(row["motion_class"] for row in rows)
        for motion_class in MOTION_CLASSES[:-1]:
            global_fraction = global_motion[motion_class] / total
            observed = motion[motion_class] / len(rows)
            score += ((observed - global_fraction) ** 2) / max(global_fraction, 1e-9)
    return score


def grouped_stratified_split(
    selected_rows: list[dict[str, Any]], ratios: tuple[float, float, float]
) -> tuple[dict[str, list[int]], dict[str, list[int]], float]:
    """Assign entire base-video groups while matching size and motion mix."""
    grouped_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped_rows[row["source_group"]].append(row)
    groups = sorted(grouped_rows)
    active_splits = sum(value > 0 for value in ratios)
    if len(groups) < active_splits:
        raise ValueError(
            f"need at least {active_splits} source groups for split {ratios}, got {groups}"
        )
    if len(groups) > 12:
        raise ValueError("exhaustive grouped split supports at most 12 source groups")

    best_assignment: tuple[int, ...] | None = None
    best_score = math.inf
    for assignment in itertools.product(range(3), repeat=len(groups)):
        score = _assignment_score(assignment, groups, grouped_rows, ratios)
        if score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None or not math.isfinite(best_score):
        raise RuntimeError("could not find a valid grouped split assignment")

    split_episodes = {name: [] for name in SPLIT_NAMES}
    split_groups = {name: [] for name in SPLIT_NAMES}
    for group, split_idx in zip(groups, best_assignment, strict=True):
        name = SPLIT_NAMES[split_idx]
        split_groups[name].append(group)
        split_episodes[name].extend(row["episode_index"] for row in grouped_rows[group])
    for values in split_episodes.values():
        values.sort()
    return split_episodes, split_groups, best_score


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "duration_hours": round(sum(row["duration_sec"] for row in rows) / 3600, 3),
        "motion_class": dict(Counter(row["motion_class"] for row in rows)),
        "decision_basis": dict(Counter(row["decision_basis"] for row in rows)),
        "task_success_label": dict(Counter(row["task_success"] for row in rows)),
        "actual_basket_label": dict(Counter(row["actual_basket"] for row in rows)),
        "source_groups": sorted({row["source_group"] for row in rows}),
    }


def parse_ratio(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 3 or any(part < 0 for part in parts) or sum(parts) <= 0:
        raise argparse.ArgumentTypeError("split ratio must be three non-negative values")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-json", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--train-ready-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-ratio", type=parse_ratio, default=(5.0, 1.0, 1.0))
    arguments = parser.parse_args()

    episodes = read_json(arguments.episodes_json)
    labels = read_labels(arguments.labels_csv)
    train_ready = read_json(arguments.train_ready_json)
    expected_ids = {int(episode["episode_index"]) for episode in episodes}
    if expected_ids != set(labels):
        missing = sorted(expected_ids - set(labels))
        extra = sorted(set(labels) - expected_ids)
        raise ValueError(f"episode/label mismatch: missing={missing[:10]}, extra={extra[:10]}")

    decisions = [
        classify_episode(episode, labels[int(episode["episode_index"])]) for episode in episodes
    ]
    selected = [row for row in decisions if row["decision"] == "selected"]
    deferred = [row for row in decisions if row["decision"] == "deferred"]
    excluded = [row for row in decisions if row["decision"] == "excluded"]
    splits, split_groups, split_score = grouped_stratified_split(selected, arguments.split_ratio)
    by_id = {row["episode_index"]: row for row in decisions}

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "episode-decisions.jsonl"
    decisions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions))
    decisions_sha256 = hashlib.sha256(decisions_path.read_bytes()).hexdigest()

    split_info = {
        "schema_version": 1,
        "curation_policy": POLICY_NAME,
        "source_dataset_manifest_sha256": train_ready["facts"]["manifest_sha256"],
        "episode_decisions_sha256": decisions_sha256,
        "task_prompt": TASK_PROMPT,
        "split_group_key": "videos.base.file_index",
        "split_ratio": list(arguments.split_ratio),
        "split_assignment_score": split_score,
        "total_episodes": len(episodes),
        "selected_episodes": sorted(row["episode_index"] for row in selected),
        "deferred_episodes": sorted(row["episode_index"] for row in deferred),
        "excluded_episodes": sorted(row["episode_index"] for row in excluded),
        "train_episodes": splits["train"],
        "val_episodes": splits["val"],
        "test_episodes": splits["test"],
        "group_assignments": split_groups,
    }
    (output / "split_info.json").write_text(json.dumps(split_info, indent=2) + "\n")

    summary = {
        "policy": POLICY_NAME,
        "task_prompt": TASK_PROMPT,
        "source_dataset_manifest_sha256": train_ready["facts"]["manifest_sha256"],
        "all": summarize_rows(decisions),
        "selected": summarize_rows(selected),
        "deferred": summarize_rows(deferred),
        "excluded": summarize_rows(excluded),
        "splits": {
            name: summarize_rows([by_id[index] for index in splits[name]]) for name in SPLIT_NAMES
        },
        "group_assignments": split_groups,
        "split_assignment_score": split_score,
    }
    (output / "curation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
