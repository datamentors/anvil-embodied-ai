"""Stratified episode-split helpers (companion to :mod:`anvil_shared.splits`).

``splits.compute_split_episodes`` shuffles *all* episodes together, so the
train/val/test proportions of any sub-population (e.g. small face-down
envelopes) are only correct on average.  When a sub-population is small, a
global shuffle can leave it absent from val or test entirely.

This module splits *within* each stratum instead: every stratum contributes its
own 80/10/10 (or whatever ratio is asked for), so each split mirrors the overall
composition of the dataset.

Each episode must belong to exactly one stratum — the caller supplies a
``{episode_index: stratum_key}`` mapping.  That makes the three resulting splits
disjoint by construction: strata are disjoint, and within a stratum an episode is
placed in exactly one of train/val/test.

Public API:
    compute_stratified_split_episodes(labels, ratio, seed) -> dict[str, list[int]]
    summarize_strata(labels, splits) -> dict[str, dict[str, int]]
    validate_split_info(split_info, total_episodes) -> list[str]
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from random import Random

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def _stratum_seed(seed: int, stratum: str) -> int:
    """Derive a per-stratum seed from ``seed`` and the stratum name.

    Seeding each stratum independently means the assignment inside one stratum
    does not depend on how many episodes the *other* strata happen to have.  Add
    a batch of new small envelopes and the big-envelope split stays put; only the
    stratum that actually grew is reshuffled.

    ``hashlib`` is used rather than ``hash()`` because the latter is salted per
    process for str inputs, which would make splits irreproducible across runs.
    """
    digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _allocate(n: int, r: Sequence[float]) -> tuple[int, int, int]:
    """Split a count of ``n`` items into (train, val, test) sizes.

    Mirrors the rounding of ``splits.compute_split_episodes`` — allocate test and
    val by rounding, give train the remainder — so the two strategies stay
    comparable.
    """
    total_ratio = sum(r)
    n_test = round(n * r[2] / total_ratio) if r[2] > 0 else 0
    n_val = round(n * r[1] / total_ratio) if r[1] > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 0:  # pathological ratio; clamp and let the caller's warning speak
        n_train = 0
        n_val = min(n_val, n)
        n_test = max(0, n - n_val)
    return n_train, n_val, n_test


def compute_stratified_split_episodes(
    labels: Mapping[int, str],
    ratio: Sequence[float],
    seed: int = 42,
) -> dict[str, list[int]]:
    """Deterministic stratified 3-way episode split.

    Args:
        labels: ``{episode_index: stratum_key}`` covering every episode to split.
            One key per episode — the caller is responsible for rejecting
            episodes that belong to several strata.
        ratio: 2- or 3-element ``[train, val, test]``.  A 2-element sequence is
            treated as ``[train, val, 0]``.
        seed: Base RNG seed; combined with each stratum name (see
            :func:`_stratum_seed`).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` — sorted, disjoint
        episode-index lists whose union is ``labels.keys()``.

    Raises:
        ValueError: empty ``labels``, or a ratio that is malformed or sums to 0.
    """
    if not labels:
        raise ValueError("labels must not be empty")

    r = list(ratio)
    if len(r) == 2:
        r.append(0.0)
    elif len(r) != 3:
        raise ValueError(f"ratio must have 2 or 3 elements, got {len(r)}")
    if any(x < 0 for x in r):
        raise ValueError(f"ratio must be non-negative, got {r}")
    if sum(r) <= 0:
        raise ValueError(f"ratio must sum to > 0, got {r}")

    by_stratum: dict[str, list[int]] = defaultdict(list)
    for ep, stratum in labels.items():
        by_stratum[str(stratum)].append(int(ep))

    out: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}

    for stratum in sorted(by_stratum):  # sorted → stable iteration order
        eps = sorted(by_stratum[stratum])
        n_train, n_val, n_test = _allocate(len(eps), r)

        shuffled = list(eps)
        Random(_stratum_seed(seed, stratum)).shuffle(shuffled)

        out["train"].extend(shuffled[:n_train])
        out["val"].extend(shuffled[n_train : n_train + n_val])
        out["test"].extend(shuffled[n_train + n_val :])

        if (r[1] > 0 and n_val == 0) or (r[2] > 0 and n_test == 0):
            log.warning(
                "[stratified] stratum %r has only %d episode(s): ratio %s yields "
                "train=%d val=%d test=%d — this stratum is absent from at least one split",
                stratum, len(eps), r, n_train, n_val, n_test,
            )

    return {name: sorted(out[name]) for name in SPLIT_NAMES}


def summarize_strata(
    labels: Mapping[int, str],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, int]]:
    """Per-stratum episode counts, for reporting.

    Returns ``{stratum: {"total": n, "train": n, "val": n, "test": n}}``.
    """
    where = {}
    for name in SPLIT_NAMES:
        for ep in splits.get(name, []):
            where[int(ep)] = name

    table: dict[str, dict[str, int]] = {}
    for ep, stratum in labels.items():
        key = str(stratum)
        row = table.setdefault(key, {"total": 0, "train": 0, "val": 0, "test": 0})
        row["total"] += 1
        name = where.get(int(ep))
        if name is not None:
            row[name] += 1
    return dict(sorted(table.items()))


def validate_split_info(split_info: Mapping, total_episodes: int) -> list[str]:
    """Check a ``split_info``-shaped dict against a dataset's episode count.

    Catches the failure mode that matters most for a pre-computed split file:
    the dataset grew (or shrank) since the file was written, so the episode
    indices inside it no longer mean what they meant.

    Returns:
        A list of human-readable problems — empty when the split is usable.
    """
    problems: list[str] = []
    lists = {name: list(split_info.get(f"{name}_episodes", []) or []) for name in SPLIT_NAMES}

    seen: dict[int, str] = {}
    for name, eps in lists.items():
        for ep in eps:
            if not isinstance(ep, int) or isinstance(ep, bool):
                problems.append(f"{name}: episode index {ep!r} is not an integer")
                continue
            if ep < 0 or ep >= total_episodes:
                problems.append(
                    f"{name}: episode {ep} is out of range for a dataset with "
                    f"{total_episodes} episodes (0..{total_episodes - 1})"
                )
            if ep in seen:
                problems.append(f"episode {ep} appears in both {seen[ep]} and {name}")
            else:
                seen[ep] = name

    if not seen:
        problems.append("split contains no episodes at all")
    elif not lists["train"]:
        problems.append("train split is empty")

    missing = total_episodes - len(seen)
    if missing > 0:
        log.info(
            "[stratified] %d of %d episode(s) are not referenced by the split file "
            "(fine if the split was built from a labelled subset)",
            missing, total_episodes,
        )

    return problems
