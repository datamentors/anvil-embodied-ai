"""Dataset episode-split helpers shared between trainer and eval.

Both ``anvil_trainer`` (when building train/val/test dataloaders) and
``anvil_eval`` (when resolving which episodes to evaluate) need identical
split computation.  Keeping it here avoids two copies that can drift.

Public API:
    compute_split_episodes(total_episodes, ratio, seed) -> dict[str, list[int]]
    load_split_info(path) -> dict | None
    save_split_info(path, split_info) -> None
    validate_split_info(split_info, total_episodes) -> dict
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)


def compute_split_episodes(
    total_episodes: int,
    ratio: Sequence[float],
    seed: int = 42,
) -> dict[str, list[int]]:
    """Deterministic random 3-way episode split.

    Given N total episodes and a ``[train, val, test]`` ratio, return a dict
    ``{"train": [...], "val": [...], "test": [...]}`` with disjoint episode
    lists sampled without replacement.  Ratios of 0 produce empty lists (no
    forced-minimum trick — the caller decides whether to error).

    The shuffle is seeded so calling this twice with the same ``seed`` yields
    the same split (important for training resume correctness).

    Args:
        total_episodes: Total number of episodes in the dataset.
        ratio: 2- or 3-element sequence of non-negative floats.  A 2-element
            sequence is treated as ``[train, val, 0]`` (no test set).
        seed: RNG seed for the shuffle.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` — empty splits are
        included for consistency but may be empty lists.

    Raises:
        ValueError: when ``total_episodes < 1`` or all ratios sum to zero.
    """
    if total_episodes < 1:
        raise ValueError(f"total_episodes must be >= 1, got {total_episodes}")

    r = list(ratio)
    if len(r) == 2:
        r.append(0.0)
    elif len(r) != 3:
        raise ValueError(f"ratio must have 2 or 3 elements, got {len(r)}")

    total_ratio = sum(r)
    if total_ratio <= 0:
        raise ValueError(f"ratio must sum to > 0, got {r}")

    # Allocate from smallest to largest: round-with-respect-for-zero
    n_test = round(total_episodes * r[2] / total_ratio) if r[2] > 0 else 0
    n_val = round(total_episodes * r[1] / total_ratio) if r[1] > 0 else 0
    n_train = total_episodes - n_val - n_test
    if n_train < 0:
        # Pathological ratio caused overflow; clamp and warn
        log.warning(
            "[splits] ratio %s produced n_train=%d for %d episodes; clamping to 0",
            r,
            n_train,
            total_episodes,
        )
        n_train = 0

    all_eps = list(range(total_episodes))
    rng = random.Random(seed)
    rng.shuffle(all_eps)

    train_eps = sorted(all_eps[:n_train])
    val_eps = sorted(all_eps[n_train : n_train + n_val])
    test_eps = sorted(all_eps[n_train + n_val :])
    return {"train": train_eps, "val": val_eps, "test": test_eps}


def load_split_info(path: Path) -> dict | None:
    """Read a ``split_info.json`` file if it exists.

    Returns:
        Parsed JSON dict, or ``None`` if the file is missing or malformed
        (with a warning log in the latter case).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("[splits] Failed to read %s: %s", path, e)
        return None


def save_split_info(path: Path, split_info: dict) -> None:
    """Write ``split_info`` as JSON to ``path``.

    The caller owns the dict schema (typical keys: ``split_ratio``,
    ``total_episodes``, ``train_episodes``, ``val_episodes``,
    ``test_episodes``).  Parent directories are not created.

    Args:
        path: Destination file.  Parent directory must already exist.
        split_info: JSON-serialisable dict.
    """
    path = Path(path)
    path.write_text(json.dumps(split_info, indent=2))


def validate_split_info(split_info: dict, total_episodes: int) -> dict:
    """Validate and normalize an explicit train/validation/test manifest.

    An explicit manifest may curate out episodes. The three split lists must
    therefore be disjoint and in range, but do not need to cover every source
    episode. Extra provenance fields are preserved.
    """
    if not isinstance(split_info, dict):
        raise ValueError("split manifest must contain a JSON object")
    if total_episodes < 1:
        raise ValueError(f"total_episodes must be >= 1, got {total_episodes}")

    declared_total = split_info.get("total_episodes")
    if declared_total is not None and declared_total != total_episodes:
        raise ValueError(
            "split manifest total_episodes does not match the dataset: "
            f"manifest={declared_total}, dataset={total_episodes}"
        )

    normalized = dict(split_info)
    seen: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_episodes"
        raw = split_info.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"{key} must be a JSON list")

        values: list[int] = []
        local_seen: set[int] = set()
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} contains a non-integer episode: {value!r}")
            if value < 0 or value >= total_episodes:
                raise ValueError(
                    f"{key} contains out-of-range episode {value}; "
                    f"valid range is 0..{total_episodes - 1}"
                )
            if value in local_seen:
                raise ValueError(f"{key} contains duplicate episode {value}")
            if value in seen:
                raise ValueError(f"episode {value} appears in both {seen[value]} and {split_name}")
            local_seen.add(value)
            seen[value] = split_name
            values.append(value)
        normalized[key] = sorted(values)

    if not normalized["train_episodes"]:
        raise ValueError("train_episodes must contain at least one episode")

    selected = split_info.get("selected_episodes")
    if selected is not None:
        if not isinstance(selected, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in selected
        ):
            raise ValueError("selected_episodes must be a JSON list of integers")
        if len(selected) != len(set(selected)):
            raise ValueError("selected_episodes contains duplicates")
        if set(selected) != set(seen):
            raise ValueError("selected_episodes must equal the union of train/val/test episodes")
        normalized["selected_episodes"] = sorted(selected)

    normalized["total_episodes"] = total_episodes
    return normalized
