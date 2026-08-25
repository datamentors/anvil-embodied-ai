"""Tests for anvil_shared.stratified."""
from __future__ import annotations

import pytest
from anvil_shared.stratified import (
    compute_stratified_split_episodes,
    summarize_strata,
    validate_split_info,
)

SIZES = ("big", "medium", "small")
FACES = ("face_up", "face_down")


def make_labels(counts: dict[str, int]) -> dict[int, str]:
    """Build a {episode: stratum} mapping with the given per-stratum counts."""
    labels: dict[int, str] = {}
    ep = 0
    for stratum, n in counts.items():
        for _ in range(n):
            labels[ep] = stratum
            ep += 1
    return labels


def six_strata(n_each: int = 100) -> dict[int, str]:
    return make_labels({f"{s}|{f}": n_each for s in SIZES for f in FACES})


# =============================================================================
# compute_stratified_split_episodes
# =============================================================================


class TestStratifiedSplit:
    def test_ratio_is_honoured_within_every_stratum(self):
        labels = six_strata(100)
        splits = compute_stratified_split_episodes(labels, [8, 1, 1], seed=42)

        for row in summarize_strata(labels, splits).values():
            assert (row["train"], row["val"], row["test"]) == (80, 10, 10)

    def test_splits_are_disjoint(self):
        labels = six_strata(37)
        s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=1)
        train, val, test = set(s["train"]), set(s["val"]), set(s["test"])

        assert not train & val
        assert not train & test
        assert not val & test

    def test_every_episode_lands_somewhere(self):
        labels = six_strata(13)
        s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=1)
        assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(labels)

    def test_deterministic_for_same_seed(self):
        labels = six_strata(50)
        a = compute_stratified_split_episodes(labels, [8, 1, 1], seed=7)
        b = compute_stratified_split_episodes(labels, [8, 1, 1], seed=7)
        assert a == b

    def test_different_seed_gives_different_split(self):
        labels = six_strata(50)
        a = compute_stratified_split_episodes(labels, [8, 1, 1], seed=7)
        b = compute_stratified_split_episodes(labels, [8, 1, 1], seed=8)
        assert a != b

    def test_growing_one_stratum_leaves_the_others_untouched(self):
        """Per-stratum seeding: new small envelopes must not reshuffle the big ones."""
        base = make_labels({"big|face_up": 100, "small|face_down": 40})
        grown = dict(base)
        for i in range(20):
            grown[1000 + i] = "small|face_down"

        a = compute_stratified_split_episodes(base, [8, 1, 1], seed=42)
        b = compute_stratified_split_episodes(grown, [8, 1, 1], seed=42)

        def big_only(eps):
            return {e for e in eps if base.get(e) == "big|face_up"}

        for split in ("train", "val", "test"):
            assert big_only(a[split]) == big_only(b[split])

    def test_two_element_ratio_means_no_test_set(self):
        labels = six_strata(10)
        s = compute_stratified_split_episodes(labels, [8, 2], seed=3)
        assert s["test"] == []
        assert len(s["val"]) == 12  # 2 per stratum, 6 strata

    def test_tiny_stratum_goes_entirely_to_train_and_warns(self, caplog):
        labels = make_labels({"big|face_up": 100, "small|face_down": 3})
        with caplog.at_level("WARNING"):
            s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=42)

        tiny = {ep for ep, k in labels.items() if k == "small|face_down"}
        assert tiny <= set(s["train"])
        assert "small|face_down" in caplog.text

    def test_ratio_1_0_0_puts_everything_in_train(self):
        labels = six_strata(9)
        s = compute_stratified_split_episodes(labels, [1, 0, 0], seed=5)
        assert set(s["train"]) == set(labels)
        assert s["val"] == [] and s["test"] == []

    def test_output_lists_are_sorted(self):
        labels = six_strata(20)
        s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=2)
        for eps in s.values():
            assert eps == sorted(eps)

    @pytest.mark.parametrize(
        "labels, ratio",
        [
            ({}, [8, 1, 1]),                       # no labels
            ({0: "a"}, [8, 1, 1, 1]),              # 4-element ratio
            ({0: "a"}, [0, 0, 0]),                 # sums to zero
            ({0: "a"}, [-1, 1, 1]),                # negative
        ],
    )
    def test_rejects_bad_input(self, labels, ratio):
        with pytest.raises(ValueError):
            compute_stratified_split_episodes(labels, ratio)


# =============================================================================
# summarize_strata
# =============================================================================


class TestSummarizeStrata:
    def test_counts_add_up_per_stratum(self):
        labels = six_strata(30)
        s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=4)
        table = summarize_strata(labels, s)

        assert len(table) == 6
        for row in table.values():
            assert row["train"] + row["val"] + row["test"] == row["total"] == 30

    def test_sorted_by_stratum_name(self):
        labels = six_strata(10)
        s = compute_stratified_split_episodes(labels, [8, 1, 1], seed=4)
        keys = list(summarize_strata(labels, s))
        assert keys == sorted(keys)


# =============================================================================
# validate_split_info
# =============================================================================


class TestValidateSplitInfo:
    def _split(self, train, val, test):
        return {"train_episodes": train, "val_episodes": val, "test_episodes": test}

    def test_accepts_a_consistent_split(self):
        assert validate_split_info(self._split([0, 1, 2], [3], [4]), 5) == []

    def test_rejects_episode_beyond_dataset(self):
        problems = validate_split_info(self._split([0, 1], [2], [99]), 5)
        assert any("out of range" in p for p in problems)

    def test_rejects_negative_episode(self):
        problems = validate_split_info(self._split([-1, 0], [1], [2]), 5)
        assert any("out of range" in p for p in problems)

    def test_rejects_overlap_between_splits(self):
        problems = validate_split_info(self._split([0, 1], [1], [2]), 5)
        assert any("appears in both" in p for p in problems)

    def test_rejects_empty_train(self):
        problems = validate_split_info(self._split([], [0], [1]), 5)
        assert any("train split is empty" in p for p in problems)

    def test_rejects_completely_empty_split(self):
        problems = validate_split_info(self._split([], [], []), 5)
        assert any("no episodes at all" in p for p in problems)

    def test_rejects_non_integer_episode(self):
        problems = validate_split_info(self._split([0, "1"], [2], [3]), 5)
        assert any("not an integer" in p for p in problems)

    def test_partial_coverage_is_allowed(self):
        """A split built from a labelled subset is valid — just doesn't cover everything."""
        assert validate_split_info(self._split([0, 1], [2], [3]), 100) == []

    def test_missing_keys_are_treated_as_empty(self):
        problems = validate_split_info({"train_episodes": [0, 1]}, 5)
        assert problems == []
