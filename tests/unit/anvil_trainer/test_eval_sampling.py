from __future__ import annotations

import pytest

from anvil_trainer.patches import evenly_spaced_sample_indices


def test_evenly_spaced_sample_indices_covers_full_range() -> None:
    indices = evenly_spaced_sample_indices(total=10_000, limit=1_600)

    assert len(indices) == 1_600
    assert len(set(indices)) == len(indices)
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 9_999


def test_evenly_spaced_sample_indices_keeps_small_dataset() -> None:
    assert evenly_spaced_sample_indices(total=3, limit=10) == [0, 1, 2]


@pytest.mark.parametrize(("total", "limit"), [(-1, 1), (1, 0), (1, -1)])
def test_evenly_spaced_sample_indices_rejects_invalid_inputs(
    total: int, limit: int
) -> None:
    with pytest.raises(ValueError):
        evenly_spaced_sample_indices(total=total, limit=limit)
