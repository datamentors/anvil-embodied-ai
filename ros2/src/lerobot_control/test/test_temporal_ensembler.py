"""Unit tests for the gradient-free temporal ensembler (policy-agnostic math)."""

import math

import numpy as np
import pytest

from lerobot_control.temporal_ensembler import TemporalEnsembler


def test_single_chunk_returns_rows_in_order():
    te = TemporalEnsembler(coeff=0.01)
    te.add_chunk(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    np.testing.assert_allclose(te.step(), [0.0, 1.0])
    np.testing.assert_allclose(te.step(), [2.0, 3.0])
    np.testing.assert_allclose(te.step(), [4.0, 5.0])
    assert te.step() is None  # chunk exhausted


def test_no_chunk_yields_none():
    assert TemporalEnsembler(coeff=0.01).step() is None


def test_coeff_zero_is_uniform_average():
    te = TemporalEnsembler(coeff=0.0)
    # two chunks that both cover step 0, constant values 0 and 10
    te.add_chunk(np.zeros((3, 1)))          # start=0
    te.add_chunk(np.full((3, 1), 10.0))     # start=0 (added same step)
    np.testing.assert_allclose(te.step(), [5.0])  # (0+10)/2 regardless of favor


def test_favor_older_weights_the_deeper_chunk():
    # older chunk value 0, newer chunk value 10, both cover the step.
    te = TemporalEnsembler(coeff=1.0, favor_older=True)
    te.add_chunk(np.zeros((5, 1)))          # OLDER (added first) -> value 0
    te.add_chunk(np.full((5, 1), 10.0))     # NEWER -> value 10
    out = te.step()[0]
    # older dominates => blended pulled toward 0
    assert out < 5.0
    # exact: ages newest=0 (w=e^-1), oldest=1 (w=e^0=1); pivot=1
    w_new, w_old = math.exp(-1.0), 1.0
    expected = (w_new * 10.0 + w_old * 0.0) / (w_new + w_old)
    assert out == pytest.approx(expected)


def test_favor_newer_weights_the_fresh_chunk():
    te = TemporalEnsembler(coeff=1.0, favor_older=False)
    te.add_chunk(np.zeros((5, 1)))          # older -> 0
    te.add_chunk(np.full((5, 1), 10.0))     # newer -> 10
    out = te.step()[0]
    assert out > 5.0  # newest dominates


def test_overlap_produces_value_between_chunks():
    # a rising newer chunk vs a flat older chunk; blend stays between them
    te = TemporalEnsembler(coeff=0.1)
    te.add_chunk(np.full((10, 1), 1.0))              # older, flat at 1
    te.add_chunk(np.linspace(2, 11, 10)[:, None])    # newer, rising 2..11
    v = te.step()[0]
    assert 1.0 <= v <= 2.0


def test_staggered_chunks_stay_continuous():
    # A new chunk arrives each step; the blended stream should have no big jumps.
    te = TemporalEnsembler(coeff=0.05, favor_older=True)
    outs = []
    base = 0.0
    for k in range(20):
        # each chunk is a short ramp starting near the current commanded value
        te.add_chunk((base + np.linspace(0, 5, 8))[:, None], start_step=k)
        outs.append(float(te.step()[0]))
        base = outs[-1]
    diffs = np.abs(np.diff(outs))
    assert diffs.max() < 5.0  # no chunk-boundary leap


def test_reset_clears_and_rewinds():
    te = TemporalEnsembler(coeff=0.01)
    te.add_chunk(np.ones((3, 2)))
    te.step()
    te.reset()
    assert te.step() is None
    assert te.in_flight == 0


def test_1d_chunk_promoted():
    te = TemporalEnsembler(coeff=0.01)
    te.add_chunk(np.array([1.0, 2.0, 3.0]))  # single action, 3 dims
    np.testing.assert_allclose(te.step(), [1.0, 2.0, 3.0])


def test_max_chunks_evicts_oldest():
    te = TemporalEnsembler(coeff=0.0, max_chunks=2)
    te.add_chunk(np.full((5, 1), 1.0), start_step=0)
    te.add_chunk(np.full((5, 1), 2.0), start_step=0)
    te.add_chunk(np.full((5, 1), 3.0), start_step=0)  # evicts the '1.0' chunk
    np.testing.assert_allclose(te.step(), [2.5])       # mean of 2 and 3


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_invalid_coeff_rejected(bad):
    with pytest.raises(ValueError):
        TemporalEnsembler(coeff=bad)


def test_output_is_fresh_copy():
    te = TemporalEnsembler(coeff=0.01)
    te.add_chunk(np.array([[7.0, 8.0]]))
    out = te.step()
    out[0] = 999.0
    te.reset()
    te.add_chunk(np.array([[7.0, 8.0]]))
    np.testing.assert_allclose(te.step(), [7.0, 8.0])  # not mutated by caller
