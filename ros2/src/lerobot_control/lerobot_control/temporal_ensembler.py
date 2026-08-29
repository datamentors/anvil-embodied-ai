"""Gradient-free temporal ensembling over an action-chunk stream.

An alternative inference *smoother* to RTC. Where RTC guides the new chunk's
prefix to match the executing tail (and needs backward passes), temporal
ensembling is pure post-processing: keep the recent chunks *in flight* and, for
each control step, return the age-weighted average of every chunk that predicts
that step. Because it never touches the policy and needs no gradients, it smooths
*any* chunk stream — including the sglang flow path where RTC guidance is
unavailable.

    wᵢ = exp(-coeff · i)   (normalised to sum to 1)

`i` is the chunk's age (recency rank among the chunks currently covering this
step). With `favor_older=True` (the ROBOTIS OMY choice) the *oldest* contributing
chunk — the one already deepest into the reach — gets the largest weight, which
keeps a grasp from restarting shallow at each re-plan. With `favor_older=False`
the newest chunk dominates (incorporate observations faster).

This class is deliberately policy-agnostic and framework-free (numpy only) so it
can be unit-tested in isolation and reused across backends.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class TemporalEnsembler:
    def __init__(
        self,
        coeff: float,
        *,
        max_chunks: int = 8,
        favor_older: bool = True,
    ) -> None:
        if not math.isfinite(coeff) or coeff < 0:
            raise ValueError(f"temporal ensemble coeff must be finite and >= 0, got {coeff!r}")
        if max_chunks < 1:
            raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
        self.coeff = float(coeff)
        self.max_chunks = int(max_chunks)
        self.favor_older = bool(favor_older)
        # (start_step, chunk[horizon, action_dim]); newest is rightmost.
        self._chunks: deque[tuple[int, np.ndarray]] = deque(maxlen=self.max_chunks)
        self._step = 0

    def reset(self) -> None:
        """Drop all in-flight chunks and rewind the clock (new episode / re-arm)."""
        self._chunks.clear()
        self._step = 0

    def add_chunk(self, chunk: np.ndarray, start_step: int | None = None) -> None:
        """Register a freshly produced chunk.

        `chunk` is [horizon, action_dim]. `start_step` is the control step its
        first row applies to; default = the next step to be published (`now`),
        which is correct when a chunk is produced for the current observation.
        """
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"chunk must be [horizon, action_dim], got shape {arr.shape}")
        start = self._step if start_step is None else int(start_step)
        self._chunks.append((start, arr))

    def _weights(self, ages: list[int]) -> np.ndarray:
        if self.favor_older:
            # oldest contributor (max age) → weight exp(0) = 1
            pivot = max(ages)
            raw = np.array([math.exp(-self.coeff * (pivot - a)) for a in ages], dtype=np.float64)
        else:
            # newest contributor (age 0) → weight exp(0) = 1
            raw = np.array([math.exp(-self.coeff * a) for a in ages], dtype=np.float64)
        s = raw.sum()
        return raw / s if s > 0 else np.full(len(ages), 1.0 / len(ages))

    def step(self) -> np.ndarray | None:
        """Blended action for the current control step, or None if no chunk covers it.

        Advances the internal clock by one step (call exactly once per control
        tick). Returns a fresh 1-D float array of length action_dim.
        """
        t = self._step
        self._step += 1
        ages: list[int] = []
        rows: list[np.ndarray] = []
        # age 0 = newest → iterate newest-first
        for age, (start, chunk) in enumerate(reversed(self._chunks)):
            idx = t - start
            if 0 <= idx < chunk.shape[0]:
                ages.append(age)
                rows.append(chunk[idx])
        if not rows:
            return None
        w = self._weights(ages)
        return np.tensordot(w, np.stack(rows, axis=0), axes=(0, 0))

    @property
    def in_flight(self) -> int:
        """Chunks still (potentially) covering the current or a future step."""
        t = self._step
        return sum(1 for start, chunk in self._chunks if t - start < chunk.shape[0])
