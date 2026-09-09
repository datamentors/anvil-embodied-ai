[← Back to README](../README.md)

# Implementation guide — gripper fix and training-parameter decisions

**Written 2026-09-09.** Companion to
[REVIEW_TRAINING_AND_GRIPPER_2026-09-08.md](REVIEW_TRAINING_AND_GRIPPER_2026-09-08.md),
which explains *why*. This document is only *what to change, where, and in what order*.

Every as-is block was copied from the file named above it on 2026-09-09. If a file on your
machine does not match its as-is block, stop and reconcile before editing — the numbers and
anchors below were checked against the running system, not from memory.

**Conventions.** Each change has a fixed shape: what it fixes, the file, the anchor
(function or key, plus the line number it was at), an **AS-IS** block, a **TO-BE** block,
how to verify, and how to roll back. Line numbers drift; the anchors do not — search for the
anchor, not the line.

**Where code lives, and what a change costs:**

| Component | Path | Cost of a change |
|---|---|---|
| GR00T node | `inference_Gr00t/ros2/src/lerobot_groot_control/lerobot_groot_control/` | **Baked into the image** — needs `docker compose build` |
| GR00T profiles | `inference_Gr00t/configs/*.yaml` | Bind-mounted — needs `down` + `up --force-recreate` |
| Pi0.5 node | `ros2/src/lerobot_control/lerobot_control/` | Baked into the image |
| Pi0.5 profiles | `configs/lerobot_control/*.yaml` | Bind-mounted |
| Converter | `packages/mcap_converter/`, `packages/mcap_convert_gpu/` | Reconvert the dataset |
| Trainer | `packages/anvil_trainer/` | Retrain |
| Evaluator | `packages/anvil_eval/` | Re-run eval only |

A bind-mounted profile edited in place gets a **new inode**, and the running container keeps
the old one. Always `--force-recreate`, never a bare `restart`.

---

## 0. Status board — read this before touching anything

Measured on the workstation on 2026-09-09 at 09:45 UTC.

| Thing | State |
|---|---|
| GR00T live container | `anvil-groot-inference`, image `groot17-011000`, **exited 3 h ago** |
| GR00T live profile | `inference_groot17_envelope2916_ensemble_live.yaml` — guidance **OFF**, ensemble coeff 0.1 / 8 chunks |
| GR00T checkpoint in use | `groot17_envelope2916-ckpt-011000` = step 11 000 of 88 000 (**0.5 epoch**) |
| Pi0.5 live | `envelope-curated-v3-full-vlm-ckpt-019922`, last monitor run 2026-09-05 |
| LingBot run #1 (`fullbackbone`) | **dead** — interrupted at step 11 200 / 2 109 384, no checkpoint, no eval |
| LingBot run #2 (`robotwin`) | **dead** — crashed 04:23 UTC at step 35 000 / 100 000, `No space left on device` |
| Workstation disk | **1.8 T used of 1.9 T, 27 G free (99 %)** |
| `/mnt/Checkpoints` (NAS) | **not mounted** — the keep-best guardians have nowhere to offload |

**Two things changed since the 09-08 review and they matter.**

1. Someone patched `_get_t5_prompt_embeds` in the LingBot venv to memoise the prompt
   embedding. Step time went **10.5 s → 0.538 s, about 20×**. The review's claim that
   LingBot is slow "because each sample encodes and decodes video latents" is **wrong**:
   the patch's own note measures **11.3 s of an 11.4 s step inside UMT5-XXL on CPU**, with
   the GPU ~90 % idle. A 256-day run became a ~15-hour run.
2. That patch lives **only in `site-packages` on one machine**
   (`~/venvs/lingbot-va/lib/python3.12/site-packages/lerobot/policies/lingbot_va/modeling_lingbot_va.py`,
   original kept as `.anvil-orig`). Any `pip install` erases it, and it does not exist on
   amax-ws-03. **Change 9 vendors it into the repo.** This is the single highest-value/lowest-risk
   item in this document.

---

## Part 1 — Unblock first (today, no code)

### Change 0.1 — Recover the LingBot run

The crash is pure disk. Checkpoint `030000` is **complete and resumable**
(`pretrained_model/model.safetensors` 10.2 GB + `training_state/optimizer_state.safetensors`
20.3 GB); step 35 000 was never written, so there is no partial directory to clean.

```bash
# 1. Mount the NAS. nfsvers=3 is mandatory — the NAS's NFSv4.1 is broken, and the
#    /etc/fstab entry omits it, which is why it is never mounted after a reboot.
sudo mkdir -p /mnt/Checkpoints
sudo mount -t nfs -o nfsvers=3 192.168.1.3:/Checkpoints /mnt/Checkpoints
df -h /mnt/Checkpoints          # must show the NAS, not the root filesystem

# 2. Move the older checkpoint off the local disk (frees 29 GB).
mkdir -p /mnt/Checkpoints/lingbot_va_envelope2916_robotwin
R=~/lingbot_va_work/runs/lingbot_va_envelope2916_robotwin_bs4_20260908T225317Z
rsync -a --remove-source-files "$R/output/checkpoints/025000" \
      /mnt/Checkpoints/lingbot_va_envelope2916_robotwin/

# 3. Resume from 030000.
cd ~/lingbot_va_work && tmux new-session -d -s lingbot2916rw \
  "RESUME=1 bash ~/lingbot_va_work/launch_2916_lingbot_robotwin.sh"
```

`launch_2916_lingbot_robotwin.sh` currently has no `RESUME` path; add one (`--resume=true` with
`--output_dir` pointing at the existing run, which lerobot requires to already exist — the
opposite of a fresh run's `FileExistsError` rule).

**Also fix the guardian.** It runs as
`keep_topk_nas.py … --keep 4 --min-free-gb 45`. Four LingBot checkpoints are **116 GB**;
the disk never had that. Two independent fixes, apply both:

* `--keep 2` while the NAS is the archive.
* **Strip the optimizer state from every retained checkpoint except the newest.**
  `training_state/optimizer_state.safetensors` is 20.3 GB of the 29 GB, and it is only needed
  to *resume*, never to *evaluate or deploy*. Keeping weights-only for older checkpoints cuts
  retention cost by 70 %.

**Verify:** `df -h /` shows > 100 G free, and the log prints a new `Checkpoint policy after
step 40000` line without a traceback.

### Change 0.2 — Stop losing evidence on restart

`inference_Gr00t/…/monitor_output/inference_data.csv` is opened with mode `"w"`, so **every
container restart truncates the previous run's data**. Both A/B runs on 09-08 survived only
because someone copied them out by hand.

Add to `inference_Gr00t/scripts/` and call it before every `up`:

```bash
#!/usr/bin/env bash
# archive_monitor.sh — preserve the last run before it is truncated.
set -euo pipefail
SRC=./monitor_output/inference_data.csv
[ -s "$SRC" ] || exit 0
DEST="./monitor_output/archive/$(date -u +%Y%m%dT%H%M%SZ)_$(basename "${CONFIG_FILE:-unknown}" .yaml).csv.gz"
mkdir -p "$(dirname "$DEST")" && gzip -c "$SRC" > "$DEST"
echo "archived -> $DEST"
```

---

## Part 2 — Inference: make the gripper decisive

Three changes on the GR00T node, in this order. Changes 1 and 2 are code (image rebuild);
change 3 is config only.

### Change 1 — Exempt the gripper channels from the ensembler

**Fixes:** averaging a near-binary channel produces half-closed commands. Measured: the
published gripper sat in the 0.005–0.045 m band **45–53 % of ticks**.

**File:** `inference_Gr00t/ros2/src/lerobot_groot_control/lerobot_groot_control/temporal_ensembler.py`
**Anchor:** `ChunkEnsembler.__init__` and `ChunkEnsembler.value_at`.

**AS-IS**

```python
    def __init__(
        self,
        coeff: float = 0.01,
        max_chunks: int = 8,
        action_dim: int | None = None,
    ) -> None:
        if not np.isfinite(coeff):
            raise ValueError("temporal ensemble coeff must be finite")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.coeff = float(coeff)
        self.max_chunks = int(max_chunks)
        self.action_dim = action_dim
```

**TO-BE**

```python
    def __init__(
        self,
        coeff: float = 0.01,
        max_chunks: int = 8,
        action_dim: int | None = None,
        passthrough_dims: "tuple[int, ...] | None" = None,
    ) -> None:
        if not np.isfinite(coeff):
            raise ValueError("temporal ensemble coeff must be finite")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.coeff = float(coeff)
        self.max_chunks = int(max_chunks)
        self.action_dim = action_dim
        # Channels that must NOT be averaged. Averaging is a low-pass filter, and a
        # low-pass filter applied to a near-binary channel returns values that lie
        # between its two modes — for a gripper that is "half closed", which is not a
        # state any demonstration contains. Measured on ckpt 011000: the published
        # gripper sat in the 0.005-0.045 m band 45-53 % of ticks.
        if passthrough_dims is None:
            self.passthrough_dims: tuple[int, ...] = ()
        else:
            dims = tuple(int(d) for d in passthrough_dims)
            if action_dim is not None and any(d < 0 or d >= action_dim for d in dims):
                raise ValueError(
                    f"passthrough_dims {dims} out of range for action_dim {action_dim}"
                )
            self.passthrough_dims = dims
```

**AS-IS** (end of `value_at`)

```python
        blended = (weights[:, None] * stacked).sum(axis=0) / total
        if blended.shape != base.shape or not np.all(np.isfinite(blended)):
            return base
        return blended
```

**TO-BE**

```python
        blended = (weights[:, None] * stacked).sum(axis=0) / total
        if blended.shape != base.shape or not np.all(np.isfinite(blended)):
            return base
        # Restore the un-averaged value on passthrough channels. `base` is what the RTC
        # queue popped, i.e. the newest merged chunk's row for this tick, so those
        # channels behave exactly as they do with the smoother disabled.
        for dim in self.passthrough_dims:
            if dim < blended.shape[0]:
                blended[dim] = base[dim]
        return blended
```

**Wiring — file:** `…/lerobot_groot_control/inference_node.py`, anchor `_setup_action_smoother`
(was line 1132).

**AS-IS**

```python
        self._ensembler = ChunkEnsembler(
            coeff=float(config.get("coeff", 0.01)),
            max_chunks=int(config.get("horizon_chunks", 8)),
            action_dim=expected_dim or None,
        )
        self.get_logger().info(
            f"Action smoother: temporal_ensemble (coeff={self._ensembler.coeff}, "
            f"horizon_chunks={self._ensembler.max_chunks}, action_dim={expected_dim}) "
            "— averaging overlapping chunks"
        )
```

**TO-BE**

```python
        # Gripper channels, derived from config rather than hardcoded: each arm's block
        # is [action_start:action_end] and within it the model order is
        # model_joint_order, whose first entry is finger_joint1 on this robot.
        model_order = list(self.joint_names_config.get("model_joint_order", []))
        gripper_dims: list[int] = []
        if config.get("passthrough_gripper", True) and model_order:
            for joint_index, joint_name in enumerate(model_order):
                if "finger" not in joint_name and "gripper" not in joint_name:
                    continue
                for arm in self.arms_config.values():
                    gripper_dims.append(int(arm.get("action_start", 0)) + joint_index)
        self._ensembler = ChunkEnsembler(
            coeff=float(config.get("coeff", 0.01)),
            max_chunks=int(config.get("horizon_chunks", 8)),
            action_dim=expected_dim or None,
            passthrough_dims=tuple(sorted(set(gripper_dims))),
        )
        self.get_logger().info(
            f"Action smoother: temporal_ensemble (coeff={self._ensembler.coeff}, "
            f"horizon_chunks={self._ensembler.max_chunks}, action_dim={expected_dim}, "
            f"passthrough_dims={self._ensembler.passthrough_dims}) "
            "— averaging overlapping chunks except the passthrough channels"
        )
```

**Verify:** startup logs `passthrough_dims=(0, 8)`. On the 16-DOF bimanual layout those are
`left_finger_joint1` and `right_finger_joint1`. If it logs `()`, `model_joint_order` was not
read — do not run.

**Roll back:** `smoother.passthrough_gripper: false` in the profile. No rebuild.

### Change 2 — Latch the gripper (hysteresis + dwell + squeeze)

**Fixes:** the remaining flicker (close/open every 3–5 ticks) and the missing squeeze. This is
item 4 on the CHANGELOG's own "where to go next" list.

**New file:** `inference_Gr00t/ros2/src/lerobot_groot_control/lerobot_groot_control/gripper_latch.py`

```python
"""Hysteresis latch for a near-binary gripper channel.

Why this exists
---------------
Two independent measurements say the gripper command is the failure, not the arm:

  * With RTC guidance OFF, consecutive GR00T forwards disagree on gripper direction 65 %
    of the time, and the published command flips open/closed every 3-5 control ticks
    (100-170 ms) - far faster than a 0.72 kg direct-drive finger can act on.
  * The training label is the follower's ACHIEVED finger position, so the model asks for
    the contact position (~4 mm) with no squeeze margin. A position-controlled gripper
    commanded to where it already is exerts no force.

A Schmitt trigger fixes the first: two thresholds plus a minimum dwell turn a noisy
bimodal signal into a decisive state. `closed_target` fixes the second: while latched
closed we publish a target past contact, which is what a teleoperator does.

Safety
------
The output is always one of {closed_target, open_target, the model's own value}, and every
one of those is clamped afterwards by `_saturate_mechanical_stops` and
`_validate_absolute_joint_targets`. The latch cannot produce a value outside the configured
joint limits. It CAN command a squeeze the model did not ask for - that is the point - so
`closed_target` must be set deliberately and tested in echo mode first.
"""

from __future__ import annotations


class GripperLatch:
    """Per-channel Schmitt trigger with a minimum dwell time.

    States are 'open' and 'closed'. A transition requires the input to cross the far
    threshold AND the current state to have been held for at least `min_dwell_ticks`.
    """

    def __init__(
        self,
        close_below: float,
        open_above: float,
        closed_target: float,
        open_target: float,
        min_dwell_ticks: int,
    ) -> None:
        if not close_below < open_above:
            raise ValueError(
                f"close_below ({close_below}) must be < open_above ({open_above}); "
                "equal values are a comparator, not a hysteresis band"
            )
        if min_dwell_ticks < 0:
            raise ValueError("min_dwell_ticks must be >= 0")
        self.close_below = float(close_below)
        self.open_above = float(open_above)
        self.closed_target = float(closed_target)
        self.open_target = float(open_target)
        self.min_dwell_ticks = int(min_dwell_ticks)
        self._state: str | None = None
        self._last_transition_tick = -(10**9)
        self.transitions = 0

    def reset(self) -> None:
        self._state = None
        self._last_transition_tick = -(10**9)

    def value(self, raw: float, tick: int) -> float:
        """Latched target for this tick, given the model's own value `raw`."""
        if self._state is None:
            # Adopt the model's opinion once, so the first published command agrees
            # with the arm's actual pose instead of jumping.
            self._state = "closed" if raw < self.close_below else "open"
            self._last_transition_tick = tick
        held = tick - self._last_transition_tick
        if held >= self.min_dwell_ticks:
            if self._state == "open" and raw < self.close_below:
                self._state = "closed"
                self._last_transition_tick = tick
                self.transitions += 1
            elif self._state == "closed" and raw > self.open_above:
                self._state = "open"
                self._last_transition_tick = tick
                self.transitions += 1
        return self.closed_target if self._state == "closed" else self.open_target

    @property
    def state(self) -> str | None:
        return self._state
```

**Wiring — file:** `…/inference_node.py`, anchor `_smooth_action` (was line 2836).

**AS-IS**

```python
        if self._ensembler is None:
            return action
        try:
            base = np.asarray(action, dtype=np.float64).reshape(-1)
            return self._ensembler.value_at(tick, base)
        except Exception as exc:
```

**TO-BE**

```python
        if self._ensembler is None and not self._gripper_latches:
            return action
        try:
            base = np.asarray(action, dtype=np.float64).reshape(-1)
            smoothed = (
                self._ensembler.value_at(tick, base)
                if self._ensembler is not None
                else base
            )
            # The latch reads the model's OWN value for this channel (`base`), never the
            # averaged one: averaging is what destroys the bimodality the latch needs.
            for dim, latch in self._gripper_latches.items():
                if dim < smoothed.shape[0]:
                    smoothed[dim] = latch.value(float(base[dim]), tick)
            return smoothed
        except Exception as exc:
```

Build the latches at the end of `_setup_action_smoother`, after `gripper_dims` is known:

```python
        latch_cfg = config.get("gripper_latch", {}) or {}
        self._gripper_latches = {}
        if latch_cfg.get("enable", False):
            from lerobot_groot_control.gripper_latch import GripperLatch

            dwell_sec = float(latch_cfg.get("min_dwell_sec", 0.5))
            for dim in sorted(set(gripper_dims)):
                self._gripper_latches[dim] = GripperLatch(
                    close_below=float(latch_cfg.get("close_below", 0.020)),
                    open_above=float(latch_cfg.get("open_above", 0.035)),
                    closed_target=float(latch_cfg.get("closed_target", 0.0)),
                    open_target=float(latch_cfg.get("open_target", 0.050)),
                    min_dwell_ticks=max(1, round(dwell_sec * float(self.control_freq))),
                )
            self.get_logger().warn(
                f"Gripper latch ENABLED on dims {sorted(self._gripper_latches)}: "
                f"close<{latch_cfg.get('close_below', 0.020)} -> "
                f"{latch_cfg.get('closed_target', 0.0)}, "
                f"open>{latch_cfg.get('open_above', 0.035)} -> "
                f"{latch_cfg.get('open_target', 0.050)}, "
                f"dwell {dwell_sec}s. The published gripper target is NO LONGER the "
                "model's value; it is a squeeze command."
            )
```

Also add `self._gripper_latches = {}` next to `self._ensembler = None` at the top of
`_setup_action_smoother`, and call `latch.reset()` wherever `self._ensembler.reset()` is
called (`_invalidate_action_state_locked`), so a watchdog latch clears the gripper state too.

**Profile keys** (`inference_Gr00t/configs/inference_groot17_envelope2916_*.yaml`, under
`inference_tuning.smoother`):

```yaml
  smoother:
    type: temporal_ensemble
    coeff: 0.01              # see Change 4 — 0.1 was over-damped
    horizon_chunks: 8
    passthrough_gripper: true
    gripper_latch:
      enable: true
      close_below: 0.020     # m; model value below this asks for a close
      open_above: 0.035      # m; above this asks for an open
      closed_target: 0.0     # m; PAST contact — this is the squeeze the label lacks
      open_target: 0.050     # m
      min_dwell_sec: 0.5     # no transition may happen faster than this
```

**Set `closed_target` deliberately.** `0.0` is the URDF lower bound and gives maximum squeeze
on a gripper rated to 350 N. For a paper envelope start at `0.0` **only after** an echo-mode
run; if the team wants to be conservative, `0.002` still gives 2 mm of over-travel past the
~4 mm contact seen in the data.

**Verify, in this order:**
1. `ECHO_TOPIC_ONLY=true` — confirm the startup warning lists dims `[0, 8]` and no arm moves.
2. Shadow run: `close/open transitions` should fall from 220–338 per 10 min into single
   digits, and the mid-band occupancy from 45–53 % to ~0 %.
3. Only then live, with a hand on the stop.

**Roll back:** `gripper_latch.enable: false`. No rebuild.

### Change 3 — Widen the finger joint limits to the real mechanism

**Fixes:** with `ENFORCE_JOINT_POSITION_LIMITS=true` the finger is clamped at `0.0`, so any
squeeze target is silently truncated — Change 2 would be a no-op on the closing side.

**File:** `inference_Gr00t/configs/inference_groot17_envelope2916_*.yaml`, anchor
`safety.joint_position_limits` (was line 463).

The dataset's own recorded range is **−0.0048 … +0.0567 m** on both fingers
(`meta/stats.json`, `action.min` / `action.max`), i.e. the mechanism genuinely travels
outside the URDF's `[0.0, 0.05]`.

**AS-IS**

```yaml
    follower_l_finger_joint1: [0.0, 0.05]
    …
    follower_r_finger_joint1: [0.0, 0.05]
```

**TO-BE**

```yaml
    # Fingers only: the URDF's [0.0, 0.05] is the nominal stroke, but the recorded
    # mechanism reaches -0.0048..+0.0567 m on this workcell (envelope-2916
    # meta/stats.json, action.min/max). Clamping at 0.0 truncates exactly the squeeze
    # the gripper latch exists to command. The 14 arm joints keep their URDF values.
    follower_l_finger_joint1: [-0.005, 0.057]
    …
    follower_r_finger_joint1: [-0.005, 0.057]
```

**Preferred alternative:** re-zero the finger encoder so that closed = 0.0 and the URDF limit
is true again. That is the correct fix; widening the limit is the fast one. Do not do both.

**Verify:** the `[SATURATE] follower_*_finger_joint1` warnings disappear from the log
(7 699 of them in the 19:09 run were the open-side clamp at 0.0502 → 0.0500).

### Change 4 — The A/B that decides the smoother settings

The 09-08 experiment is not repeatable as run: it changed guidance ON *and* kept the
ensemble at coeff 0.1 / 8 chunks, and the result ("gripper never closes") cannot be
attributed to either. Run four arms, changing one thing each, on the same 20 envelopes,
10 large / 10 small, half face-down, same start pose.

| Arm | `use_rtc_guidance` | `smoother.type` | `coeff` | `passthrough_gripper` | `gripper_latch` |
|---|---|---|---|---|---|
| **A** control | true | none | — | — | false |
| **B** | true | temporal_ensemble | 0.01 | true | false |
| **C** | true | temporal_ensemble | 0.01 | true | **true** |
| **D** | false | temporal_ensemble | 0.01 | true | true |

**Arm A has never been run for GR00T.** It is guidance-on with no post-hoc damping — the
configuration the Pi0.5 stack has always used and the one that closes decisively there. Run
it first; if A already grasps, changes 1 and 2 become optimisations rather than fixes.

**Score on the task, not on smoothness.** Per arm: grasp attempts, successful lifts,
successful sorts, and the four gripper numbers below. A smoother trace that never grasps is
the failure mode we already measured.

```python
#!/usr/bin/env python3
"""gripper_report.py — the four numbers that decide the A/B. Verified on both 09-08 CSVs."""
import csv, sys
import numpy as np

fn = sys.argv[1]
with open(fn) as f:
    lines = [l for l in f if not l.startswith("#")]      # the monitor writes a '#' header
rows = list(csv.reader(lines))
hdr, data = rows[0], rows[1:]
fl = lambda x: float(x) if x not in ("", "nan") else np.nan
A = np.array([[fl(x) for x in r] for r in data if len(r) == len(hdr)])

# obs_state_* is CONTROLLER order (finger last: 7, 15).
# raw_output_* / control_cmd_* are MODEL order (finger first: 0, 8).
for label, col in (("left", "control_cmd_7"), ("right", "control_cmd_15")):
    v = A[:, hdr.index(col)]
    v = v[~np.isnan(v)]
    closed = (v < 0.025).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, closed, 0]))
    runs = (edges[1::2] - edges[0::2]) if len(edges) > 1 else np.array([])
    print(
        f"{label:5s} transitions={int(np.abs(np.diff(closed)).sum()):4d}  "
        f"closed_runs={len(runs):3d}  median_ticks={np.median(runs) if len(runs) else 0:5.0f}  "
        f"mid_band={np.mean((v > 0.005) & (v < 0.045)):.3f}  min={v.min():+.4f}"
    )
```

Targets: `transitions` in single digits per 10 min, `median_ticks` ≥ 30 (≥ 1 s), `mid_band`
< 0.05, `min` at or below `closed_target`.

---

## Part 3 — Data and training: remove the cause

Part 2 makes the gripper decisive despite the label. Part 3 fixes the label. Both are needed:
without Part 3 every future model inherits the same defect.

### Change 5 — Hybrid gripper labelling in the converter

**Fixes:** the root cause. In AFO mode the converter derives **all 16 channels** from the
follower's own future state and ignores the recorded command topics entirely, so the gripper
label is the achieved contact position (dataset `q10` = 0.0041 / 0.0038 m) instead of the
commanded closure.

**Before writing code, verify the premise** on raw MCAP — the commanded finger must actually
go deeper than the observed one during grasps. The NAS is not mounted on the workstation and
no MCAP reader is installed there; raw is on S3.

```bash
# On a box with mcap_converter installed, against one grasping episode:
python - <<'PY'
from mcap_converter.core.reader import McapReader   # adjust to the local API
# print, per timestamp: observed follower_*_finger_joint1 vs the finger element of
# /follower_*_forward_position_controller/commands (joint_order index 7)
PY
```

*If the command is deeper than the observation during a grasp*, implement the hybrid label.
*If it is not* (e.g. the teleop sends the same position it reads), skip to the fallback at the
end of this change.

**File 1:** `packages/mcap_converter/src/mcap_converter/config/schema.py`, anchor
`action_from_observation` (was lines 138–145).

**AS-IS**

```python
    action_from_observation: bool = False

    # Positive number of output frames to look ahead when
    # action_from_observation=True. action[t] = observation[t + n]. The final
    # n observations are omitted because they have no future target. Default: 10.
    action_from_observation_n: int = 10
```

**TO-BE**

```python
    action_from_observation: bool = False

    # Positive number of output frames to look ahead when
    # action_from_observation=True. action[t] = observation[t + n]. The final
    # n observations are omitted because they have no future target. Default: 10.
    action_from_observation_n: int = 10

    # Joint-name suffixes whose action label is taken from the recorded command
    # topic instead of the future observation, even when action_from_observation
    # is true.
    #
    # Why: AFO labels a joint with the position the follower REACHED. For an arm
    # joint that is what we want - it is where the operator drove it to. For a
    # gripper it is not: the finger stops on the object at the contact position,
    # so the label carries no squeeze, and a position-controlled gripper commanded
    # to the position it already occupies exerts no force. Measured on
    # envelope-2916: action q10 = 0.0041 m on both fingers, and both the GR00T and
    # Pi0.5 fine-tunes reproduce it by re-opening instead of holding.
    #
    # Empty tuple = the previous behaviour, every channel from AFO.
    afo_command_sourced_joints: tuple[str, ...] = ()
```

**File 2:** `packages/mcap_converter/src/mcap_converter/config/loader.py`, anchor
`action_from_observation_n=` (was lines 252–254).

**AS-IS**

```python
            action_from_observation_n=config_dict.get(
                "action_from_observation_n", defaults.action_from_observation_n
            ),
```

**TO-BE**

```python
            action_from_observation_n=config_dict.get(
                "action_from_observation_n", defaults.action_from_observation_n
            ),
            afo_command_sourced_joints=tuple(
                config_dict.get(
                    "afo_command_sourced_joints",
                    defaults.afo_command_sourced_joints,
                )
            ),
```

**File 3:** `packages/mcap_converter/src/mcap_converter/core/extractor.py`.

Three edits.

**3a — keep reading the command topics.** Anchor: the `[ACTION SOURCE]` block (was lines
702–723), inside `extract_frames`.

*AS-IS*

```python
        action_topic_set = set()
        if self.config.action_topics:
            if self.config.action_from_observation:
```

*TO-BE*

```python
        action_topic_set = set()
        if self.config.action_topics:
            if (
                self.config.action_from_observation
                and not self.config.afo_command_sourced_joints
            ):
```

and add an `elif` branch that, when `afo_command_sourced_joints` is non-empty, subscribes to
the command topics exactly as the non-AFO path does (`action_topic_set = set(...)`,
`all_topics.extend(...)`, `self._check_action_topics_present(...)`) while printing that only
the listed joints come from them.

**3b — resolve command positions in AFO+hybrid mode.** Anchor: `_align_frame_at_cursor`
pass 2 (was lines 985–1000).

*AS-IS*

```python
        if not self.config.action_from_observation:
            for (role, robot), data in joint_buffers.items():
```

*TO-BE*

```python
        if not self.config.action_from_observation or self.config.afo_command_sourced_joints:
            for (role, robot), data in joint_buffers.items():
```

`_init_joint_buffers` must also stop returning early — same condition — so the `("action",
robot)` buffers exist. Its current first statement is:

```python
        if self.config.action_from_observation:
            return joint_buffers
```

which becomes:

```python
        if self.config.action_from_observation and not self.config.afo_command_sourced_joints:
            return joint_buffers
```

**3c — splice the two sources.** Anchor: `_finalize_afo_frame` (was line 1061). This is where
the future observation becomes the action.

*AS-IS*

```python
        ready_frame = pending_frames.popleft()
        if "observation.state" not in ready_frame:
            raise DataExtractionError(
                "action_from_observation requires an observation.state feature"
            )
        ready_frame["action"] = future_state.copy()
        return ready_frame
```

*TO-BE*

```python
        ready_frame = pending_frames.popleft()
        if "observation.state" not in ready_frame:
            raise DataExtractionError(
                "action_from_observation requires an observation.state feature"
            )
        action = future_state.copy()
        # Hybrid labelling: arm joints keep the t+N observation (where the operator
        # drove them to); the listed joints - the grippers - take the value the
        # operator COMMANDED at t, which carries the squeeze the achieved position
        # does not. Indices come from the same joint ordering the concatenation used,
        # so this stays correct if the layout changes.
        if self.config.afo_command_sourced_joints:
            commanded = ready_frame.get("action")   # written by pass 2 at time t
            if commanded is None:
                raise DataExtractionError(
                    "afo_command_sourced_joints is set but no command-topic action was "
                    "resolved for this frame; was the command topic recorded?"
                )
            for index in self._afo_command_sourced_indices():
                action[index] = commanded[index]
        ready_frame["action"] = action
        return ready_frame
```

Add the helper (cache it — it is called per frame):

```python
    def _afo_command_sourced_indices(self) -> list[int]:
        """Indices in the concatenated action vector taken from the command topic."""
        cached = getattr(self, "_afo_cmd_idx_cache", None)
        if cached is not None:
            return cached
        suffixes = tuple(self.config.afo_command_sourced_joints)
        indices: list[int] = []
        offset = 0
        # Same ordering as _align_frame_at_cursor: robots sorted, then each robot's
        # joint_order from its action topic config.
        for robot in sorted(
            {t.arm for t in self.config.action_topics.values() if t.arm}
        ):
            topic_cfg = next(
                t for t in self.config.action_topics.values() if t.arm == robot
            )
            for position, joint in enumerate(topic_cfg.joint_order or []):
                if joint in suffixes:
                    indices.append(offset + position)
            offset += len(topic_cfg.joint_order or [])
        if not indices:
            raise DataExtractionError(
                f"afo_command_sourced_joints={suffixes} matched no joint in "
                "action_topics[*].joint_order"
            )
        self._afo_cmd_idx_cache = indices
        return indices
```

**Careful:** `joint_order` in `action_topics` is `["joint1"…"joint7", "finger_joint1"]` —
finger **last**, index 7 — whereas the model's `model_joint_order` puts the finger **first**.
The indices above are in the converter's own concatenation order. Assert this in the test.

**File 4:** `packages/mcap_convert_gpu/src/mcap_convert_gpu/core/extractor.py` — the GPU
converter is a parallel copy (the same functions at lines 589, 702, 707). **Apply all of 3a–3c
there too.** They have drifted apart before; PR #2 had to realign them at +118/−70 each.

**File 5:** `configs/mcap_converter/openarm_bimanual_quest_afo.yaml`

*AS-IS*

```yaml
action_from_observation: true
action_from_observation_n: 10
```

*TO-BE*

```yaml
action_from_observation: true
action_from_observation_n: 10
# The gripper label comes from the command topic, not from the t+10 observation.
# See docs/data-conversion.md and the converter's schema note for why.
afo_command_sourced_joints: ["finger_joint1"]
```

**Test** (`packages/mcap_converter/tests/`): build a synthetic episode where the commanded
finger closes to 0.0 while the observed finger stalls at 0.004, and assert
`action[7] == 0.0` and `action[15] == 0.0` while `action[0:7]` still equals
`observation.state[t+10][0:7]`.

**Fallback if the command topic is unusable.** Binarise instead: label the gripper
`closed_value` when the observed finger is below a threshold and `open_value` otherwise. Same
splice point in `_finalize_afo_frame`; the model then learns a decisive squeeze even though
the demonstrations never contained one. Less faithful, but strictly better than labelling
contact-with-no-force.

**Cost:** a full reconvert of the 66 sessions plus `prepare_trainready_dataset.py`, then
retraining. Do not `--resume` an existing dataset — mixing label conventions in one dataset is
exactly the trap `docs/data-conversion.md` already warns about for AFO.

### Change 6 — Gripper state dropout during training

**Fixes:** the self-sustaining loop. The policy sees its own half-closed finger in
`observation.state`, which no demonstration contains, and the nearest in-distribution
behaviour is "re-open and retry". Dropping that input forces the gripper decision to come
from vision.

**File:** `packages/anvil_trainer/src/anvil_trainer/transforms.py` — new transform, same shape
as the existing three.

```python
# =============================================================================
# GripperStateDropoutTransform
# =============================================================================


class GripperStateDropoutTransform(Transform):
    """Randomly mask the gripper channels of observation.state during training.

    The policy otherwise conditions its gripper decision on its own previous gripper
    position. That is a stable loop while the command is decisive, and a vicious one
    once the command goes half-closed: no demonstration contains a half-closed gripper
    over an envelope, so the nearest behaviour in the data is to re-open. Masking the
    channel on a fraction of samples forces the decision onto the cameras, which do
    contain the evidence (is the envelope between the fingers?).

    Masked with the feature's own training mean, not zero: zero is a real, meaningful
    gripper position (fully closed) and would teach the policy that closed grippers
    are common.
    """

    def __init__(self) -> None:
        self._indices: list[int] | None = None
        self._fill: dict[int, float] = {}

    @property
    def name(self) -> str:
        return "gripper_state_dropout"

    def is_enabled(self, config: TrainingConfig) -> bool:
        return bool(config.gripper_state_dropout_p)

    def apply(self, item: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
        import random

        state = item.get("observation.state")
        if state is None or not self._indices:
            return item
        if random.random() >= config.gripper_state_dropout_p:
            return item
        for index in self._indices:
            state[index] = self._fill.get(index, 0.0)
        item["observation.state"] = state
        return item
```

Resolve `self._indices` from `meta/info.json` (`features["observation.state"]["names"]`,
matching `finger`/`gripper`) and `self._fill` from `meta/stats.json` `mean` in
`patch_metadata`, mirroring how `DeltaActionTransform._build_mappings` reads the dataset.

**Registration — file:** `packages/anvil_trainer/src/anvil_trainer/patches.py`, anchor the
`transforms` list (was lines 68–72).

*AS-IS*

```python
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            DeltaActionTransform(),
```

*TO-BE*

```python
            ExcludeObservationTransform(),
            TaskOverrideTransform(),
            DeltaActionTransform(),
            GripperStateDropoutTransform(),
```

plus the import next to the other three at the top of the file.

**Config — file:** `packages/anvil_trainer/src/anvil_trainer/config.py`. Add the field beside
`delta_exclude_joints` (was line 76):

```python
    gripper_state_dropout_p: float = 0.0  # P(mask the gripper channels of observation.state)
```

and parse it where `--delta-stats-n-steps` is parsed (was line 136):

```python
        _gsd_raw = _pop_argv("gripper-state-dropout-p") or "0"
        try:
            gripper_state_dropout_p = float(_gsd_raw)
        except ValueError:
            raise ValueError(
                f"--gripper-state-dropout-p={_gsd_raw!r} is not a valid float."
            ) from None
        if not 0.0 <= gripper_state_dropout_p <= 1.0:
            raise ValueError("--gripper-state-dropout-p must be in [0, 1]")
```

remembering to pass it through both `from_argv` (line ~338) and `from_dict` (line ~366).

**Recommended value: `--gripper-state-dropout-p=0.5`.** At inference, apply the same mask —
otherwise train and test disagree. For GR00T that means masking the two channels in the
node's observation assembly; note this **only** if the ablation shows it helps, because it
costs a second inference-side change.

**This is an ablation, not a certainty.** Run it against an identical no-dropout run and
compare grasp success, not loss.

### Change 7 — A gripper metric the checkpoint selector can see

**Fixes:** flow-matching validation loss averages 16 channels in metres and radians; the
gripper is 1/16 of it, and its whole dynamic range (0.05 m) is smaller than a single arm
joint's typical error. **No validation number we currently compute can detect this failure.**

**File:** `packages/anvil_eval/src/anvil_eval/metrics.py`, anchor `EpisodeMetrics` (line 11)
and `compute_episode_metrics` (line 32).

*AS-IS*

```python
    cosine_similarity: float
    pred_smoothness_mean: float
    pred_smoothness_std: float
    gt_smoothness_mean: float
    gt_smoothness_std: float
```

*TO-BE*

```python
    cosine_similarity: float
    pred_smoothness_mean: float
    pred_smoothness_std: float
    gt_smoothness_mean: float
    gt_smoothness_std: float
    # Gripper-specific. Per-joint MAE cannot express this: a policy that never closes
    # and one that closes 200 ms late can share an MAE while only one of them grasps.
    gripper_close_precision: dict[str, float] = field(default_factory=dict)
    gripper_close_recall: dict[str, float] = field(default_factory=dict)
    gripper_transitions_pred: dict[str, int] = field(default_factory=dict)
    gripper_transitions_gt: dict[str, int] = field(default_factory=dict)
    gripper_mid_band_fraction: dict[str, float] = field(default_factory=dict)
```

and, in `compute_episode_metrics`, treating "closed" as `< 0.025` on any joint whose name
contains `finger` or `gripper`:

```python
    gripper_close_precision, gripper_close_recall = {}, {}
    gripper_transitions_pred, gripper_transitions_gt, gripper_mid_band_fraction = {}, {}, {}
    for j, name in enumerate(joint_names):
        if "finger" not in name and "gripper" not in name:
            continue
        pred_closed = predicted[:, j] < 0.025
        gt_closed = ground_truth[:, j] < 0.025
        true_positive = int(np.sum(pred_closed & gt_closed))
        gripper_close_precision[name] = (
            float(true_positive / max(1, int(np.sum(pred_closed))))
        )
        gripper_close_recall[name] = (
            float(true_positive / max(1, int(np.sum(gt_closed))))
        )
        gripper_transitions_pred[name] = int(np.sum(np.abs(np.diff(pred_closed.astype(int)))))
        gripper_transitions_gt[name] = int(np.sum(np.abs(np.diff(gt_closed.astype(int)))))
        gripper_mid_band_fraction[name] = float(
            np.mean((predicted[:, j] > 0.005) & (predicted[:, j] < 0.045))
        )
```

Add the five to the `EpisodeMetrics(...)` construction (line ~98), to `compute_summary_metrics`
(line 108) and to the CSV fieldnames in `reporting.py` (line ~107).

**Read it like this:** recall < 0.5 means the policy does not close when the demonstration
does. `transitions_pred` ≫ `transitions_gt` is the dither. `mid_band_fraction` > 0.1 means it
lives between open and closed. Select checkpoints on these, not on `eval_loss` alone.

---

## Part 4 — Should we train all the parameters?

The honest short answer, per model:

| Model | Trainable now | Recommendation | Confidence |
|---|---|---|---|
| **GR00T N1.7** | **all 3.14 B** (`tune_llm` + `tune_visual` on) | **No — freeze the LLM, keep the vision encoder trainable** | High |
| **Pi0.5** | all (~4 B, full VLM) | Keep full VLM for the flagship, but run expert-only as the control | Medium |
| **LingBot-VA** | all 5.09 B (transformer) | Defensible now that a step is 0.54 s; the binding cost is 29 GB per checkpoint, not compute | Medium |

**Before any of it: unfreezing parameters cannot fix the gripper.** The failure is a label
and a decoder, not model capacity. A bigger fine-tune reproduces the same 4 mm contact label
more faithfully. Do Parts 2 and 3 regardless of what is decided here.

### 4.1 GR00T N1.7 — do not train the language model

Four reasons, in order of weight:

1. **There is one prompt.** `meta/tasks.parquet` holds a single task string for all 2 916
   episodes. A language model receives no signal that teaches it anything from a constant
   input; the gradient it does receive only moves it away from its pretrained weights.
2. **LeRobot's own default is frozen** — `tune_llm=False`, `tune_visual=False`,
   `tune_projector/diffusion/vlln=True` — and that is NVIDIA's fine-tuning recipe for N1.7.
   Our 1473 run followed it; only the 2916 run departed from it.
3. **The comparison is already confounded.** The 1473 run was frozen backbone at lr 1e-4;
   the 2916 run is full VLM at lr 3e-5. Two variables moved together, so neither the data
   effect nor the unfreezing effect is measurable from the pair.
4. **The learning rate was borrowed across architectures.** 3e-5 came from a Pi0.5
   (PaliGemma) run. Cosmos/Qwen3-VL is a different backbone; nothing was swept.

**Recommended run matrix** on envelope-2916, in priority order:

| Run | Flags | Why |
|---|---|---|
| **G1** (do this) | `--policy.tune_llm=false --policy.tune_visual=true --policy.optimizer_lr=5e-5` | The defensible middle: adapt vision to our cameras, leave language alone |
| **G2** (control) | `--policy.tune_llm=false --policy.tune_visual=false --policy.optimizer_lr=1e-4` | Stock recipe; isolates the data effect vs the 1473 run |
| **G3** (only if G1/G2 plateau) | `--policy.tune_llm=true --policy.tune_top_llm_layers=4 --policy.optimizer_lr=3e-5` | Partial language adaptation without moving 3 B parameters |

`tune_top_llm_layers` is a real field on `GrootConfig` (default `0`) and is forwarded to
`GR00TN17.from_pretrained` as a config override read back by `set_trainable_parameters`. It
is the knob to reach for before full unfreezing.

**File:** the launcher on amax-ws-03, `experiments/groot17-envelope2916/launch_2916_fullvlm.sh`
(this repo's `docs/training.md` §10 documents the same command).

*AS-IS*

```bash
  --policy.tune_llm=true \
  --policy.tune_visual=true \
  --policy.optimizer_lr=3e-5 \
```

*TO-BE (run G1)*

```bash
  # The LLM sees one constant prompt across all 2916 episodes, so unfreezing it
  # cannot teach it anything - it can only drift from the pretrained weights.
  # The vision encoder IS worth adapting: our cameras and lighting differ from
  # pre-training. tune_projector / tune_diffusion_model / tune_vlln stay true by
  # default and are not passed.
  --policy.tune_llm=false \
  --policy.tune_visual=true \
  --policy.optimizer_lr=5e-5 \
```

**Verify at startup:** `num_learnable_params` must now be **less than**
`num_total_params = 3 144 016 000`. If they are still equal, the flag did not take.

**Also change the run length.** 88 000 steps is 4 epochs. The 1473 run reached its
validation minimum at step 8 580 of 17 160 — **halfway** — and was flat within one standard
error afterwards. A full-parameter run on the same distribution overfits sooner, not later.
Set `--steps=44000` (2 epochs) and keep `eval_steps=1100`; the keep-best guardian protects
the selected checkpoint either way. This also halves the GPU cost of the matrix above.

**Optional, advanced — per-group learning rates.** If full-VLM is kept for some run, the
backbone should move slower than the action head. `GrootPolicy.get_optim_params` returns two
groups that differ only in weight decay:

```python
        return [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
```

GR00T trains through plain `lerobot-train`, so the clean injection point is a
`sitecustomize.py` shim like the existing `scripts/_ddp_shim`, splitting each group by whether
the parameter name starts with the backbone prefix and setting `"lr": base_lr * 0.1` on the
backbone half. Only worth doing if G1 and G2 both underperform.

### 4.2 Pi0.5 — keep full VLM, but fix three inconsistencies

Full VLM is more defensible here than for GR00T: the openpi recipe fine-tunes the VLM, and the
checkpoint that actually reaches the envelope today (`envelope-curated-v3-full-vlm-ckpt-019922`,
`train_expert_only=False`, `freeze_vision_encoder=False`, lr 3e-5) is a full-VLM one. But we
have never run the control, so "full VLM is better" is untested belief. `expert_only` already
exists in the launcher and is cheaper — run it on the same data and settle it.

Three defects in `scripts/run_pi05_multigpu.sh` to fix regardless:

**(a) The full-VLM learning-rate default does not match what we ship.**

*AS-IS* (line ~45)

```bash
  full_vlm)
    TRAIN_EXPERT_ONLY=false
    LEARNING_RATE="${FULL_VLM_LR:-1e-5}"
```

*TO-BE*

```bash
  full_vlm)
    TRAIN_EXPERT_ONLY=false
    # 3e-5 is what the deployed envelope checkpoints were actually trained with
    # (see pretrained_model/config.json: optimizer_lr=3e-5). The old 1e-5 default
    # meant every run had to override FULL_VLM_LR by hand or silently differ.
    LEARNING_RATE="${FULL_VLM_LR:-3e-5}"
```

**(b) Augmentation is off, while the GR00T method document claims the profile was replicated
from Pi0.5.** One of the two statements is wrong.

*AS-IS*

```bash
    --dataset.image_transforms.enable=false \
```

*TO-BE* — either pass the same `mild_photometric` dictionary the GR00T run uses, or leave it
off and correct the claim in the GR00T document. Do not leave both as they are.

**(c) The split is a plain shuffle, not the stratified split.**

*AS-IS*

```python
shuffled = episode_ids.copy()
random.Random(seed).shuffle(shuffled)
```

The stratified tooling (`scripts/curate_envelope_dataset.py`, `stratified_split_info.json`)
lives on the unmerged `feat/temporal-ensemble-smoother` branch. **Merge the split tooling on
its own branch, separately from the smoother**, then read `stratified_split_info.json` here
exactly as the GR00T and LingBot launchers do. Until then, the claim that Pi0.5 and GR00T
share a split by sha256 is only true for runs that were driven by hand.

**(d) Checkpoint policy.** `save_freq = steps/2` keeps two checkpoints and selects neither;
the final one is by construction the most overfit. Port the keep-best guardian from the GR00T
experiment.

### 4.3 LingBot-VA — the constraint is disk, not compute

The T5 cache changed the economics: 100 000 steps is ~15 hours, so a full-backbone run is now
affordable and the LoRA-vs-full question is no longer decided by wall clock. Full backbone is
reasonable **for this run** because the `lingbot_va_robotwin` base already has our 16 EEF
channels trained, so we are adapting rather than learning channels from scratch — which is
exactly why the earlier `lingbot_va_base` run on 1473 failed to converge.

What full backbone costs is **29 GB per checkpoint** (10 GB weights + 20 GB Adam state), and
that is what filled the disk and killed the run. LoRA at r=32 would make it ~0.3 GB. So:

* **Keep full backbone**, and fix retention (Change 0.1): NAS mounted, `--keep 2`,
  weights-only for all but the newest.
* **Run LoRA as the cheap control** once, at the same step count, if a second opinion is
  wanted on whether full backbone is buying anything.

**Watch the validation curve, which is currently flat and noisy:**

| step | 22 500 | 25 000 | 27 500 | 30 000 | 32 500 | 35 000 |
|---|---|---|---|---|---|---|
| `eval_loss` | 0.2075 | 0.1978 | 0.2436 | 0.2122 | 0.1982 | 0.2003 |

The spread between consecutive evaluations (0.198 → 0.244 → 0.212) is larger than any trend,
and `max_eval_samples=256` is the likely reason. Raise it to 1 024 before reading anything
into this curve; at 0.54 s/step the extra evaluation cost is minutes.

### Change 9 — Vendor the T5 prompt cache into the repo (do this first)

**Fixes:** a 20× speedup that exists only as an unversioned edit inside one machine's
`site-packages`, with the original saved beside it as `.anvil-orig`. A `pip install` erases it.
amax-ws-03 does not have it.

The patched method is pure in `(prompt, max_sequence_length)`, runs under `@torch.no_grad()`,
and the encoder is frozen and `.eval()` — so memoising it is sound. It should not live where
it lives.

Two acceptable homes, in order of preference:

1. **A transform-style patch in `anvil_trainer`.** `TransformRunner._patch` already exists to
   monkey-patch lerobot at training time and restore it afterwards
   (`packages/anvil_trainer/src/anvil_trainer/patches.py`). A `LingbotPromptCachePatch` there
   is versioned, tested, and applies on every machine that installs the package.
2. **A vendored `.patch` file** under `packages/` with a `scripts/apply_lingbot_patches.sh`
   that applies it to the venv and refuses if the upstream file's hash is unexpected.

Whichever is chosen, copy the measurement into the commit message — "11.3 s of an 11.4 s step
inside UMT5-XXL on CPU; GPU 90 % idle" is the justification, and it is currently written only
in a comment in an untracked file.

**Related, still open:** `_encode_training_latents` VAE-encodes the camera clips on **every**
step under `@torch.no_grad()` with a frozen VAE. Those latents are a pure function of the
frames and the resolution, so they can be pre-computed once for the whole dataset. That was
worth 256 days before; at 0.54 s/step it is worth far less, so treat it as a **later
optimisation, not a fix**. Measure where the 0.54 s now goes before building it.

---

## Part 5 — Getting this into git

Nothing in Part 2 is in version control today. The live GR00T stack — node, ensembler,
profiles, five documents — is untracked on one workstation, and a 45-line RTC change sits
uncommitted in `inference_node.py` on `main`.

Suggested branches, smallest first:

| Branch | Contents | Reviewable? |
|---|---|---|
| `feat/groot-inference-stack` | `inference_Gr00t/` as it stands today, plus its docs | Large but mechanical — it is new, not a diff |
| `fix/rtc-coverage-grace` | the uncommitted 45-line `inference_node.py` change on `main` | Yes, small |
| `feat/gripper-passthrough-and-latch` | Changes 1, 2, 3 + unit tests | Yes |
| `feat/stratified-split-tooling` | `curate_envelope_dataset.py` split off `feat/temporal-ensemble-smoother` | Yes |
| `feat/afo-command-sourced-gripper` | Change 5, both converters + tests | Yes |
| `feat/gripper-training-signals` | Changes 6 and 7 | Yes |
| `feat/lingbot-prompt-cache` | Change 9 | Yes, small |

Also decide **once, for both stacks**, what happens to `fix/remove-inference-action-clamp`
(open since 09-02): it sets `max_position_delta: null` for the Pi0.5 profiles, while the GR00T
profiles still carry `0.1`. Note that `0.1 rad` is **twice the gripper's entire travel**, so it
has never constrained the finger in either stack — it is an arm-joint bound only.

Finally, move `inference_Gr00t/configs/_backups/CHANGELOG.md` into `docs/`. It is the only
record of the 09-08 guidance experiment and the only place its numbers are written down.

---

## Appendix — order of work

1. **Change 0.1** — mount the NAS, fix retention, resume LingBot. *Blocking, today.*
2. **Change 9** — vendor the T5 cache. *One hour, protects a 20× speedup.*
3. **Change 0.2** — stop truncating monitor CSVs. *Minutes.*
4. **Changes 1–3** — ensembler passthrough, latch, finger limits. *One rebuild.*
5. **Change 4 arm A** — guidance on, no smoother. *Run before assuming 1–3 are needed.*
6. **Changes 4 B–D** — the rest of the matrix, scored on grasps.
7. **Change 7** — gripper metric. *Independent of everything, do it in parallel.*
8. **Run G1/G2** — the GR00T freezing decision, at 2 epochs.
9. **Change 5** — hybrid gripper label, after verifying the premise on raw MCAP.
10. **Change 6** — state dropout, as an ablation against the Change 5 dataset.
