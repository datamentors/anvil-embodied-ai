[← Back to README](../README.md)

# Review — envelope training runs and the gripper open/close failure (2026-09-08)

**Scope.** (1) How the GR00T N1.7, Pi0.5 and LingBot-VA envelope trainings are run, including
the decision to train all GR00T parameters. (2) Why the live runs move the arm to the envelope
but open and close the gripper without grasping. Every number below was measured on the
`datamentors` inference workstation (`100.116.187.114`) or read from the code on `main`
(`d103e65`) plus the untracked `inference_Gr00t/` stack. `amax-ws-03` (GR00T trainer) was
not reachable this session, so the GR00T validation curve is quoted from the run's own method
document, not re-read.

---

## 1. Headline

**The gripper failure is a labelling + decoding problem, not a limiter problem, and it is the
same problem in every model we train from this dataset.**

1. **The gripper label is the follower's *achieved* finger position, never the operator's
   command.** `openarm_bimanual_quest_afo.yaml` sets `action_from_observation: true`, and in that
   mode the converter *ignores the recorded command topics entirely*
   (`extractor.py:1250`, `_finalize_afo_frame`: `action[t] = observation.state[t+10]`). In the
   demonstrations the fingers stop on the envelope at **~4 mm** (`q10 = 0.0041 / 0.0038`), so the
   policy learns to command *the contact position with zero squeeze margin*. A position-controlled
   gripper sent to the position it is already at exerts no force. This is the ALOHA/ACT lesson:
   the gripper action must be the leader/command, not the follower state.
2. **The gripper channel is near-binary, and both of our decoders are linear operators on it.**
   The RTC queue *replaces* on merge; the ensembler *averages*. Averaging a bimodal channel
   yields half-closed values, which a 150 N direct-drive gripper follows faithfully.
3. **GR00T chunks disagree on gripper direction 65 % of the time when guidance is OFF**
   (João's measurement today, `inference_Gr00t/configs/_backups/CHANGELOG.md`). Guidance ON
   removed the dither (13.7 → 0.17 reversals/s) but, stacked with `temporal_ensemble` coeff 0.1
   over 8 chunks, the gripper **never closed at all**, so it was reverted.
4. **The policy then sees its own half-closed gripper in `observation.state`.** No demonstration
   contains a half-closed gripper hovering over an envelope, so it is out of distribution and the
   nearest behaviour in the data is "re-open and retry". That is why the flicker self-sustains.

---

## 2. What the live runs actually did (measured from the monitor CSVs)

Observed state is in controller order (finger at index 7 / 15); raw model output is in model
order (finger at 0 / 8). "Closed" = command < 0.025 m; "mid band" = 0.005–0.045 m.

| Run | Model / profile | Duration | Gripper | close/open transitions | median closed phase | time in mid band | min observed |
|---|---|---|---|---:|---:|---:|---:|
| `inference_data.csv` (19:09) | GR00T 2916 ckpt 011000, `_ensemble_live` (guidance OFF, ensemble 0.1/8) | 513 s | left | **220** | **5 ticks (170 ms)** | **45 %** | −0.0004 |
| same | same | | right | 30 | 3 ticks | 6 % | 0.0116 (never closed) |
| `run_16h02_1473profile…csv` | GR00T 2916 ckpt 011000, 1473-profile | 626 s | right | **338** | **4 ticks** | **53 %** | 0.0012 |
| `monitor_output/inference_data.csv` (09-05) | Pi0.5 `envelope-curated-v3-full-vlm-ckpt-019922`, guided RTC, no ensembler | 138 s | right | 60 | **29 ticks (~1 s)** | 12 % | −0.0002 |

Reading: GR00T closes for 100–170 ms and reopens, at the ~6.8 Hz re-plan cadence — this is
chunk-to-chunk disagreement reaching the motor. Pi0.5 under guided RTC closes decisively
(~1 s, reaches the dataset's closed value) but still re-attempts every ~4.6 s: the grasp is
decisive but does not hold, which is consistent with finding 1 (no squeeze) rather than dither.

**Saturation is not the cause.** The GR00T live env runs `ENFORCE_JOINT_POSITION_LIMITS=true`;
the 7 699 right-finger saturations in the 19:09 run are all on the *open* side
(`0.0502 → 0.0500`). The raw command never went below −0.0022. But note for later: the URDF
finger limit is `[0.0, 0.05]` while the recorded mechanism reaches **−0.0048 … +0.0567**. Once
the label is fixed to ask for a squeeze, the clamp at 0.0 *will* block it. Widen the finger
limits to the measured mechanical range (or calibrate the zero) before shipping fix 5 below.

**Chain of decoders on the GR00T path (`inference_Gr00t/…/inference_node.py`):**
`predict_action_chunk(obs, inference_delay, prev_chunk_left_over=None when guidance OFF)` →
`ActionQueue.merge` (replace from `merge_delay_steps`) → `ChunkEnsembler.value_at` (age-weighted
mean over ≤8 chunks, **all 16 dims, gripper included**, `temporal_ensembler.py`) →
`_saturate_mechanical_stops` → `_validate_absolute_joint_targets` → `ActionLimiter` (`max_delta`
0.1 rad: 2× the finger's full travel, i.e. no bound on it) → publish. Nothing in this chain is
gripper-aware.

---

## 3. Fixes, in order

### A. Inference — config/code on the node, days

1. **Exempt the gripper dims (model-order 0 and 8) from the ensembler.** Averaging is the wrong
   operator for a near-binary channel. `ChunkEnsembler.value_at` takes a full row; add a
   `passthrough_dims` mask and return `fallback[d]` for those.
2. **Add a gripper latch (Schmitt trigger) on the *fresh* model output**, independent of RTC or
   ensembling: close when raw < 0.020, open when raw > 0.035, hold otherwise; minimum dwell
   ≥ 0.5 s after a transition; while latched closed publish the **full-close target** (mechanical
   minimum), not the predicted contact value. This gives the squeeze margin the label lacks, and
   removes the 3–5-tick flicker regardless of which smoother wins for the arm joints.
3. **Re-run the A/B as three arms, judged on grasps, not smoothness:** (a) guidance ON +
   `smoother.type: none` (the Pi0.5-equivalent control — this is the one we have never run for
   GR00T); (b) guidance ON + ensemble coeff 0.01 with gripper exempt; (c) guidance OFF + ensemble
   with gripper exempt + latch. Same 20 envelopes each; count grasp-and-lift.
4. **Widen `joint_position_limits` for the fingers** to the measured range and keep enforcement on.
   With Pi0.5 (`ENFORCE=false`) raw commands already reach −0.0044; GR00T is clamped to 0.0.

### B. Data and training — next dataset build, ~1 week

5. **Hybrid labelling in the converter**: arm joints via AFO (`state[t+N]`), **gripper via the
   recorded command topic** (`/follower_*_forward_position_controller/commands`, finger element).
   The topic *is* configured in `openarm_bimanual_quest_afo.yaml`; AFO mode just drops it.
   Verify on raw MCAPs first that the commanded finger goes deeper than the observed finger during
   grasps (the NAS is not mounted on the workstation and no MCAP reader is installed there; raw is
   on S3). If the command is not usable, **binarize the gripper label** (threshold 0.02, closed =
   mechanical minimum) so the model learns a decisive squeeze.
6. **State dropout on the gripper dims** (mask or noise `observation.state[0]` and `[8]` during
   training, mask at inference) so gripper decisions come from vision, breaking the self-sustaining
   half-closed feedback loop in finding 4.
7. **A gripper-specific validation metric per checkpoint.** Flow-matching val loss is 1/16 dims
   in 0.05-m units; it cannot see this failure. `anvil_eval` already computes per-joint MAE — add
   close-timing precision/recall on the finger channels on the val split and log it beside
   `eval_loss` for checkpoint selection.

---

## 4. Training-run review

### 4.1 GR00T N1.7 on envelope-2916 — full-parameter fine-tune

What runs: `tune_llm/visual/projector/diffusion/vlln = true` (3.144 B trainable), AdamW
`3e-5`, batch 16×4, 88 000 steps ≈ 4 epochs, `eval_steps 1100`, `save_freq 5500`, keep-best
guardian. The live checkpoint is **011000 = 0.5 epoch**.

Concerns:

* **Two variables changed at once, again.** 1473 run = frozen backbone, lr 1e-4; 2916 run = full
  VLM, lr 3e-5. Neither the data effect nor the unfreezing effect can be isolated. The 1473 run
  reached its val-loss minimum at step 8 580 of 17 160 (~2.5 epochs) and was flat within 1 SEM
  after — so 4 full-parameter epochs on the same distribution will overfit earlier, not later.
  **Run the frozen-backbone control on 2916** (NVIDIA's own N1.7 recipe: tune projector +
  diffusion + vlln, lr 1e-4). It is ~1 GPU-day on amax-ws-03.
* **The learning rate is borrowed from Pi0.5 (PaliGemma), not chosen for Cosmos/Qwen3-VL.** If
  full-VLM stays, use two parameter groups (backbone ≤ 1e-5, action head 1e-4) or LoRA on the LLM.
  `GrootConfig` has one `optimizer_lr`; this needs a small patch in the trainer.
* **Single-task data cannot justify unfreezing the language model.** There is one prompt. The VLM
  gains nothing from it and can only drift. Unfreezing the *vision* encoder is the defensible half
  (our cameras and lighting differ from pre-training).
* **`eval_split` relies on LeRobot taking the *last N episodes per task*.** It is verified by eye
  at startup. Add an assertion that the resolved holdout equals `stratified_split_info.json`'s
  `val_episodes`, so a future LeRobot change cannot silently randomise the split.
* **Keep-best-only leaves no resume point** (known). Keep best + last; it is 36 GB.
* **Augmentation is photometric only and untested for GR00T** (known). Not the gripper problem.

### 4.2 Pi0.5 launcher (`scripts/run_pi05_multigpu.sh`, main)

* `full_vlm` defaults to **lr 1e-5** and `expert_only` to 3e-5, while the GR00T method document
  says 3e-5 "worked for the Pi0.5 full-VLM run". The deployed `envelope-3000-full-vlm-ckpt-010982`
  has `optimizer_lr 3e-5`, so `FULL_VLM_LR` was overridden by hand. Record the effective value in
  the doc, or change the default.
* `--dataset.image_transforms.enable=false`: **Pi0.5 on main trains with no augmentation**, yet the
  GR00T document claims `mild_photometric` "replicated from Pi0.5". One of the two is wrong;
  check the Pi0.5 run's log before quoting comparability.
* The launcher's split is `random.Random(seed).shuffle` — **not the stratified split** the GR00T
  document says both trainings share by sha256. The stratified tooling
  (`curate_envelope_dataset.py`) lives on the **unmerged** `feat/temporal-ensemble-smoother`
  branch. Merge the split tooling separately from the smoother.
* `save_freq = steps/2`: two checkpoints, no validation-based selection. The final one is by
  construction the most overfit. Port the keep-best guardian from the GR00T experiment.

### 4.3 LingBot-VA on envelope-2916 (EEF), full backbone

* **The run is dead.** Killed by `KeyboardInterrupt` at step **11 200 of 2 109 384** after 37 h
  (0.03 epochs, 10.5 s/step). `eval_steps` was 21 972 (≈2.7 days) and `save_freq` 87 891
  (≈10.7 days): **37 GPU-hours produced no evaluation and no checkpoint.** Cadence defined as
  epoch fractions is wrong for a 256-day run; define it in wall-clock: eval every ~2 000 steps
  (6 h), checkpoint every ~4 000 steps (12 h), and a step budget (the September run used 20 000).
* **The per-step cost is dominated by work that can be cached.** The task prompt is a single
  string, yet UMT5 runs on CPU "once per episode" — cache one embedding. Wan-VAE latents for the
  training videos can be pre-encoded once (the VAE is frozen by construction); this is standard
  for video-diffusion fine-tuning and removes the encode from every step.
* **LoRA vs full backbone** was switched at the same time as 2× data; the previous run was 1.6 %
  trainable. Same objection as 4.1.
* **Actions are EEF poses relative to the episode's first frame.** A continuously running node has
  no "first frame"; `inference_lingbot/` (untracked) must define when the reference pose is
  captured. Review it before any live run.
* The gripper channel keeps the same AFO observed-state label, so fixes 5–7 apply here too.

---

## 5. Repo hygiene that affects reproducibility

* `inference_Gr00t/`, `inference_lingbot/`, five new `docs/*.md` and a 45-line change to
  `inference_node.py` (RTC coverage-failure grace) are **uncommitted on the workstation**. The
  live GR00T system is not in git.
* `fix/remove-inference-action-clamp` (2026-09-02) removes the `max_position_delta` default for
  the Pi0.5 stack; the GR00T profiles still carry `0.1`. Decide once, for both stacks.
* `feat/temporal-ensemble-smoother` bundles the Pi0.5 ensembler, the stratified split tooling and
  checkpoint archiving. Split it.
* `_backups/CHANGELOG.md` under `inference_Gr00t/configs/` is the only record of today's
  guidance ON/OFF experiment and its numbers; move it into `docs/`.
