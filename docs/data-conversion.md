[← Back to README](../README.md)

# Data Conversion

Convert a **recorded session** into a LeRobot v3.0 dataset.

A session is a directory of raw MCAP files (e.g. `data/raw/my-session/`) — one `.mcap` file per episode, all from the same recording run. Every tool below targets that same session directory, from the initial quality scan through to the finished dataset.

**The usual flow:** `mcap-valid` (scan the target session for problems) → `mcap-exclude` (move out episodes that must not be converted, then re-run `mcap-valid`) → `mcap-convert` (convert that session into a dataset) → `dataset-valid` (sanity-check the result) → `session-valid` (check the dataset against the raw session). `mcap-exclude` and `session-valid` are what the LOOP processing contract runs around every conversion; by hand they are optional. The other tools below (`mcap-to-video`, `merge-datasets`, `hf-upload`) are used as needed, not part of every conversion.

```
┌─────────────────┐
│ target session  │  data/raw/<session>/ — one *.mcap file per episode
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   mcap-valid    │  scan every episode in the session for quality issues
│                 │  - severity per episode: critical / warning / pass
│                 │  - writes <session>/mcap_valid_reports/report.{json,md}
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  mcap-exclude   │  move out episodes that must not be converted
│                 │  - low fps, silent command arms, a manual list
│                 │  - then re-run mcap-valid on what remains
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  mcap-convert   │  convert the whole session → one LeRobot v3.0 dataset
│                 │  - required: refuses to run without the report above
│                 │  - skips critical episodes by default (--include-flagged to override)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  dataset-valid  │  sanity-check the converted dataset
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  session-valid  │  check the dataset against the raw session
│                 │  - episode accounting, lengths, videos, labels
└─────────────────┘

Used as needed, not part of every conversion:
  mcap-to-video   — MCAP camera topics → MP4
  merge-datasets  — combine multiple LeRobot datasets
  hf-upload       — push a dataset to HuggingFace Hub
```

---

## mcap-valid

Scan a **raw session** — the same `data/raw/<session>/` directory you're about to convert — for quality issues before conversion: dropped frames, silent topics, cross-episode fps degradation. Point `-i` at the raw session, not a converted dataset: converted datasets have gap-filled timestamps that hide the original drops.

No config needed — topic roles (joint-state stream, camera stream, action command) are inferred entirely from each topic's own ROS2 message type. Every topic present in the file appears in the output; message types outside the 3 known roles show up as `unclassified` (informational only, never affects severity).

```bash
uv run mcap-valid -i data/raw/my-session
uv run mcap-valid -i data/raw/my-session --verbose            # show healthy topics too
uv run mcap-valid -i data/raw/my-session --fail-on-critical   # CI gate, exit 1 on any critical episode
uv run mcap-valid -i data/raw/my-session --topic /joint_states  # deep field-structure dump for one topic
```

A JSON report and a comprehensive Markdown report are **always** written to `<session>/mcap_valid_reports/report.{json,md}` — inside the session directory itself, not relative to wherever you happened to run the command from — in addition to the terminal table, no flags required. **`mcap-convert` refuses to run without this report** (see below) — running `mcap-valid` first is a required step, not optional.

Every run also prints a baseline table of every topic found in the file (`Topic | Type | Messages | Role`), regardless of severity — this replaces the old standalone `mcap-inspect` tool's topic listing. Pass `--topic TOPIC` for a deeper per-message field-structure dump of one topic (also folded in from the old `mcap-inspect`).

Severity model:

| Severity | Meaning |
|----------|---------|
| 🔴 `critical` | A camera or `joint_states` stream has zero messages, or an internal/leading/trailing gap — real data loss, no benign explanation. Also raised if a topic present in the majority of sibling episodes in the same batch is completely absent from this one (catches a camera/driver that never started) |
| 🟡 `warning` | An action topic has zero messages or an idle gap (e.g. one arm not yet picked up — this is normal teleop behavior, not necessarily a defect), or a stream's average fps dropped noticeably relative to the rest of the batch |
| 🟢 `pass` | No issues detected. Unclassified topics are always `pass` — they never affect the episode's overall severity |

An episode's overall status is its single worst topic's severity. `--fail-on-critical` only fails on `critical` — `warning` episodes convert normally unless you also pass `mcap-convert --include-flagged pass`.

**Known tradeoff — `action_from_observation` (AFO) datasets:** without a config, `mcap-valid` can't know a dataset is configured to derive actions from observations instead of a dedicated command topic. If the action-command topic was never recorded at all, it just doesn't appear in the report (not a warning). If the topic exists but has zero messages, it shows as `warning` here (vs. `pass` under the old config-aware behavior). This never blocks conversion — it only matters if you explicitly pass `mcap-convert --include-flagged pass` on AFO data.

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | A single MCAP file, or a session directory scanned recursively for `*.mcap` episode files |
| `--format` | `table` | `table` · `json` |
| `--output PATH` | — | Additionally write the report here (independent of the always-on `mcap_valid_reports/` output) |
| `--fail-on-critical` | — | Exit 1 if any episode has a critical issue — for CI gating |
| `--verbose` | — | Show per-topic detail even for episodes with no issues |
| `--topic TOPIC` | — | Deep field-structure dump for one topic (folded in from the old `mcap-inspect`) |
| `--max-samples N` | `5` | Max message samples to analyze per topic, for `--topic` |
| `--stream-gap-factor N` | `5.0` | Stream gap threshold, as a multiple of that topic's own median interval |
| `--stream-min-gap N` | `0.5` | Absolute floor (seconds) below which a stream gap is never flagged |
| `--action-warn-gap N` | `1.0` | Minimum idle duration (seconds) before an action-topic gap is reported |
| `--fps-tolerance N` | `0.15` | Fraction below the batch's median fps before flagging degradation |

---

## mcap-exclude

Move out of a session the episodes that must not be converted, before `mcap-convert` runs. It reads the session's `mcap-valid` report and the conversion config you are about to use, and moves each offending episode folder into `--excluded-dir`. It changes the session in place: run it on a copy when the raw recording must stay untouched.

| Reason | When |
|--------|------|
| `low_fps` | The reference camera (first `camera_topic_mapping` entry, the one `mcap-convert` detects fps from) snaps below `--fps`. `mcap-convert` refuses an `--fps` above the slowest episode's, skipped episodes included, so one slow episode would otherwise abort the whole session |
| `no_commands_<arm>` | The report shows zero messages on that arm's command topic (`action_topics`). `mcap-valid` rates this only `warning`, so `mcap-convert` would keep the episode and fill its actions with measured positions. `--no-commands all` only moves episodes where every arm is silent, keeping one-handed ones. Not applied to `action_from_observation` configs, which never read the commands |
| `manual` | Listed in `--manual`: a TSV of `episode<TAB>reason` lines, `#` comments allowed |

Critical episodes stay in place — `mcap-convert` skips them itself. Re-run `mcap-valid` afterwards so the report covers only the episodes that remain.

```bash
uv run mcap-exclude -i data/raw/my-session \
  --report data/raw/my-session/mcap_valid_reports/report.json \
  --config configs/mcap_converter/openarm_bimanual_quest.yaml --fps 30 \
  --excluded-dir data/excluded/my-session --out data/excluded/my-session.json
uv run mcap-exclude ... --no-commands all       # keep one-handed episodes
uv run mcap-exclude ... --manual reviewed.tsv   # also move out reviewed episodes
uv run mcap-valid -i data/raw/my-session        # refresh the report for what remains
```

`--out` records what was moved, for `session-valid`. `total` and `kept` count MCAP files (one per episode) before and after the move; `fps` is the episode's snapped reference-camera fps, and `manual_reason` is added for `manual` exclusions:

```json
{"excluded": [{"episode": "0007", "reasons": ["low_fps"], "fps": 24}], "kept": 41, "total": 42}
```

Exits 0 on success, also when every episode is excluded (`kept: 0`), and 2 on bad input: a missing report or config, an MCAP missing from the report (re-run `mcap-valid`), or an episode folder already present in `--excluded-dir`. Nothing is moved when it exits 2.

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | The session directory, one folder per episode |
| `--report PATH` | _(required)_ | The session's `mcap-valid` JSON report |
| `--config PATH` | _(required)_ | The `mcap-convert` config: reference camera and command topics |
| `--fps N` | _(required)_ | The output fps `mcap-convert` will run at |
| `--excluded-dir PATH` | _(required)_ | Where excluded episode folders are moved; must be outside the session |
| `--out PATH` | _(required)_ | Where the exclusions JSON is written |
| `--no-commands [any\|all]` | `any` | Exclude when any arm's command topic is silent, or only when all are |
| `--manual PATH` | — | TSV of episodes to exclude by hand (`episode<TAB>reason`) |

---

## mcap-convert

Converts every episode in a target session (`--input-dir`) into one LeRobot v3.0 dataset — this is the core, required step of the whole pipeline; everything else here supports or follows it.

**Requires a `mcap-valid` quality report for that same session to exist first** — either auto-discovered at the default `<input-dir>/mcap_valid_reports/report.json` (written automatically by `mcap-valid`, see above, inside the session directory itself), or pointed at explicitly with `--quality-report PATH`. If neither is found, `mcap-convert` exits with an error telling you to run `mcap-valid` first — it does not fall back to converting without one. This gate only checks that a report *file* exists — but `mcap-convert` also acts on its *contents* automatically: `--include-flagged` defaults to `warning`, so `pass` and `warning` episodes convert normally while `critical` episodes (e.g. a camera with zero messages) are skipped without you having to pass anything. `--include-flagged pass` is the stricter override — it also skips warning-level episodes, converting only fully-clean ones. `--include-flagged critical` is the looser override — it converts every episode regardless of severity.

Pick the config that matches your recording setup. `--config` is technically optional — omitting it falls back to a bare default `DataConfig()` that doesn't match any real robot's topic layout — so in practice always pass one of these:

| Config | Teleop mode | Arms | Action source | Image width |
|--------|-------------|------|---------------|-------------|
| `openarm_bimanual.yaml` | Leader-follower | Bimanual | Leader joint positions | 640 |
| `openarm_bimanual_quest.yaml` | Quest VR | Bimanual | Command topics | 640 |
| `openarm_bimanual_quest_afo.yaml` | Quest VR | Bimanual | Observation lookahead | 640 |
| `openarm_bimanual_quest_16x9.yaml` | Quest VR | Bimanual | Command topics | 480 |
| `openarm_single_quest.yaml` | Quest VR | Single (right) | Command topics | 640 |
| `openarm_single_quest_afo.yaml` | Quest VR | Single (right) | Observation lookahead | 640 |

Only the width of `image_resolution` is used. The height follows the cameras' native aspect ratio, so frames are never letterbox-padded, cropped or stretched: 1920×1080 cameras give 640×360 (or 480×270 with `_16x9`). All cameras in a session must share one aspect ratio — otherwise the conversion stops with an error. Datasets converted before this change were padded to 640×480 with black bars; reconvert them to get full frames — see [docs/training.md](training.md#diffusion).

**Command actions (default)** — the normal Quest configs use recorded position
commands (`action_from_observation: false`). `action_match: causal` selects the
newest command at or before the reference camera timestamp, keeping each arm
and its gripper together. An idle arm retains its last causal command; before
its first eligible command, its measured position is used. Reading ahead never
makes a future command eligible. Explicit
`action_match: nearest` retains the older nearest-neighbour behaviour and can
select future commands. The `_afo.yaml` configs remain an explicit opt-in.

Reconvert command datasets produced with the old `hold_last` fallback into a
new output directory. Do not append corrected episodes with `--resume`, because
that would mix temporal selection rules within one dataset.

**action_from_observation** — used by the single- and bimanual `_afo.yaml` configs when `/follower_*/commands` was not recorded or should not be used. Instead of reading from command topics, the converter derives actions from the follower's own joint positions shifted N output frames forward in time. When enabled, configured command topics are ignored even if they exist in the recording. Enable it with a positive lookahead:

```yaml
action_from_observation: true
action_from_observation_n: 10 # action[t] = observation.state[t + n] (default n=10, ≈333ms at 30fps)
```

The shift is applied after output-FPS subsampling, so its nominal duration is `N / fps` seconds. The final N observations in each episode are omitted because no future action target exists for them; every emitted row therefore satisfies the configured `t+N` relationship without clamping to the final state. The converter retains N pending image frames to construct these pairs, so unusually large N values increase memory use.

Do not use `--resume` to append corrected AFO episodes to a dataset created by a converter version that produced same-frame actions. Reconvert every source episode into a new output directory so one dataset never mixes the old `action[t] = observation[t]` rows with corrected `t+N` rows.

```bash
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --output-dir data/datasets \
  --fps 30
```

**`--output-dir`** sets the base output directory. Output is always saved to `<output-dir>/<input-dir-name>/`.

**`--output-path`** bypasses auto-naming entirely — the dataset lands exactly where you point it.

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input-dir PATH` | _(required)_ | The target session directory — walked recursively for `*.mcap` episode files |
| `--config PATH` | bare `DataConfig()` | Conversion config YAML (see table above) — always pass one in practice |
| `-o / --output-dir PATH` | `data/datasets` | Base output directory — dataset lands at `<output-dir>/<input-dir-name>/` |
| `--output-path PATH` | — | Full output path override — bypasses auto-naming |
| `--resume` | — | Skip already-converted episodes — safe to re-run after interruption |
| `--max-episodes N` | all | Convert only the first N episodes |
| `--fps N` | auto | Override output FPS (must not exceed source FPS) |
| `--tolerance-s N` | `0.001` | Timestamp sync tolerance in seconds |
| `--task NAME` | `manipulation` | Task name recorded into the dataset |
| `--buffer-seconds N` | `5.0` | Buffer window for time alignment, in seconds |
| `--debug-plot-episodes N` | `5` | Number of episodes to include in debug plots |
| `--vcodec` | `h264` | `h264` · `hevc` · `libsvtav1` |
| `--robot-type` | `anvil_openarm` | `anvil_openarm` · `anvil_yam` |
| `--act-from-obs-n-step N` | config value | Override `action_from_observation_n` at runtime: `action[t] = observation[t+N]` |
| `--no-calib` | off | Undistort camera frames with the session's `meta/camera_calibration.json` before resizing (see below). `--no-calib` keeps the frames as recorded |
| `--quality-report PATH` | auto-discovered | Path to a mcap-valid JSON report — mcap-convert requires one to exist; if omitted, the default `<input-dir>/mcap_valid_reports/report.json` is used |
| `--include-flagged [pass\|warning\|critical]` | `warning` | Highest severity tier to include when converting (inclusive threshold). `warning` (default) converts `pass` and `warning` episodes, skipping only `critical` ones automatically. `pass` is stricter — it also skips `warning` episodes. `critical` is the "convert everything" escape hatch — nothing is skipped. Works against whichever report the mandatory gate resolved (explicit or auto-discovered) |
| `--skip-episode-idx SPEC` | — | Manually skip episodes by 1-based index (see below) |
| `--push-to-hub` | — | Upload to HuggingFace Hub after conversion |
| `--hf-user NAME` | auto-detect | HuggingFace username, used when `--push-to-hub` is set |
| `--hf-repo NAME` | output dir name | HuggingFace dataset repo name, used when `--push-to-hub` is set |

**Skipping flagged or known-bad episodes** — two independent mechanisms, usable together:

```bash
# Scan first — critical episodes from this report are now skipped by default,
# with no extra flag needed:
uv run mcap-valid -i data/raw/my-session --format json --output /tmp/quality.json
uv run mcap-convert -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml \
  --quality-report /tmp/quality.json

# Also skip warning-level episodes (only convert fully-clean episodes)
uv run mcap-convert -i data/raw/my-session --config ... --quality-report /tmp/quality.json --include-flagged pass

# Opt out of quality-based skipping entirely — convert every episode, including critical ones
uv run mcap-convert -i data/raw/my-session --config ... --quality-report /tmp/quality.json --include-flagged critical

# Manually skip specific episodes by 1-based index — no quality report needed
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "3,7"       # episodes 3 and 7
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1:4"       # episodes 1,2,3 (end EXCLUSIVE, like Python's range())
uv run mcap-convert -i data/raw/my-session --config ... --skip-episode-idx "1,5:8,12"  # mixed: 1, 5, 6, 7, 12
```

`--skip-episode-idx` ranges follow Python slice convention — the end index is **not included** (`1:4` → episodes 1, 2, 3; matches `range(1, 4)`, not "1 through 4 inclusive"). An omitted start defaults to `1`; an omitted end reaches the last episode (`3:` → episode 3 through the end).

`--skip-episode-idx` doesn't need the quality report's *contents* to be relevant — but `mcap-convert` still needs *a* report to exist at all (per the mandatory gate above) before it will run, even if you're only using `--skip-episode-idx`.

### Lens undistortion (on by default)

By default, each camera frame is undistorted at its native resolution, then resized as usual. The intrinsics come from `<input-dir>/meta/camera_calibration.json`, keyed by raw camera name (the first segment of the image topic, e.g. `/cam_chest/image_raw/compressed` → `cam_chest`):

```json
{
  "cam_chest": {
    "camera_matrix": [[537.66, 0.0, 943.83], [0.0, 538.20, 574.32], [0.0, 0.0, 1.0]],
    "dist_coeffs": [0.01296, -0.01833, -0.00171, -0.00102, 0.00237],
    "width": 1920,
    "height": 1080
  }
}
```

`dist_coeffs` are OpenCV plumb_bob `[k1, k2, p1, p2, k3]`. The conversion stops if the file is missing, if a configured camera has no entry, or if a frame is not at the calibrated resolution. To convert a session without a calibration, pass `--no-calib` explicitly.

The undistorted camera matrix is computed once per camera with `cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0)`, which keeps only valid pixels: no black borders, at the cost of the outermost field of view. The dataset gets its own `meta/camera_calibration.json`, keyed by dataset feature:

```json
{
  "observation.images.base": {
    "source": "cam_chest",
    "undistorted": true,
    "alpha": 0.0,
    "camera_matrix": [[492.02, 0.0, 933.08], [0.0, 536.59, 572.68], [0.0, 0.0, 1.0]],
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    "width": 1920,
    "height": 1080,
    "output_width": 640,
    "output_height": 360
  }
}
```

`camera_matrix` is at the native resolution (`width` × `height`), not at the stored frame size (`output_width` × `output_height`); scale it by the resize factor to project into the stored frames. `dist_coeffs` are zero because the frames are already undistorted. The source calibration stays in the raw session. With `--no-calib`, frames are stored as recorded and no `meta/camera_calibration.json` is written.

---

## dataset-valid

Validate a converted dataset — loads it and runs 5 checks (load, inspect features, read first frame, batch-read, print stats).

```bash
uv run dataset-valid --root data/datasets/my-sessions
```

Expected: 5 checks all showing `[OK]`.

| Flag | Default | Description |
|------|---------|-------------|
| `--root PATH` | `output_dataset` | Dataset root directory to validate |
| `--repo-id ID` | `anvil_robot/manipulation_v1` | Dataset repository ID passed to `LeRobotDataset` |

To browse the converted dataset with a Rerun viewer, see [Dataset Visualization](dataset-viz.md).

---

## session-valid

Check a converted dataset against the raw session it came from, before it is published or merged. Point `--raw` at the session `mcap-convert` read (after `mcap-exclude`), `--dataset` at what it wrote, and `--exclusions` at the `mcap-exclude` JSON. The `mcap-valid` report is read from `<raw>/mcap_valid_reports/report.json`.

| Check | Result |
|-------|--------|
| `episode_accounting` | Fail unless raw MCAPs = dataset episodes + excluded + skipped by `mcap-convert` (above `--include-flagged` in the report). An episode `mcap-convert` dropped on its own (corrupt or empty frames) only warns, named through the `source_episode_dir` / `source_mcap` columns |
| `info_json` | Fail when `meta/info.json` is missing, `codebase_version` is not `v3.0`, or fps is not a positive integer (or not `--fps`) |
| `episode_lengths` | Fail when `meta/episodes` does not have one row per episode, an episode is empty, or the lengths do not add up to `total_frames` |
| `data_rows` | Fail when the data parquet rows or episodes differ from `meta/info.json` |
| `action_state` | Share of an episode's frames whose action equals the measured state, a sign of missing commands: warn above `--action-state-warn`, fail above `--action-state-fail` |
| `frames_vs_duration` | Warn when an episode's length / (recorded duration × fps) is outside [`--frames-ratio-min`, `--frames-ratio-max`] |
| `videos` | Fail unless the non-empty video features match `--expected-cameras`, or when a video feature has missing or empty MP4s |
| `labels` | Label columns (`--labels`) missing or empty: warn, or fail with `--require-labels`. A label that differs from the episode's raw `metadata.json`: fail |
| `provenance` | Fail when `conversion_config.yaml` is missing; warn when it records no `code_commit` / `code_tag` / `code_version` |

```bash
uv run session-valid --raw data/raw/my-session --dataset data/datasets/my-session \
  --exclusions data/excluded/my-session.json --out data/datasets/my-session.valid.json
uv run session-valid ... --require-labels --expected-cameras 2
```

The result goes to `--out`; the overall status is the worst check's:

```json
{"status": "warn",
 "checks": [{"name": "episode_accounting", "status": "pass", "detail": "raw 42 = dataset 39 + excluded 2 + skipped 1"}, ...],
 "episodes": {"raw": 42, "dataset": 39, "excluded": 2, "skipped": 1}}
```

Exits 0 for `pass` or `warn` and 1 for `fail`.

| Flag | Default | Description |
|------|---------|-------------|
| `--raw PATH` | _(required)_ | The raw session `mcap-convert` read |
| `--dataset PATH` | _(required)_ | The dataset `mcap-convert` wrote |
| `--exclusions PATH` | _(required)_ | The `mcap-exclude` JSON for this session |
| `--out PATH` | _(required)_ | Where the result JSON is written |
| `--expected-cameras N` | `3` | Non-empty video features the dataset must have |
| `--require-labels` | — | Fail, not warn, when episode labels are missing or empty |
| `--labels K1,K2` | the keys `mcap-convert` lifts | Label columns to check (default: `envelope_size`, `envelope_facing_side`, `destination_basket_side`, `arm`, whichever the raw or the dataset carries) |
| `--report PATH` | `<raw>/mcap_valid_reports/report.json` | The `mcap-valid` JSON report |
| `--include-flagged [pass\|warning\|critical]` | `warning` | The `--include-flagged` `mcap-convert` ran with, to know which episodes it skipped |
| `--fps N` | — | The fps `meta/info.json` must declare |
| `--action-state-warn F` | `0.2` | Share of action = state frames above which an episode warns |
| `--action-state-fail F` | `0.5` | Share of action = state frames above which an episode fails |
| `--frames-ratio-min F` | `0.9` | Lowest accepted length / (duration × fps) |
| `--frames-ratio-max F` | `1.05` | Highest accepted length / (duration × fps) |

---

## Additional tools

Used as needed — not part of every conversion.

### mcap-to-video

Extract image topics from an MCAP file to MP4 videos (one file per camera). Memory-efficient — processes one frame at a time.

```bash
uv run mcap-to-video -i recording.mcap -o ./videos
uv run mcap-to-video -i recording.mcap --scan-only                            # list topics only
uv run mcap-to-video -i recording.mcap -o ./videos --fps 30 --resize 640x480 # resize + fps
uv run mcap-to-video -i recording.mcap -o ./videos \
  --topics /cam_waist/image_raw/compressed                                    # specific topic
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input PATH` | _(required)_ | MCAP file or directory of MCAP files |
| `-o / --output-dir PATH` | `./videos` | Output directory for MP4 files |
| `--topics TOPIC [...]` | auto-detect | Specific topics to convert |
| `--fps N` | `30` | Output video FPS |
| `--codec` | `libx264` | `libx264` · `libx265` · `libaom-av1` |
| `--crf N` | `23` | Constant rate factor — lower = better quality |
| `--resize WxH` | — | Resize frames, e.g. `640x480` |
| `--scan-only` | — | List image topics without converting |

### merge-datasets

Merge two or more LeRobot datasets into one. All datasets must share the same feature schema; use `--remove-features` to strip mismatched features before merging.

```bash
uv run merge-datasets data/datasets/ds-a data/datasets/ds-b \
  --output data/datasets/ds-merged

# Strip extra features from datasets recorded with velocity+effort
uv run merge-datasets data/datasets/ds-a data/datasets/ds-b \
  --output data/datasets/ds-merged \
  --remove-features observation.velocity,observation.effort
```

| Flag | Description |
|------|-------------|
| `PATH [PATH ...]` | Two or more dataset paths to merge (positional) |
| `--output PATH` | _(required)_ Output path for the merged dataset |
| `--remove-features F1,F2` | Comma-separated features to strip from any dataset that has them |

> When `--remove-features` is used, a trimmed copy (`<path>-trimmed`) is written alongside the original and reused on subsequent runs.

### hf-upload

Upload a converted LeRobot dataset to HuggingFace Hub.

```bash
# Login first (one-time)
huggingface-cli login

uv run hf-upload /path/to/dataset                                  # repo-id auto from dir name
uv run hf-upload /path/to/dataset --repo-id your-org/my_dataset
uv run hf-upload /path/to/dataset --repo-id your-org/my_dataset --private
```

| Flag | Default | Description |
|------|---------|-------------|
| `dataset_path` | _(required)_ | Path to local dataset directory (positional) |
| `--repo-id ORG/NAME` | auto from dir name | HuggingFace repository ID |
| `--private` | — | Make the repository private |
| `--force` | — | Skip confirmation prompt if repo already exists |
| `--hf-user USER` | auto-detect | HuggingFace username |

---

[← Back to README](../README.md)
