[← Back to README](../README.md)

# Run Inference

All inference scenarios go through `scripts/run_inference.sh`.

**Start by copying `.env.example` to `.env` and editing it for your setup:**

```bash
cp .env.example .env
# then edit .env
```

### `.env` Variables

`.env` is the primary configuration file — set it once and reuse across runs. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | Yes (inference) | Host path to checkpoint dir. Must be absolute or start with `./` — bare relative paths are treated as Docker named volumes. |
| `ROS_DOMAIN_ID` | Yes | ROS2 domain ID — must match the Anvil Devbox. Leave empty for localhost-only. |
| `CYCLONEDDS_URI` | Yes | Path to CycloneDDS XML config (e.g. `configs/cyclonedds/two_pc_gpu.xml`). |
| `LEROBOT_EXTRAS` | VLA only | Comma-separated policy extras built into the Docker image — e.g. `smolvla`, `pi,smolvla`. **Rebuild the image after changing:** `docker compose build`. ACT and Diffusion leave this empty. |
| `HF_CACHE` | VLA only | Host path to HuggingFace model cache (default: `~/.cache/huggingface`). Required for Pi0, Pi0.5, SmolVLA — they load the PaliGemma tokenizer at runtime. |
| `CONFIG_FILE` | Yes | Path to inference config YAML (default: `configs/lerobot_control/inference_default.yaml`). |
| `ACTION_TYPE` | No | Action type passed to the **inference monitor node** (`inference_monitor_node`) only. The main inference node always reads this from `anvil_config.json` in the checkpoint via `resolve_action_type()` — this env var does **not** override it. Options: `absolute` · `delta_obs_t` · `delta_sequential`. |
| `ECHO_TOPIC_ONLY` | No | `true` = skip model loading, subscribe topics and log FPS only. For verifying DDS connectivity without a GPU or checkpoint. Equivalent to `--echo-topic-only`. |
| `MONITOR_ENABLE` | No | `true` = enable the inference monitor node (records per-step CSV + PNG report). Equivalent to `--monitor-enable`, but without the auto-plot on exit and output dir pre-creation that the flag provides. |
| `DEBUG` | No | `true` = enable extra metrics: action smoothness, queue depth stats, Action FPS. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | No | Native CPU thread-pool limits (default: `4`). These prevent the inference process and spawned camera workers from oversubscribing the host and delaying CUDA work. |

For full descriptions and defaults, see [`.env.example`](../.env.example).

### Script Flags

The script flags are a lightweight way to override behaviour at the command line without editing `.env`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor-enable] [--echo-topic-only] [--debug] [COMPOSE_ARGS...]
```

| Flag | What it does |
|------|-------------|
| `--fake-hardware` | Switches to `docker-compose.fake-hardware.yml` — simulates a 2-PC setup locally over a bridge network (CycloneDDS, no real robot). |
| `--monitor-enable` | Adds `--profile monitor` to the compose command. In production (non-fake-hardware) mode also exports `MONITOR_ENABLE=true`, pre-creates the output dir as the current user, and auto-plots the CSV on exit. |
| `--echo-topic-only` | Exports `ECHO_TOPIC_ONLY=true` — subscribes to topics and logs FPS without loading a model or GPU. Equivalent to setting `ECHO_TOPIC_ONLY=true` in `.env`. |
| `--debug` | Exports `DEBUG=true` — enables extra metrics: action smoothness, queue depth stats, Action FPS. Equivalent to setting `DEBUG=true` in `.env`. |

## Test with Fake Hardware First (Recommended)

```bash
# 1. Verify DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor-enable up --build

# 2. Validate full pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware --profile inference up --build
```

> **Fake-hardware note:** `--echo-topic-only` / `ECHO_TOPIC_ONLY` and `MONITOR_ENABLE` env vars are
> **not** read by `docker-compose.fake-hardware.yml`. The monitor service hardcodes
> `echo_topic_only:=true` regardless; the inference service does not expose `MONITOR_ENABLE`.
> These variables only take effect with the production `docker-compose.yml`.

If `Control Loop` hits 30 Hz, the setup is ready for real hardware.

## Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With inference monitor
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor-enable up --build

# Verify DDS connectivity without a checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

> **`MODEL_PATH` must be absolute or start with `./`.** Bare relative paths are treated as named Docker volumes.
> ```bash
> MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last   # recommended
> MODEL_PATH=./model_zoo/my-task/checkpoints/last        # also valid
> ```

## Inference Config (`configs/lerobot_control/inference_default.yaml`)

Before running, review this file:

**Model**
```yaml
model:
  task_description: null
  # VLA-only (SmolVLA / Pi0 / Pi0.5): task prompt the model was trained on.
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
```

**Per-model inference tuning** — override checkpoint defaults without retraining:
```yaml
inference_tuning:

  act:
    n_action_steps: null
    # Steps to execute per chunk before re-running inference.
    # null = use training value. Jittery? → raise. Hesitates? → lower.
    temporal_ensemble_coeff: null
    # Re-infers every step with exponentially weighted overlapping predictions.
    # Use 0.01 (paper default). Forces n_action_steps=1.

  diffusion:
    n_action_steps: null
    # Steps to execute per chunk. null = use training value.
    num_inference_steps: 10
    # Denoising iterations at inference time.
    # null = num_train_timesteps (100 steps, ~300ms on GPU).
    # 10   = ~30ms on GPU — recommended for real-time deployment.

  rtc:
    # VLA models only (SmolVLA / Pi0 / Pi0.5)
    inference_delay: 10
    # Fallback step-count before LatencyTracker auto-calibrates.
    # Rule of thumb: ceil(first_inference_ms × control_freq / 1000)
    queue_trigger_threshold: 50
    # Re-trigger inference when ActionQueue depth ≤ this.
    execution_horizon: 12
    # Steps consumed per chunk before the next inference fires.
    max_guidance_weight: 10.0
    prefix_attention_schedule: EXP
    readiness_guided_forwards: 5
    # Required consecutive guided refills before publication can start.
    readiness_latency_guard_steps: 2
    # Extra control periods added to the worst guided latency.
    readiness_index_phase_tolerance_steps: 1
    # Allowed control-timer phase difference between wall time and queue index.
    readiness_scheduler_guard_steps: 1
    # Additional queue step reserved for dispatch/polling scheduler jitter.
    readiness_min_guided_overlap_steps: 3
    # Minimum guided prefix that must survive the bounded refill latency.

diagnostics:
  rtc_timing: false
  rtc_cuda_events: false
  rtc_provenance: false
  # Enable only in a reviewed shadow profile. The node emits correlated
  # per-stage wall timings and queries CUDA events asynchronously, without
  # synchronizing the inference stream or changing readiness calculations.
  # rtc_provenance additionally records the exact joint/camera counters,
  # ROS header stamps, receipt ages and a digest/summary of each action chunk.
```

**Joint-state process isolation:** the `joint_state_worker` launch
parameter moves the `/joint_states` subscription into a spawned ROS2 process.
The complete serialized message and callback-ingress monotonic timestamp cross
a seqlock-protected shared-memory slot and are parsed normally in the main
process. The option defaults to `false`. Debug-only command topics may enable
it directly; a live profile must additionally set
`runtime.allow_live_joint_state_worker: true` so promotion cannot happen by
changing only a topic name.

**Safety limits:**
```yaml
safety:
  max_position_delta: 0.1
  # Hard limit on joint position change per control step (radians).
  min_position_delta: null
  joint_limit_tolerance: 0.000001
  saturate_joint_targets: []
  saturate_joint_margins: {}
  # Optional bounded acceptance for known recording artefacts. Every named
  # joint requires an explicit positive margin no larger than 0.05 rad.
  # Targets inside the margin clamp to the existing hard limit and are counted;
  # larger violations still fail closed.
  joint_position_limits:
    # Required full mapping keyed by exact ROS joint names. See the default
    # config for all 16 values sourced from the deployed robot URDF.
    follower_l_finger_joint1: [0.0, 0.05]
    # ... all remaining configured joints ...
```

Absolute limits are evaluated after model/controller reordering, before the
delta limiter can hide an invalid raw target, and once more on the final
command. The node prepares and validates both arms before publishing either
one, so a bad target on the right arm cannot leave a left-only command behind.
Missing, extra, inverted, or non-finite limit entries abort startup.
Targets outside the configured limit plus the numerical tolerance fail closed
unless that exact joint has a bounded saturation margin. Saturation never
expands the hard limit: accepted overshoot is clamped to the existing bound,
counted in the periodic statistics and limited to 0.05 rad. It is intended only
for measured recording artefacts and cannot replace correcting the dataset or
policy. See `inference_envelope_afo.yaml` for a profile whose margins document
the checkpoint statistics from which they were derived.

`inference_default_afo.yaml` provides the three-camera AFO feature mapping
(`base`, `left_wrist`, `right_wrist`). `inference_envelope_afo.yaml` extends it
with the envelope task prompt, measured RTC settings, watchdog limits, bounded
saturation and opt-in latency/provenance diagnostics. Neither profile contains
a checkpoint path; pass that separately at launch time.

**Fail-closed input watchdog:**
```yaml
watchdog:
  camera_timeout_sec: 0.25
  joint_state_timeout_sec: 0.10
  max_sensor_skew_sec: 0.10
  max_action_age_sec: 1.50
  startup_grace_sec: 10.0
```

Freshness uses local monotonic receive time, not ROS message stamps. Publication
starts only after all configured cameras and required joints are present, finite,
fresh, mutually synchronized, and have produced a new sequence. If an input
stops, an observation repeats, inference fails, or an action is invalid, the node:

1. latches the watchdog;
2. clears RTC, classic-policy, delta-restore, and limiter state;
3. suppresses all action publication; and
4. discards inference results that were already running when the fault occurred.

For RTC policies, input health and policy readiness are separate gates. After
startup, reset, or rearm, the node performs these phases without publishing:

1. discard one unconditional GPU/model cold forward;
2. merge one fresh unguided chunk as a provisional seed; and
3. require five consecutive guided refills to pass all sustainability bounds.

RTC alignment and action freshness use separate clocks. Under the queue lock,
dispatch captures the queue identity, depth `q0`, consumer index `i0`, leftover,
and time `t0`. Immediately before merge under the same lock it captures `q1`,
`i1`, and `t1`. With runtime `L=t1-t0` and index-phase tolerance `P=1`:

```text
pre-ready:  i1-i0 = 0; D_merge = ceil(f * L)
post-ready: D_idx = i1-i0 = q0-q1
            0 <= D_idx <= ceil(f * L)+P
            q1 >= 1; D_merge = D_idx
```

Queue identity, index, depth, leftover length, or consumption above the
wall-clock upper bound is a fail-closed rejection. Consumption may be below the
wall estimate when the ROS executor delays the publish timer; in that case the
exact queue index remains authoritative because it is aligned with the leftover
passed to RTC. Source observation age is not used as a merge delay; it remains
the independent freshness/provenance clock.

For chunk length `C`, control frequency `f`, queue threshold `T`, exact source
age at merge `A`, action-age limit `M`, execution horizon `H`, latency guard
`G=2`, scheduler guard `S=1`, and the last five guided runtimes:

```text
L_bound   = max(last 5 L) + G/f
D_bound   = ceil(f * max(last 5 L)) + G
q_start   = C - D_merge
wait      = max(0, q_start - T)
q_trigger = q_start - wait
q_required = max(0, q_trigger - S)

q_required >= D_bound + 1
A + (wait + S)/f + L_bound < M
max(0, min(H, q_required) - D_bound) >= 3
```

Only the fifth consecutive passing refill reports `[RTC] POLICY_READY`. A miss
before readiness discards the provisional queue and seed; the next result must
be unguided and seed a new proof. Before readiness the queue threshold is
intentionally ignored because publication cannot drain the provisional queue.
A miss after readiness—including an empty queue, stale result, queue/index
misalignment, insufficient refill coverage, or insufficient useful
guidance—latches the watchdog and clears the queue while holding the same safety
lock used by publication.

These checks deliberately expose timing/configuration incompatibilities. For
example, `C=50`, `f=30`, `H=12`, and steady `0.55 s` guided forwards yield a
19-step bounded refill delay but only 12 horizon steps, so useful guided overlap
is zero and readiness remains closed. Changing `execution_horizon` changes RTC
policy behavior and requires a reviewed shadow run; do not bypass the gate or
raise `max_action_age_sec` merely to make it open.

Compressed-camera workers validate the JPEG envelope and capture native decoder
diagnostics around each decode. A frame with invalid SOI/EOI markers, any native
decoder warning, an exception, or a missing decoded image is discarded before
shared memory and does not advance that camera's sequence. Repeated corruption
therefore reduces the measured input rate and eventually trips the same camera
freshness watchdog instead of feeding a partially decoded image to the policy.

The watchdog never resumes automatically. Restart the inference node, or first
restore every input and then explicitly rearm it:

```bash
ros2 service call /lerobot_inference/rearm_watchdog std_srvs/srv/Trigger '{}'
```

Rearm is rejected until every required input is healthy and has advanced beyond
the sequence observed at the fault. The action queue stays empty until a new
complete observation has been accepted after rearm.

For Pi0.5, startup also fails unless `model.safetensors`, both saved processor
pipelines, and a valid SHA-256 manifest are present. Create the manifest after
copying the checkpoint into its isolated deployment directory:

```bash
python3 scripts/create_checkpoint_manifest.py /absolute/path/to/pretrained_model
```

The loader accepts `SHA256SUMS.expected` or `checkpoint_manifest.sha256`, hashes
every entry before allocating the model, and verifies that strict state-dict
loading actually completed. This guards against LeRobot 0.5.1 returning a
randomly initialized Pi0.5 model after an internal weight-loading error.

## DDS Middleware Selection

Both Fast DDS and CycloneDDS are supported. **CycloneDDS is the default** (faster in our tests).

> ⚠ **Both sides must use the same RMW** — mixing Fast DDS and CycloneDDS = zero discovery (no error, just silence).

| Deployment | `RMW_IMPLEMENTATION` | `CYCLONEDDS_URI` | anvil-loader `.env.config` |
|-----------|----------------------|------------------|---------------------------|
| **Single-PC · CycloneDDS** *(default)* | `rmw_cyclonedds_cpp` | `file://.../single_pc.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=127.0.0.1` |
| Single-PC · Fast DDS | `rmw_fastrtps_cpp` | *(ignored)* | `ENABLE_CYCLONEDDS=false` |
| Two-PC · CycloneDDS | `rmw_cyclonedds_cpp` | `file://.../two_pc_gpu.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=<gpu_pc_ip>` |

All CycloneDDS configs live in `configs/cyclonedds/`. The defaults in `docker-compose.yml` and `.env.example` target single-PC CycloneDDS — override in `.env` to switch modes.

## Deployment Topologies

### Single-PC — inference and workcell on the same machine

```
  Same machine
┌────────────────────────────────────────────────────────────┐
│  anvil-loader (ros2_control)       anvil-embodied-ai       │
│  joint_states (500 Hz)  ◄─────────  inference_node (30 Hz) │
│  cameras (4× 30 Hz)      CycloneDDS  action commands       │
│                           (host net)                       │
└────────────────────────────────────────────────────────────┘
```

Both sides use CycloneDDS on the host network — multicast handles peer discovery automatically. Set in anvil-loader's `.env.config`:
```
ENABLE_CYCLONEDDS=true
CYCLONEDDS_PEER_IP=127.0.0.1
```

### Two-PC — GPU PC separate from the robot PC

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4× 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

Set `CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/two_pc_gpu.xml` and configure peer IPs in both `two_pc_gpu.xml` and anvil-loader's `.env.config`. See the [full documentation](https://docs.anvil.bot/) for network setup.

---

[← Back to README](../README.md)
