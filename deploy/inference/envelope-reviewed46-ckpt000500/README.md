# Reviewed46 checkpoint 000500 inference profile

This directory preserves the guarded deployment used to evaluate the reviewed
46-episode Pi0.5 checkpoint on the two-arm Anvil workcell. It is intentionally
checkpoint- and graph-specific: the manifest, prompt, feature shapes, joint
order, camera aliases, controller endpoints and ROS graph owners are checked
before the policy can run.

The profile does not contain a checkpoint, Hugging Face cache, credentials,
host paths, packet captures, logs or monitor output. Those values and artifacts
remain local and are ignored by Git.

## What is versioned

- A pinned checkpoint manifest and strict preflight.
- Shadow and live-diagnostic policy configurations.
- A Compose stack for inference, monitoring and read-only DDS checks.
- Exact ROS publisher/subscriber ownership gates.
- Shadow action-flow validation and continuous fatal-log supervision.
- Echo, shadow, joint-worker, live-diagnostic and cleanup wrappers.
- A CycloneDDS unicast template with multicast disabled.
- Tests for the authority and runner-evidence contracts.

The live configuration is retained as a reproducible diagnostic profile. It
publishes to the real forward-position controllers, enables joint-state worker
isolation and saturates raw targets to the reviewed URDF limits. It is not a
general production default and must not be copied to a different robot without
reviewing every joint, topic, camera, watchdog and authority assumption.

## Local configuration

Create the untracked runtime file:

```bash
cd deploy/inference/envelope-reviewed46-ckpt000500
cp runtime.env.example runtime.env
$EDITOR runtime.env
```

`MODEL_PATH` must name the checkpoint step directory that contains
`pretrained_model/`. `HF_CACHE` must contain the pinned PaliGemma tokenizer.
Both paths must be absolute Docker bind paths.

The three `DDS_*` values in `runtime.env` are the source of truth for the
inference workstation side of the CycloneDDS configuration. `preflight.sh`
renders the ignored `cyclonedds_two_pc_gpu.xml` from the tracked template and
then verifies the selected interface, local address and peer. The robot-side
CycloneDDS profile must mirror the peer pair and use the same domain. Static
unicast avoids accidental multicast discovery; it is not an authentication or
firewall boundary.

Copy the tracked manifest into the checkpoint once after transfer:

```bash
cp checkpoint_manifest.sha256 \
  "$MODEL_PATH/pretrained_model/checkpoint_manifest.sha256"
```

The preflight rejects missing, extra, changed or incompatible model artifacts
and verifies that the policy and processor contract is Pi0.5, absolute 16-D
actions, three 480x640 RGB cameras, quantile normalization, chunk size 50,
10 denoising steps and the checkpoint prompt:

```text
Pick up the envelope and place it in the target area
```

## Required progression

Run from this directory in an attended terminal or tmux session:

```bash
./preflight.sh
./run_echo.sh
```

Echo mode loads no policy and creates no command publishers. While the real
sensors are flowing, a second terminal can prove that domain-204 traffic is
unicast only:

```bash
./verify_dds_wire.sh 15
```

Stop echo with `Ctrl-C`, then run the normal shadow acceptance path:

```bash
./run_shadow_joint_worker_monitor.sh
```

Shadow publishes only under `/debug/*`. The runner requires zero publishers on
the live controller topics, exact sensor/controller identities, policy
readiness, finite 8-D messages for both arms at 25-35 Hz and no watchdog latch
or model/processor fallback. It continuously supervises the process during the
startup gates and attended run, and stores logs, monitor CSV and host telemetry
under ignored `logs/` and `outputs/` directories.

The other shadow wrappers are diagnostic comparisons:

- `run_shadow.sh`: main-process joint subscriber plus monitor/debug logging.
- `run_shadow_quiet.sh`: removes action logging and monitor publishers.
- `run_shadow_joint_worker.sh`: joint worker without monitor output.

They do not replace the monitored joint-worker acceptance run.

## Live diagnostic

Do not run live until echo and a continuous monitored shadow have passed, the
robot has an operator with a tested E-stop, teleoperation is stopped, the
workspace and cables are clear, and the controller graph has no other command
publisher.

The wrapper has two independent confirmations and performs guarded homing with
the robot's existing reset service before starting inference:

```bash
export LIVE_ROBOT_CONFIRM=RUN_CKPT000500_ON_REAL_ROBOT
./run_live.sh
```

It then requires the operator to type `HOME AND RUN CKPT000500 LIVE` on a real
terminal. Before homing, after homing and after policy startup, the runner
rechecks the complete graph and refuses missing, duplicated or unknown
endpoints. A watchdog trip, process failure, model-loading fallback or authority
change tears down the isolated project.

The live diagnostic still enforces the hard URDF bounds. Its explicit
saturation behavior exists to reproduce the reviewed real-robot experiment; it
must not be interpreted as permission to remove controller or hardware limits.

Use `Ctrl-C` for normal shutdown. If the terminal disconnects, run:

```bash
./stop.sh
```

Only containers carrying this profile's Compose project label are stopped.

## Validation

The repository-level inference tests cover the watchdog, RTC readiness and
alignment, coherent camera/joint shared memory, strict checkpoint loading,
joint mapping and limits, paired command prevalidation, and guarded homing.
This directory additionally provides:

```bash
python -m pytest -q \
  test_dds_authority_contract.py \
  test_runner_evidence.py
bash -n ./*.sh
```

Compose resolution and host/network checks are intentionally part of
`preflight.sh`, because they require the complete target workstation runtime.
