#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  DATASET_ROOT=/absolute/path/to/trainready-dataset \
    scripts/run_pi05_1000plus.sh {full_vlm|expert_only}

Optional environment variables:
  EPOCHS=5
  TASK_DESCRIPTION="Pick up the envelope and place it in the target area"
  ANVIL_TRAIN_SOURCE=/path/to/frozen/training/source
  RUN_ROOT=/path/to/output/root
  PROBE_FULL_VLM=1
  PREFLIGHT_ONLY=0
  SMOKE_TEST=0
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODE="${1:-}"
DATASET_ROOT="${DATASET_ROOT:-}"

if [[ -z "${DATASET_ROOT}" || -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi

case "${MODE}" in
  full_vlm)
    TRAIN_EXPERT_ONLY=false
    LEARNING_RATE=1e-5
    MAIN_PORT=29631
    ;;
  expert_only)
    TRAIN_EXPERT_ONLY=true
    LEARNING_RATE=3e-5
    MAIN_PORT=29632
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

SOURCE="${ANVIL_TRAIN_SOURCE:-/home/datamentors/experiments/envelope-afo30-s04-55-20260811/training/source}"
VENV="${VENV:-/home/datamentors/experiments/envelope-afo30-s04-55-20260811/training/.venv}"
HF_CACHE="${HF_CACHE:-/data/work/hf-cache}"
RUN_ROOT="${RUN_ROOT:-/home/datamentors/experiments/envelope-pi05-1000plus/training}"
TASK="${TASK_DESCRIPTION:-Pick up the envelope and place it in the target area}"

EPOCHS="${EPOCHS:-5}"
BATCH_PER_GPU=16
WORLD_SIZE=4
GLOBAL_BATCH_SIZE=64
SEED=1000
LOG_FREQ=500
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

test -d "${DATASET_ROOT}"
test -f "${DATASET_ROOT}/meta/info.json"
test -f "${DATASET_ROOT}/meta/stats.json"
test -f "${DATASET_ROOT}/TRAIN_READY.json"
test -x "${VENV}/bin/accelerate"
test -x "${VENV}/bin/anvil-trainer"
test -f "${SOURCE}/scripts/_ddp_shim/sitecustomize.py"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT_MIN=30
export ANVIL_EVAL_MAX_BATCHES="${ANVIL_EVAL_MAX_BATCHES:-500}"
export NCCL_DEBUG=WARN

export WANDB_DIR="${RUN_ROOT}/wandb"
export WANDB_CACHE_DIR="${RUN_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${RUN_ROOT}/wandb-config"
export XDG_CACHE_HOME="${RUN_ROOT}/xdg-cache"
export TORCH_HOME="${RUN_ROOT}/torch-cache"

mkdir -p \
  "${RUN_ROOT}/runs" \
  "${RUN_ROOT}/logs" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}"

# Reproduce the trainer's deterministic 8:1:1 episode split and derive the
# number of optimizer steps from the actual train frames and global batch.
eval "$("${VENV}/bin/python" - "${DATASET_ROOT}" "${EPOCHS}" <<'PY'
import math
import random
import sys
from pathlib import Path

import pyarrow.dataset as pads

root = Path(sys.argv[1])
epochs = int(sys.argv[2])
table = pads.dataset(root / "meta" / "episodes", format="parquet").to_table(
    columns=["episode_index", "length"]
)

lengths = {}
for row in table.to_pylist():
    episode = int(row["episode_index"])
    if episode in lengths:
        raise RuntimeError(f"Duplicate episode_index: {episode}")
    lengths[episode] = int(row["length"])

episode_ids = sorted(lengths)
if episode_ids != list(range(len(episode_ids))):
    raise RuntimeError("episode_index must be contiguous from 0 to N-1")

shuffled = episode_ids.copy()
random.Random(1000).shuffle(shuffled)
total_episodes = len(shuffled)
n_test = round(total_episodes * 0.1)
n_val = round(total_episodes * 0.1)
n_train = total_episodes - n_val - n_test
train_episodes = sorted(shuffled[:n_train])
train_frames = sum(lengths[episode] for episode in train_episodes)
steps_per_epoch = math.ceil(train_frames / 64)
steps = steps_per_epoch * epochs
# Keep two resumable checkpoints (midpoint and final). Full-VLM Adam state is
# substantially larger than expert-only state, so per-epoch saves can exhaust
# the training workstation before the second comparison run starts.
save_freq = math.ceil(steps / 2)
warmup_steps = min(1000, max(10, steps // 30))

print(f"TOTAL_EPISODES={total_episodes}")
print(f"TRAIN_EPISODES={n_train}")
print(f"VAL_EPISODES={n_val}")
print(f"TEST_EPISODES={n_test}")
print(f"TRAIN_FRAMES={train_frames}")
print(f"STEPS_PER_EPOCH={steps_per_epoch}")
print(f"STEPS={steps}")
print(f"SAVE_FREQ={save_freq}")
print(f"WARMUP_STEPS={warmup_steps}")
PY
)"

export PYTHONPATH="${SOURCE}/scripts/_ddp_shim:${SOURCE}/packages/anvil_trainer/src:${SOURCE}/packages/anvil_shared/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s\n' \
  "Mode: ${MODE}" \
  "Dataset: ${DATASET_ROOT}" \
  "Episodes: total=${TOTAL_EPISODES}, train=${TRAIN_EPISODES}, val=${VAL_EPISODES}, test=${TEST_EPISODES}" \
  "Train frames: ${TRAIN_FRAMES}" \
  "Epochs: ${EPOCHS}" \
  "Steps: ${STEPS_PER_EPOCH}/epoch, ${STEPS} total" \
  "Batch: ${BATCH_PER_GPU}/GPU x ${WORLD_SIZE} = ${GLOBAL_BATCH_SIZE}" \
  "Learning rate: ${LEARNING_RATE}"

assert_gpus_idle() {
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
    echo "ERROR: GPUs are already in use. Training was not started." >&2
    nvidia-smi >&2
    return 1
  fi
}

assert_disk_space() {
  local required_gib=35
  if [[ "${MODE}" == "full_vlm" ]]; then
    required_gib=75
  fi

  local available_blocks block_size available_gib
  read -r available_blocks block_size < <(stat -f --format='%a %S' "${RUN_ROOT}")
  available_gib="$(( available_blocks * block_size / 1024 / 1024 / 1024 ))"
  if (( available_gib < required_gib )); then
    echo "ERROR: ${available_gib} GiB free under ${RUN_ROOT}; ${MODE} requires at least ${required_gib} GiB." >&2
    return 1
  fi
}

run_training() {
  local kind="$1"
  local run_steps="$2"
  local save_freq="$3"
  local save_checkpoint="$4"
  local wandb_enable="$5"
  local port="$6"
  local run_warmup="${WARMUP_STEPS}"
  local run_log_freq="${LOG_FREQ}"

  if (( run_steps < run_warmup )); then
    run_warmup=$(( run_steps / 10 ))
    (( run_warmup >= 1 )) || run_warmup=1
  fi
  if (( run_steps < run_log_freq )); then
    run_log_freq=4
  fi

  local job_name="pi05_1000plus_${kind}_4gpu_bs16_${RUN_TS}"
  local run_dir="${RUN_ROOT}/runs/${job_name}"
  local output_dir="${run_dir}/output"
  local log_file="${RUN_ROOT}/logs/${job_name}.log"

  test ! -e "${run_dir}"
  mkdir -p "${run_dir}"
  printf '%s\n' \
    "dataset=${DATASET_ROOT}" \
    "base_checkpoint=lerobot/pi05_base" \
    "mode=${kind}" \
    "train_expert_only=${TRAIN_EXPERT_ONLY}" \
    "freeze_vision_encoder=false" \
    "learning_rate=${LEARNING_RATE}" \
    "batch_size_per_gpu=${BATCH_PER_GPU}" \
    "world_size=${WORLD_SIZE}" \
    "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "steps=${run_steps}" \
    "seed=${SEED}" \
    "started_utc=$(date -u +%FT%TZ)" \
    > "${run_dir}/RUN_CONFIG.txt"

  cd "${SOURCE}"
  "${VENV}/bin/accelerate" launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --main_process_port="${port}" \
    "${VENV}/bin/anvil-trainer" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.repo_id=local \
    --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=false \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --policy.optimizer_lr="${LEARNING_RATE}" \
    --policy.scheduler_warmup_steps="${run_warmup}" \
    --policy.scheduler_decay_steps="${run_steps}" \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}' \
    --action-type=absolute \
    --split-ratio=8,1,1 \
    --task-description="${TASK}" \
    --seed="${SEED}" \
    --job_name="${job_name}" \
    --output_dir="${output_dir}" \
    --batch_size="${BATCH_PER_GPU}" \
    --steps="${run_steps}" \
    --save_freq="${save_freq}" \
    --log_freq="${run_log_freq}" \
    --num_workers=4 \
    --eval_freq=0 \
    --save_checkpoint="${save_checkpoint}" \
    --resume=false \
    --wandb.enable="${wandb_enable}" \
    --wandb.mode=offline \
    --note="pi05_base; ${kind}; ${TOTAL_EPISODES} episodes; AFO N10 at 30Hz; absolute actions; 4 GPUs; batch 16/rank global 64; lr ${LEARNING_RATE}" \
    2>&1 | tee "${log_file}"

  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> "${run_dir}/RUN_CONFIG.txt"
  touch "${run_dir}/TRAINING_COMPLETE"
}

assert_gpus_idle
assert_disk_space
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Training preflight passed; no process was started."
  exit 0
fi

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  run_training "${MODE}_smoke" 20 20 false false "${MAIN_PORT}"
  exit 0
fi

if [[ "${MODE}" == "full_vlm" && "${PROBE_FULL_VLM:-1}" == "1" ]]; then
  echo "Running the mandatory 20-step full-VLM memory probe."
  run_training full_vlm_probe 20 20 false false 29630
  assert_gpus_idle
fi

run_training "${MODE}" "${STEPS}" "${SAVE_FREQ}" true true "${MAIN_PORT}"
