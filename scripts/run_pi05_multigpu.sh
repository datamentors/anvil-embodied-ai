#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  DATASET_ROOT=/absolute/path/to/trainready-dataset \
  HF_CACHE=/absolute/path/to/huggingface-cache \
    scripts/run_pi05_multigpu.sh {full_vlm|expert_only}

Optional environment variables:
  TASK_DESCRIPTION=<override unique prompt from TRAIN_READY.json>
  SPLIT_MANIFEST=/absolute/path/to/curated/split_info.json
  ANVIL_TRAIN_SOURCE=<repo root; defaults to this checkout>
  VENV=<venv path; defaults to $ANVIL_TRAIN_SOURCE/.venv>
  RUN_ROOT=<output root; defaults to $ANVIL_TRAIN_SOURCE/runs/pi05>
  CUDA_DEVICES=0,1,2,3
  BATCH_PER_GPU=16
  EPOCHS=5
  CHECKPOINT_FREQ=<steps; defaults to midpoint and final>
  ANVIL_EVAL_MAX_BATCHES=100
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
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"

if [[ -z "${DATASET_ROOT}" || -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi

case "${MODE}" in
  full_vlm)
    TRAIN_EXPERT_ONLY=false
    LEARNING_RATE="${FULL_VLM_LR:-1e-5}"
    DEFAULT_MAIN_PORT=29631
    ;;
  expert_only)
    TRAIN_EXPERT_ONLY=true
    LEARNING_RATE="${EXPERT_ONLY_LR:-3e-5}"
    DEFAULT_MAIN_PORT=29632
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

SOURCE="${ANVIL_TRAIN_SOURCE:-${REPO_ROOT}}"
VENV="${VENV:-${SOURCE}/.venv}"
: "${HF_CACHE:?HF_CACHE must point to the offline Hugging Face cache}"
RUN_ROOT="${RUN_ROOT:-${SOURCE}/runs/pi05}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-${DEFAULT_MAIN_PORT}}"

EPOCHS="${EPOCHS:-5}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-}"
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
IFS=',' read -r -a GPU_IDS <<<"${CUDA_DEVICES}"
WORLD_SIZE="${#GPU_IDS[@]}"
SEED="${SEED:-1000}"
LOG_FREQ="${LOG_FREQ:-500}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

if ((WORLD_SIZE < 1)) || ! [[ "${BATCH_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CUDA_DEVICES and BATCH_PER_GPU must select a non-empty valid batch" >&2
  exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid CUDA device ID: ${gpu_id@Q}" >&2
    exit 2
  fi
done
GLOBAL_BATCH_SIZE="$((BATCH_PER_GPU * WORLD_SIZE))"

test -d "${DATASET_ROOT}"
test -f "${DATASET_ROOT}/meta/info.json"
test -f "${DATASET_ROOT}/meta/stats.json"
test -f "${DATASET_ROOT}/TRAIN_READY.json"
test -x "${VENV}/bin/accelerate"
test -x "${VENV}/bin/anvil-trainer"
test -f "${SOURCE}/scripts/_ddp_shim/sitecustomize.py"
test -d "${HF_CACHE}"
if [[ -n "${SPLIT_MANIFEST}" ]]; then
  test -f "${SPLIT_MANIFEST}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT_MIN=30
export ANVIL_EVAL_MAX_BATCHES="${ANVIL_EVAL_MAX_BATCHES:-100}"
if ! [[ "${ANVIL_EVAL_MAX_BATCHES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ANVIL_EVAL_MAX_BATCHES must be a non-negative integer" >&2
  exit 2
fi
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
if ! metadata_exports="$("${VENV}/bin/python" - \
  "${DATASET_ROOT}" "${EPOCHS}" "${GLOBAL_BATCH_SIZE}" "${SEED}" "${SPLIT_MANIFEST}" <<'PY'
import json
import math
import random
import shlex
import sys
from pathlib import Path

import pyarrow.dataset as pads

root = Path(sys.argv[1])
epochs = int(sys.argv[2])
global_batch_size = int(sys.argv[3])
seed = int(sys.argv[4])
split_manifest_path = Path(sys.argv[5]) if sys.argv[5] else None
marker = json.loads((root / "TRAIN_READY.json").read_text())
facts = marker["facts"]
if facts["action_type"] != "absolute":
    raise RuntimeError("This Pi0.5 recipe requires absolute actions")
lookahead = int(facts["afo_lookahead_frames"])
prompts = facts.get("task_prompts", [])
dataset_task = prompts[0] if len(prompts) == 1 else ""
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

total_episodes = len(episode_ids)
manifest_task = ""
if split_manifest_path is not None:
    split_info = json.loads(split_manifest_path.read_text())
    if split_info.get("total_episodes", total_episodes) != total_episodes:
        raise RuntimeError("Split manifest total_episodes does not match dataset")
    train_episodes = split_info.get("train_episodes", [])
    val_episodes = split_info.get("val_episodes", [])
    test_episodes = split_info.get("test_episodes", [])
    selected = train_episodes + val_episodes + test_episodes
    if not train_episodes:
        raise RuntimeError("Split manifest has no train episodes")
    if len(selected) != len(set(selected)):
        raise RuntimeError("Split manifest episode lists overlap or contain duplicates")
    if any(episode not in lengths for episode in selected):
        raise RuntimeError("Split manifest contains an out-of-range episode")
    n_train = len(train_episodes)
    n_val = len(val_episodes)
    n_test = len(test_episodes)
    manifest_task = split_info.get("task_prompt", "")
else:
    shuffled = episode_ids.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = round(total_episodes * 0.1)
    n_val = round(total_episodes * 0.1)
    n_train = total_episodes - n_val - n_test
    train_episodes = sorted(shuffled[:n_train])
train_frames = sum(lengths[episode] for episode in train_episodes)
steps_per_epoch = math.ceil(train_frames / global_batch_size)
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
print(f"AFO_LOOKAHEAD_FRAMES={lookahead}")
print(f"DATASET_TASK={shlex.quote(dataset_task)}")
print(f"MANIFEST_TASK={shlex.quote(manifest_task)}")
PY
)"; then
  echo "ERROR: could not read the train-ready dataset contract" >&2
  exit 1
fi
eval "${metadata_exports}"

if [[ -n "${CHECKPOINT_FREQ}" ]]; then
  if ! [[ "${CHECKPOINT_FREQ}" =~ ^[1-9][0-9]*$ ]] || ((CHECKPOINT_FREQ > STEPS)); then
    echo "ERROR: CHECKPOINT_FREQ must be a positive integer no greater than ${STEPS}" >&2
    exit 2
  fi
  SAVE_FREQ="${CHECKPOINT_FREQ}"
fi

TASK="${TASK_DESCRIPTION:-${MANIFEST_TASK:-${DATASET_TASK}}}"
if [[ -z "${TASK}" ]]; then
  echo "ERROR: TRAIN_READY.json does not contain one unique prompt; set TASK_DESCRIPTION" >&2
  exit 2
fi

export PYTHONPATH="${SOURCE}/scripts/_ddp_shim:${SOURCE}/packages/anvil_trainer/src:${SOURCE}/packages/anvil_shared/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s\n' \
  "Mode: ${MODE}" \
  "Dataset: ${DATASET_ROOT}" \
  "Split manifest: ${SPLIT_MANIFEST:-random 8:1:1}" \
  "Episodes: total=${TOTAL_EPISODES}, train=${TRAIN_EPISODES}, val=${VAL_EPISODES}, test=${TEST_EPISODES}" \
  "Train frames: ${TRAIN_FRAMES}" \
  "Task: ${TASK}" \
  "AFO lookahead: ${AFO_LOOKAHEAD_FRAMES} frames" \
  "Epochs: ${EPOCHS}" \
  "Steps: ${STEPS_PER_EPOCH}/epoch, ${STEPS} total" \
  "Checkpoint frequency: every ${SAVE_FREQ} steps" \
  "Evaluation: at most ${ANVIL_EVAL_MAX_BATCHES} batches per split, uniformly sampled" \
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

  local dataset_name job_name
  dataset_name="$(printf '%s' "$(basename "${DATASET_ROOT}")" | tr -cs 'A-Za-z0-9_.-' '-')"
  job_name="pi05_${dataset_name}_${kind}_${WORLD_SIZE}gpu_bs${BATCH_PER_GPU}_${RUN_TS}"
  local run_dir="${RUN_ROOT}/runs/${job_name}"
  local output_dir="${run_dir}/output"
  local log_file="${RUN_ROOT}/logs/${job_name}.log"

  test ! -e "${run_dir}"
  mkdir -p "${run_dir}"
  printf '%s\n' \
    "dataset=${DATASET_ROOT}" \
    "split_manifest=${SPLIT_MANIFEST}" \
    "base_checkpoint=lerobot/pi05_base" \
    "mode=${kind}" \
    "train_expert_only=${TRAIN_EXPERT_ONLY}" \
    "freeze_vision_encoder=false" \
    "learning_rate=${LEARNING_RATE}" \
    "batch_size_per_gpu=${BATCH_PER_GPU}" \
    "world_size=${WORLD_SIZE}" \
    "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "steps=${run_steps}" \
    "save_freq=${save_freq}" \
    "eval_max_batches=${ANVIL_EVAL_MAX_BATCHES}" \
    "seed=${SEED}" \
    "started_utc=$(date -u +%FT%TZ)" \
    > "${run_dir}/RUN_CONFIG.txt"

  cd "${SOURCE}"
  local split_arguments=(--split-ratio=8,1,1)
  if [[ -n "${SPLIT_MANIFEST}" ]]; then
    split_arguments=(--split-manifest="${SPLIT_MANIFEST}")
  fi

  "${VENV}/bin/accelerate" launch \
    --num_processes="${WORLD_SIZE}" \
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
    "${split_arguments[@]}" \
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
    --note="pi05_base; ${kind}; ${TRAIN_EPISODES}/${TOTAL_EPISODES} train episodes; AFO N${AFO_LOOKAHEAD_FRAMES}; absolute actions; ${WORLD_SIZE} GPUs; batch ${BATCH_PER_GPU}/rank global ${GLOBAL_BATCH_SIZE}; lr ${LEARNING_RATE}" \
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
  run_training full_vlm_probe 20 20 false false "${PROBE_PORT:-29630}"
  assert_gpus_idle
fi

run_training "${MODE}" "${STEPS}" "${SAVE_FREQ}" true true "${MAIN_PORT}"
