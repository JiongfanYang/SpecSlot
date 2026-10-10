#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"
# Import the repository from its working directory.
unset PYTHONHOME PYTHONPATH

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_DIR {specslot|jit} 256 [BATCH_PER_GPU]" >&2
  echo "       For JiT, set JIT_VARIANT=auto|b8|b16." >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_DIR=$2
MODEL=$3
IMAGE_SIZE=$4
if [[ "${IMAGE_SIZE}" != "256" ]]; then
  echo "[ERROR] This T2I launcher supports 256-pixel images only." >&2
  exit 2
fi
if [[ $# -eq 5 ]]; then
  BATCH_PER_GPU=$5
else
  BATCH_PER_GPU=64
fi
JIT_VARIANT=${JIT_VARIANT:-auto}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29521}
EPOCHS=${EPOCHS:-400}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_EVERY=${LOG_EVERY:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}
EVAL_PROMPT_FILE=${EVAL_PROMPT_FILE:-}
EVAL_ASSETS_DIR=${EVAL_ASSETS_DIR:-}
EVAL_REFERENCE_DIR=${EVAL_REFERENCE_DIR:-}
EVAL_STEPS=${EVAL_STEPS:-50}
if [[ "${MODEL}" == "specslot" ]]; then
  EVAL_CFG=${EVAL_CFG:-4.0}
else
  EVAL_CFG=${EVAL_CFG:-2.9}
fi
EVAL_MAX_IMAGES=${EVAL_MAX_IMAGES:-0}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-4}
EVAL_DPG_MAX_PROMPTS=${EVAL_DPG_MAX_PROMPTS:-0}
EVAL_DPG_SAMPLES=${EVAL_DPG_SAMPLES:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_WORKERS=${EVAL_WORKERS:-0}
EVAL_GEN_EVAL_COMMAND=${EVAL_GEN_EVAL_COMMAND:-}
EVAL_DPG_COMMAND=${EVAL_DPG_COMMAND:-}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-7200}
REQUIRE_EVAL_METRICS=${REQUIRE_EVAL_METRICS:-0}
RESUME=${RESUME:-}
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_SOURCE="explicit PYTHON_BIN"
elif [[ -e "${ROOT_DIR}/.venv" || -L "${ROOT_DIR}/.venv" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  PYTHON_SOURCE="repository .venv"
else
  PYTHON_BIN=python
  PYTHON_SOURCE="current PATH"
fi
LR=${LR:-2e-4}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-0}

if [[ "${MODEL}" == "specslot" ]]; then
  BOTTLENECK=${BOTTLENECK:-1536}
  BOTTLENECK_ARGS=(--bottleneck "${BOTTLENECK}")
else
  BOTTLENECK_ARGS=()
fi

export SPECSLOT_DISABLE_TORCH_COMPILE="${SPECSLOT_DISABLE_TORCH_COMPILE:-0}"
export SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE="${SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE:-1}"
export PYTHONUNBUFFERED=1

COMMAND=("${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_t2i.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_PER_GPU}" \
  --epochs "${EPOCHS}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every "${LOG_EVERY}" \
  --lr "${LR}")
if [[ "${MODEL}" == "jit" ]]; then
  COMMAND+=(--jit_variant "${JIT_VARIANT}")
fi
if (( ${#BOTTLENECK_ARGS[@]} > 0 )); then
  COMMAND+=("${BOTTLENECK_ARGS[@]}")
fi
COMMAND+=(--eval_interval "${EVAL_INTERVAL}" --eval_workers "${EVAL_WORKERS}")
if [[ -n "${EVAL_PROMPT_FILE}" ]]; then
  COMMAND+=(--eval_prompt_file "${EVAL_PROMPT_FILE}")
fi
if [[ -n "${EVAL_ASSETS_DIR}" ]]; then
  COMMAND+=(--eval_assets_dir "${EVAL_ASSETS_DIR}")
fi
if [[ -n "${EVAL_REFERENCE_DIR}" ]]; then
  COMMAND+=(--eval_reference_dir "${EVAL_REFERENCE_DIR}")
fi
if [[ "${EVAL_INTERVAL}" != "0" || -n "${EVAL_PROMPT_FILE}" || -n "${EVAL_ASSETS_DIR}" ]]; then
  COMMAND+=(--eval_steps "${EVAL_STEPS}" --eval_cfg "${EVAL_CFG}" \
    --eval_max_images "${EVAL_MAX_IMAGES}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_num_samples "${EVAL_NUM_SAMPLES}" \
    --eval_dpg_max_prompts "${EVAL_DPG_MAX_PROMPTS}" \
    --eval_dpg_samples "${EVAL_DPG_SAMPLES}" \
    --eval_timeout "${EVAL_TIMEOUT}")
fi
if [[ -n "${EVAL_GEN_EVAL_COMMAND}" ]]; then
  COMMAND+=(--eval_geneval_command "${EVAL_GEN_EVAL_COMMAND}")
fi
if [[ -n "${EVAL_DPG_COMMAND}" ]]; then
  COMMAND+=(--eval_dpg_command "${EVAL_DPG_COMMAND}")
fi
if [[ "${REQUIRE_EVAL_METRICS}" == "1" ]]; then
  COMMAND+=(--eval_require_metrics)
fi
if [[ -n "${RESUME}" ]]; then
  if [[ "${RESUME}" == "auto" ]]; then
    RESUME="${OUTPUT_DIR}/checkpoint-last.pth"
  fi
  if [[ ! -s "${RESUME}" ]]; then
    echo "[ERROR] Resume checkpoint does not exist: ${RESUME}" >&2
    exit 1
  fi
  COMMAND+=(--resume "${RESUME}")
fi
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  COMMAND+=(--activation_checkpointing)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

python_environment_error() {
  echo "[ERROR] Cannot launch T2I with ${PYTHON_SOURCE}: ${PYTHON_BIN}" >&2
  echo "Prepare the repository environment once from ${ROOT_DIR}:" >&2
  echo "  uv sync --locked --extra eval" >&2
  echo "Then unset PYTHON_BIN and rerun this launcher; .venv is selected without activation." >&2
  echo "Alternatively, set PYTHON_BIN to the actual working environment's Python executable." >&2
}

PYTHON_PATH=$(command -v -- "${PYTHON_BIN}" || true)
if [[ -z "${PYTHON_PATH}" || ! -x "${PYTHON_PATH}" || -d "${PYTHON_PATH}" ]]; then
  python_environment_error
  exit 1
fi

# Check the selected interpreter before starting workers.
if ! PYTHON_PATH=$("${PYTHON_PATH}" - <<'PY'
import os
import sys

try:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Expected Python 3.10, found {sys.version.split()[0]}")
    import torch
    import torch.distributed.run
    from torch.utils.tensorboard import SummaryWriter
except Exception as error:
    print(f"[ERROR] {sys.executable}: {type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"[T2I] Python: {sys.executable}; Torch: {torch.__version__}", file=sys.stderr)
print(os.path.abspath(sys.executable))
PY
); then
  python_environment_error
  exit 1
fi

echo "[T2I] Interpreter source: ${PYTHON_SOURCE}" >&2
COMMAND[0]="${PYTHON_PATH}"
# Ensure distributed workers use the same interpreter.
export PYTHON_EXEC="${PYTHON_PATH}"
exec "${COMMAND[@]}"
