#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_specslot.sh IMAGENET_ROOT OUTPUT_ROOT RESOLUTION ALLOCATION

RESOLUTION is 256 or 512. ALLOCATION is k2 or k4.
The launcher trains one SpecSlot configuration and then evaluates its final EMA1
checkpoint on 50,000 class-balanced samples.
EOF
}

if [[ $# -ne 4 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$(cd "$1" 2>/dev/null && pwd)" || {
  echo "ImageNet root does not exist: $1" >&2
  exit 2
}
OUT_ROOT="$2"
RESOLUTION="$3"
ALLOCATION="$4"

[[ -d "$DATA/train" && -d "$DATA/val" ]] || {
  echo "ImageNet root must contain train/ and val/: $DATA" >&2
  exit 2
}
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

case "$RESOLUTION/$ALLOCATION" in
  256/k2)  WINDOW=32; STRIDE=16; TOKENS=512;  BOTTLENECK=512;  CFG=5.0;  BATCH=64;  ONLINE_GEN_BSZ=512; EVAL_FREQ=5; SAVE_LAST_FREQ=10; SAVE_PERIODIC_FREQ=20; DEFAULT_NUM_WORKERS=4; TRAIN_RNG_SEED=12345; ACT_ARGS=();;
  256/k4)  WINDOW=32; STRIDE=16; TOKENS=1024; BOTTLENECK=1536; CFG=4.0;  BATCH=64;  ONLINE_GEN_BSZ=512; EVAL_FREQ=5; SAVE_LAST_FREQ=10; SAVE_PERIODIC_FREQ=20; DEFAULT_NUM_WORKERS=4; TRAIN_RNG_SEED=12345; ACT_ARGS=();;
  512/k2)  WINDOW=64; STRIDE=32; TOKENS=512;  BOTTLENECK=3072;  CFG=10.5; BATCH=128; ONLINE_GEN_BSZ=512; EVAL_FREQ=5; SAVE_LAST_FREQ=10; SAVE_PERIODIC_FREQ=20; DEFAULT_NUM_WORKERS=12; TRAIN_RNG_SEED=; ACT_ARGS=(--activation_checkpointing);;
  512/k4)  WINDOW=64; STRIDE=32; TOKENS=1024; BOTTLENECK=4096; CFG=8.25; BATCH=128; ONLINE_GEN_BSZ=512;  EVAL_FREQ=5; SAVE_LAST_FREQ=10; SAVE_PERIODIC_FREQ=20; DEFAULT_NUM_WORKERS=12; TRAIN_RNG_SEED=; ACT_ARGS=(--activation_checkpointing);;
  *)
    echo "Unsupported resolution/allocation: $RESOLUTION/$ALLOCATION" >&2
    exit 2
    ;;
esac

FINAL_GEN_BSZ=512
ONLINE_CFG="$CFG"

WANDB_PROJECT="${WANDB_PROJECT:-specslot}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_NUM_VIS="${WANDB_NUM_VIS:-16}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_NAME="specslot-${RESOLUTION}-w256-${ALLOCATION}-b${BOTTLENECK}-seed0"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Keep workers on this checkout.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SPECSLOT_DISABLE_TORCH_COMPILE="${SPECSLOT_DISABLE_TORCH_COMPILE:-0}"
export SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE="${SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE:-1}"
echo "[SpecSlot] block_compile=$([[ \"$SPECSLOT_DISABLE_TORCH_COMPILE\" == 1 ]] && echo disabled || ([[ \"$SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE\" == 1 ]] && echo enabled || echo ddp-eager)); root=$ROOT"

run_logged() {
  local logfile="$1"
  shift
  "$@" 2>&1 | tee -a "$logfile"
}

COMMON=(
  --model JiT-B/16 --img_size "$RESOLUTION" --class_num 1000
  --data_path "$DATA" --batch_size "$BATCH" --accum_iter 1
  --blr 5e-5 --epochs 120 --warmup_epochs 5
  --lr_schedule constant --weight_decay 0 --ema_decay1 0.9999 --ema_decay2 0.9996
  --P_mean -0.8 --P_std 0.8 --noise_scale 1 --t_eps 0.05
  --label_drop_prob 0.1 --pred_param x --grad_clip 0
  --seed 0 --generation_seed 0
  --num_workers "${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}" --data_loader_context spawn --pin_mem
  --sampling_method heun --num_sampling_steps 50 --sampling_grid legacy_uniform
  --cfg "$ONLINE_CFG" --interval_min 0.1 --interval_max 1.0
  --gen_bsz "$ONLINE_GEN_BSZ" --eval_weights ema1
  --save_last_freq "$SAVE_LAST_FREQ" --save_periodic_freq "$SAVE_PERIODIC_FREQ" --save_completed_epochs
  --eval_freq "$EVAL_FREQ" --log_freq 100
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_num_vis "$WANDB_NUM_VIS"
  --specslot --expected_token_count "$TOKENS" --spec_n_max "$TOKENS"
  --window_size "$WINDOW" --spec_window_stride "$STRIDE"
  --spec_frequency_order radial --spec_fixed_support
  --spec_factorized_position
  --spec_hierarchical --spec_global_depth 6 --spec_hierarchy_mode interleave
  --spec_semantic_blocks 1 --spec_spatialized_input
  --spec_spatialized_bottleneck_dim "$BOTTLENECK"
  --spec_spatialized_activation linear --spec_spatial_input_mixing --spec_hybrid_output
  --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR"
)

if [[ ${#ACT_ARGS[@]} -gt 0 ]]; then
  COMMON+=("${ACT_ARGS[@]}")
fi

if [[ -n "$TRAIN_RNG_SEED" ]]; then
  COMMON+=(--train_rng_seed "$TRAIN_RNG_SEED")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  COMMON+=(--wandb_entity "$WANDB_ENTITY")
fi

run_logged "$LOG_DIR/train.log" env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PYTHONPATH" \
  SPECSLOT_DISABLE_TORCH_COMPILE="$SPECSLOT_DISABLE_TORCH_COMPILE" \
  SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE="$SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE" \
  torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr localhost \
  --master_port "${MASTER_PORT:-29500}" \
  "$ROOT/main_jit.py" "${COMMON[@]}" \
  --online_eval --num_images 10000 --wandb_run_name "$WANDB_RUN_NAME"

run_logged "$LOG_DIR/eval_full.log" env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PYTHONPATH" \
  SPECSLOT_DISABLE_TORCH_COMPILE="$SPECSLOT_DISABLE_TORCH_COMPILE" \
  SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE="$SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE" \
  torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr localhost \
  --master_port "${MASTER_PORT:-29500}" \
  "$ROOT/main_jit.py" "${COMMON[@]}" \
  --evaluate_gen --num_images 50000 --gen_bsz "$FINAL_GEN_BSZ" --cfg "$CFG" \
  --resume_checkpoint "$OUTPUT_DIR/checkpoint-last.pth" \
  --wandb_run_name "${WANDB_RUN_NAME}_eval"

echo "Completed: $OUTPUT_DIR"
