#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints/pixeldit600/checkpoint-600.pth}"
EVAL_SEED="${SEED:-99985}"
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval_pixeldit600}"
FID_STATS="${FID_STATS-${ROOT_DIR}/fid_stats/imagenet256_train.npz}"
if [[ -n "${FID_STATS:-}" ]]; then
  [[ -f "${FID_STATS}" ]] || { echo "FID statistics not found: ${FID_STATS}" >&2; exit 1; }
fi

"${TORCHRUN_BIN}" --nnodes=1 --node_rank=0 --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/main.py" \
  --model PixDiT-XL/16 \
  --prediction_target velocity \
  --img_size 256 \
  --resume_checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_save_root "${EVAL_SAVE_ROOT:-${OUTPUT_DIR}/samples}" \
  --freq_ab_sigma0 3.5 \
  --freq_ab_curve smootherstep \
  --freq_release_start_time 0 \
  --freq_release_time 1 \
  --timeshift 1 \
  --repa_weight 0.5 \
  --repa_layer 8 \
  --repa_encoder dinov2_vitb14 \
  --sampling_method flowdpm \
  --num_sampling_steps 100 \
  --cfg 2.55 \
  --interval_min 0.11 \
  --interval_max 0.975 \
  --gen_bsz "${GEN_BSZ:-128}" \
  --num_images "${NUM_IMAGES:-50000}" \
  --seed "${EVAL_SEED}" \
  --compile \
  --compile_backend inductor \
  --compile_mode max-autotune \
  --no_compile_optimize_ddp \
  --evaluate_gen \
  --fid_stats "${FID_STATS}" \
  "$@"
