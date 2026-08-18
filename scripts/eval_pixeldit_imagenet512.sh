#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints/pixeldit512/checkpoint-240.pth}"
EVAL_SEED="${SEED:-1}"
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval_pixeldit512}"
FID_STATS="${FID_STATS-${ROOT_DIR}/example/jit/fid_stats/jit_in512_stats.npz}"
FID_ARGS=()
if [[ -n "${FID_STATS}" ]]; then
  [[ -f "${FID_STATS}" ]] || { echo "FID statistics not found: ${FID_STATS}" >&2; exit 1; }
  FID_ARGS=(--fid_stats "${FID_STATS}")
fi

"${TORCHRUN_BIN}" --nnodes=1 --node_rank=0 --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/main.py" \
  --model PixDiT-XL/16 \
  --prediction_target velocity \
  --img_size 512 \
  --resume_checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_save_root "${EVAL_SAVE_ROOT:-${OUTPUT_DIR}/samples}" \
  --freq_ab_sigma0 3.5 \
  --freq_ab_curve smootherstep \
  --freq_release_start_time 0 \
  --freq_release_time 1 \
  --timeshift 3 \
  --repa_weight 0.5 \
  --repa_layer 8 \
  --repa_encoder dinov2_vitb14 \
  --sampling_method flowdpm \
  --num_sampling_steps 100 \
  --cfg 3.0 \
  --interval_min 0.1 \
  --interval_max 0.9 \
  --gen_bsz "${GEN_BSZ:-16}" \
  --num_images "${NUM_IMAGES:-50000}" \
  --seed "${EVAL_SEED}" \
  --compile \
  --compile_backend inductor \
  --compile_mode max-autotune \
  --no_compile_optimize_ddp \
  --evaluate_gen \
  "${FID_ARGS[@]}" \
  "$@"
