#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_PATH:?Set DATA_PATH to the ImageNet root containing train/}"

MASTER_PORT="${MASTER_PORT:-29510}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/pixeldit_xl_imagenet256}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Checkpoint not found: ${RESUME_CHECKPOINT}" >&2; exit 1; }
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  GRAD_CLIP="${GRAD_CLIP:-0.5}"
  EPOCHS="${EPOCHS:-601}"
  LR="${LR:-1e-5}"
  MIN_LR="${MIN_LR:-1e-5}"
  LR_SCHEDULE="${LR_SCHEDULE:-constant}"
  WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
  GRAD_CLIP="${GRAD_CLIP:-0}"
  EPOCHS="${EPOCHS:-161}"
  LR="${LR:-1e-4}"
  MIN_LR="${MIN_LR:-1e-4}"
  LR_SCHEDULE="${LR_SCHEDULE:-constant}"
  WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
  RESUME_ARGS=(--resume "${OUTPUT_DIR}")
fi

"${TORCHRUN_BIN}" --nnodes=1 --node_rank=0 --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/main.py" \
  --model PixDiT-XL/16 \
  --prediction_target velocity \
  --img_size 256 \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --blr "${LR}" \
  --min_lr "${MIN_LR}" \
  --lr_schedule "${LR_SCHEDULE}" \
  --weight_decay 0 \
  --grad_clip "${GRAD_CLIP}" \
  --t_sampler lognormal \
  --endpoint_low_prob 0 \
  --endpoint_high_prob 0 \
  --timeshift 1 \
  --freq_ab_sigma0 3.5 \
  --freq_ab_curve smootherstep \
  --freq_release_start_time 0 \
  --freq_release_time 1 \
  --label_drop_prob 0.1 \
  --repa_weight 0.5 \
  --repa_layer 8 \
  --repa_encoder dinov2_vitb14 \
  --sampling_method flowdpm \
  --num_sampling_steps 100 \
  --cfg 2.55 \
  --interval_min 0.11 \
  --interval_max 0.975 \
  --num_workers "${NUM_WORKERS:-12}" \
  --save_last_freq 5 \
  --save_epoch_freq 20 \
  --seed "${SEED:-0}" \
  --compile \
  --compile_backend inductor \
  --compile_mode max-autotune \
  --no_compile_optimize_ddp \
  "${RESUME_ARGS[@]}" \
  "$@"
