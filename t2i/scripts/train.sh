#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAGE="${STAGE:-pretrain512}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the BLIP3o shard directory}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:?Set TEXT_ENCODER_PATH to Gemma-2-2B-IT}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/t2i/${STAGE}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

CMD=(
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
  "${REPO_DIR}/t2i/train.py"
  --stage "${STAGE}"
  --data_path "${DATA_PATH}"
  --text_encoder_path "${TEXT_ENCODER_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --resume "${OUTPUT_DIR}"
)

if [[ -n "${LOAD_FROM:-}" ]]; then
  CMD+=(--load_from "${LOAD_FROM}")
fi
if [[ -n "${DATA_CACHE_DIR:-}" ]]; then
  CMD+=(--data_cache_dir "${DATA_CACHE_DIR}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  CMD+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  CMD+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi
if [[ "${COMPILE:-0}" == "1" ]]; then
  CMD+=(--compile)
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}" "$@"
