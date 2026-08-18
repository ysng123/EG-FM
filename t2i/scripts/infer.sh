#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a PixelDiT-T2I checkpoint}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/samples/t2i}"
PROMPT_FILE="${PROMPT_FILE:-${REPO_DIR}/t2i/prompts.txt}"

CMD=(
  python "${REPO_DIR}/t2i/inference.py"
  --checkpoint "${CHECKPOINT}"
  --prompt_file "${PROMPT_FILE}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${TEXT_ENCODER_PATH}" ]]; then
  CMD+=(--text_encoder_path "${TEXT_ENCODER_PATH}")
fi

exec "${CMD[@]}" "$@"
