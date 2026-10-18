#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${NPROC:-${#visible_devices[@]}}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]] || (( NPROC > ${#visible_devices[@]} )); then exit 2; fi
launcher=("$PYTHON_BIN")
if (( NPROC > 1 )); then launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"); fi
command=("${launcher[@]}" -m evaluation.eval.qwen --protocol "${EVAL_PROTOCOL:-less}"
    --model_path "${EVAL_MODEL_PATH:-${RUN_DIR:?Set RUN_DIR or EVAL_MODEL_PATH}/model}"
    --base_model_path "${MODEL_PATH:-/path/to/qwen3.5-4b}"
    --data_dir "${DATA_DIR:-$PROJECT_ROOT/data}" --task "${TASK:-mmlu}"
    --output_dir "${EVAL_OUTPUT_DIR:-${RUN_DIR:?Set RUN_DIR or EVAL_OUTPUT_DIR}/eval}"
    --seed 42 --max_length 8192 --batch_size "${EVAL_BATCH_SIZE:-2}" --dtype bfloat16)
if [[ -n "${EVAL_MANIFEST:-}" ]]; then command+=(--eval-manifest "$EVAL_MANIFEST"); fi
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
