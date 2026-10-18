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
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]] || (( NPROC > ${#visible_devices[@]} )); then
    printf 'NPROC must be positive and fit CUDA_VISIBLE_DEVICES.\n' >&2
    exit 2
fi
launcher=("$PYTHON_BIN")
if (( NPROC > 1 )); then
    launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC")
fi
command=("${launcher[@]}" -m less.train.online_proxy
    --model-path "${MODEL_PATH:-/path/to/qwen3.5-4b}"
    --data-dir "${DATA_DIR:-$PROJECT_ROOT/data}"
    --output-dir "${OUTPUT_DIR:-$PROJECT_ROOT/out/fd/${TASK:-mmlu}/${FILTER:-fd}/s${SEED:-3}}"
    --task "${TASK:-mmlu}" --filter "${FILTER:-fd}" --seed "${SEED:-3}"
    --model-seed "${MODEL_SEED:-0}" --batch-size "${GLOBAL_BATCH_SIZE:-16}"
    --fd-batch-size "${FD_BATCH_SIZE:-2}" --fd-epsilon "${FD_EPSILON:-1e-3}"
    --fd-scheme "${FD_SCHEME:-forward}" --fd-deadzone "${FD_DEADZONE:-0}"
    --save-every-batches "${SAVE_EVERY_BATCHES:-500}")
if [[ -n "${TRAIN_FILE:-}" ]]; then command+=(--train-files "$TRAIN_FILE"); fi
if [[ -n "${FD_CHOICE:-}" ]]; then command+=(--fd-choice "$FD_CHOICE"); fi
if [[ -n "${RANDOM_MATCH_LOG:-}" ]]; then command+=(--random-match-log "$RANDOM_MATCH_LOG"); fi
if [[ -n "${RANDOM_SEED:-}" ]]; then command+=(--random-seed "$RANDOM_SEED"); fi
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
