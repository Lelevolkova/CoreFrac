#!/usr/bin/env bash
# Run prompt evolution for drill-core crack segmentation (frozen SAM 3 + GigaEvo).
# Validation is pure SAM 3 + Dice (no LLM); an LLM is used only to mutate prompts.
set -euo pipefail

# Repo root (the directory that contains run.py and problems/). Override with GIGAEVO_ROOT.
ROOT="${GIGAEVO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"

# Optional: load secrets (LLM endpoint key, etc.) from a local .env (never committed).
if [[ -f "$ROOT/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$ROOT/.env"; set +a
fi

PYTHON_BIN="${GIGAEVO_PYTHON:-python3}"
REDIS_DB="${REDIS_DB:-3}"

# Mutation LLM: any OpenAI-compatible endpoint (we used a locally hosted Qwen3-235B).
INF_URL="${LOCAL_INF_URL:-http://localhost:8000/v1}"
EVOL_MODEL="${EVOL_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
MAX_MUTANTS="${MAX_MUTANTS:-200}"
export OPENAI_API_KEY="${LOCAL_INF_KEY:-EMPTY}"

# Dataset: the released CoreFrac patch benchmark; evolution scores on the fixed
# 96-patch train subset (champions are re-validated on all 450 separately).
export COREFRAC_ROOT="${COREFRAC_ROOT:-$ROOT/data/corefrac/patches}"
export COREFRAC_SPLIT_MANIFEST="${COREFRAC_SPLIT_MANIFEST:-$ROOT/problems/prompts/corefrac/corefrac_split_manifest.json}"
export COREFRAC_EVAL_MODE="${COREFRAC_EVAL_MODE:-full}"
export COREFRAC_TRAIN_N_SAMPLES="${COREFRAC_TRAIN_N_SAMPLES:-96}"

# SAM 3 (GPU). Concurrent validation workers are spread across visible GPUs.
export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
export SAM3_MODEL_NAME="${SAM3_MODEL_NAME:-MTerryJack/sam3}"
export SAM3_CONFIDENCE="${SAM3_CONFIDENCE:-0.5}"
export SAM3_GPU_SLOT_DIR="${SAM3_GPU_SLOT_DIR:-/tmp/corefrac_gpu_slots}"

EVO_CONCURRENCY="${EVO_CONCURRENCY:-4}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-7200}"
DAG_TIMEOUT="${DAG_TIMEOUT:-14400}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; SAM 3 evolution is GPU-only" >&2
  exit 1
fi

# Warm up SAM 3 weights once, then reset the GPU slot counter.
SAM3_CUDA_INDEX=0 "$PYTHON_BIN" -c "from problems.prompts.corefrac.utils.sam3_tool import _get_runtime; _get_runtime(); print('SAM 3 warmed up')"
rm -f "$SAM3_GPU_SLOT_DIR/counter"

echo "Redis DB: $REDIS_DB | train: $COREFRAC_TRAIN_N_SAMPLES patches | concurrency: $EVO_CONCURRENCY | LLM: $EVOL_MODEL"
exec "$PYTHON_BIN" "$ROOT/run.py" \
  problem.name=prompts/corefrac \
  llm=local_inf \
  "llm_base_url=$INF_URL" \
  "model_name=$EVOL_MODEL" \
  "redis.db=$REDIS_DB" \
  "max_mutants=$MAX_MUTANTS" \
  max_tokens=16384 \
  num_parents=1 \
  "max_in_flight=$EVO_CONCURRENCY" \
  "max_concurrent_dags=$EVO_CONCURRENCY" \
  "stage_timeout=$STAGE_TIMEOUT" \
  "dag_timeout=$DAG_TIMEOUT"
