#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/skim3674/Desktop/CoAssembly"
DATASET="$REPO_ROOT/study_logs/study1/referring_expression_user_fixed.csv"
RESULTS="$REPO_ROOT/task_graph/referring_expression_user_benchmark_results.csv"
LEADERBOARD="$REPO_ROOT/task_graph/referring_expression_user_benchmark_leaderboard.csv"
RUN_LOG="$REPO_ROOT/task_graph/referring_expression_user_benchmark.log"

cd "$REPO_ROOT"

python3 - <<'PY'
import pandas
import torch
import transformers
print(f"PyTorch {torch.__version__}; Transformers {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}; {props.total_memory / 1024**3:.2f} GiB total")
PY

# Models run sequentially. --resume preserves completed model/sample pairs and
# continues after interruption. Hugging Face authentication/license acceptance
# is required for gated Gemma 3 and Llama 3.2 checkpoints.
python3 task_graph/evaluate_referring_expression_models.py \
  --dataset "$DATASET" \
  --results "$RESULTS" \
  --leaderboard "$LEADERBOARD" \
  --models \
    gemma3-12b \
    gemma4-e4b \
    gemma3-4b \
    qwen3-vl-8b \
    qwen3-vl-4b \
    qwen3-4b \
    llama32-vision-11b \
  --checkpoint-every 1 \
  --resume \
  2>&1 | tee -a "$RUN_LOG"

echo
echo "Detailed results: $RESULTS"
echo "Leaderboard:      $LEADERBOARD"
echo "Run log:          $RUN_LOG"
