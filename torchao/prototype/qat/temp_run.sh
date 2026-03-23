#!/bin/bash
# End-to-end: SFT (with and without QAT) + NVFP4 eval.
#
# Usage:
#   TASK=gsm8k bash torchao/prototype/qat/temp_run.sh          # default
#   TASK=arc_challenge bash torchao/prototype/qat/temp_run.sh
#
# Checkpoints (under ./checkpoints/qwen3-30b-a3b-${TASK}-*):
#   base        bf16 base model (no training)
#   sft         bf16 SFT checkpoint
#   sft-qat     bf16 SFT+QAT checkpoint
#
# Evals (results in ./logs/${TASK}/):
#   eval_base_bf16.log          base model, bf16 inference
#   eval_sft_bf16.log           SFT model, bf16 inference
#   eval_sft_nvfp4.log          SFT model, NVFP4 inference
#   eval_qat_nvfp4.log          SFT+QAT model, NVFP4 inference

set -euo pipefail

TASK="${TASK:-gsm8k}"
MAX_STEPS="${MAX_STEPS:-100}"

BASE_DIR="./checkpoints/qwen3-30b-a3b-${TASK}-base"
SFT_DIR="./checkpoints/qwen3-30b-a3b-${TASK}-sft"
QAT_DIR="./checkpoints/qwen3-30b-a3b-${TASK}-sft-qat"
LOG_DIR="./logs/${TASK}"

mkdir -p "$LOG_DIR"

echo "Saving base model checkpoint (no training)..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps 0 --output-dir "$BASE_DIR" > "$LOG_DIR/finetune_base.log" 2>&1

echo "Fine-tuning with SFT..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps "$MAX_STEPS" --output-dir "$SFT_DIR" > "$LOG_DIR/finetune_sft.log" 2>&1

echo "Fine-tuning with SFT + QAT..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps "$MAX_STEPS" --qat --output-dir "$QAT_DIR" > "$LOG_DIR/finetune_qat.log" 2>&1

echo "Running all 5 evals in parallel on GPUs 0-4..."
CUDA_VISIBLE_DEVICES=0 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" > "$LOG_DIR/eval_base_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" > "$LOG_DIR/eval_sft_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$QAT_DIR" > "$LOG_DIR/eval_qat_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" --bf16 > "$LOG_DIR/eval_base_bf16.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" --bf16 > "$LOG_DIR/eval_sft_bf16.log" 2>&1 &
wait
