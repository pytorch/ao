#!/bin/bash
# End-to-end: SFT + tailpatch QAT + NVFP4 eval.
#
# Flow:
#   1. Save base model (no training)
#   2. SFT for MAX_STEPS
#   3. Tailpatch: starting from the SFT checkpoint, apply QAT and train
#      for TAILPATCH_STEPS more
#   4. Evaluate all variants
#
# Environment variables:
#   TASK              Eval task name (default: arc_challenge)
#   MAX_STEPS         Number of SFT training steps (default: 100)
#   TAILPATCH_STEPS   Extra QAT steps after SFT (default: 50)
#   QAT_IMPL          QAT implementation: reference_subclass, reference_module_swap, or simple (default: reference_subclass)
#   RUN_TAG           Name for the run directory (auto-generated if unset)
#
# Examples:
#   bash torchao/prototype/qat/temp_run.sh
#   TASK=gsm8k MAX_STEPS=200 TAILPATCH_STEPS=100 bash torchao/prototype/qat/temp_run.sh
#
# Everything is saved under ./logs/${RUN_TAG}/:
#   base/                           bf16 base model (no training)
#   sft/                            bf16 SFT checkpoint
#   qat/                            bf16 SFT + QAT checkpoint
#   eval_base_bf16.log              base model, bf16 inference
#   eval_sft_bf16.log               SFT model, bf16 inference
#   eval_base_nvfp4.log             base model, NVFP4 inference
#   eval_sft_nvfp4.log              SFT model, NVFP4 inference
#   eval_qat_nvfp4.log              QAT model, NVFP4 inference

set -euo pipefail

TASK="${TASK:-arc_challenge}"
MAX_STEPS="${MAX_STEPS:-100}"
TAILPATCH_STEPS="${TAILPATCH_STEPS:-50}"
QAT_IMPL="${QAT_IMPL:-reference_subclass}"
MODEL_NAME="qwen3-30b-a3b"
RUN_TAG="${RUN_TAG:-${MODEL_NAME}_${TASK}_sft${MAX_STEPS}_qat${TAILPATCH_STEPS}_${QAT_IMPL}}"

RUN_DIR="./logs/${RUN_TAG}"
BASE_DIR="${RUN_DIR}/base"
SFT_DIR="${RUN_DIR}/sft"
QAT_DIR="${RUN_DIR}/qat"

mkdir -p "$RUN_DIR"

echo "Saving base model checkpoint (no training)..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps 0 --output-dir "$BASE_DIR" > "$RUN_DIR/finetune_base.log" 2>&1

echo "Fine-tuning with SFT (${MAX_STEPS} steps)..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps "$MAX_STEPS" --output-dir "$SFT_DIR" > "$RUN_DIR/finetune_sft.log" 2>&1

echo "QAT (${TAILPATCH_STEPS} steps of ${QAT_IMPL} resuming from SFT)..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --resume-from "$SFT_DIR" --qat-impl "$QAT_IMPL" --max-steps "$TAILPATCH_STEPS" --output-dir "$QAT_DIR" > "$RUN_DIR/finetune_qat.log" 2>&1

echo "Running evals in parallel..."
CUDA_VISIBLE_DEVICES=0 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" > "$RUN_DIR/eval_base_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" > "$RUN_DIR/eval_sft_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$QAT_DIR" > "$RUN_DIR/eval_qat_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" --bf16 > "$RUN_DIR/eval_base_bf16.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" --bf16 > "$RUN_DIR/eval_sft_bf16.log" 2>&1 &
wait
