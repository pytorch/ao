#!/bin/bash
# End-to-end: SFT (with and without QAT) + NVFP4 eval.
#
# Environment variables:
#   TASK          Eval task name (default: arc_challenge)
#   MAX_STEPS     Number of training steps (default: 100)
#   QAT_IMPL      QAT implementation: tensor_subclass or module_swap (default: tensor_subclass)
#   RUN_TAG       Name for the run directory (default: qwen3-30b-a3b_${TASK}_${MAX_STEPS}_${QAT_IMPL})
#
# Examples:
#   bash torchao/prototype/qat/temp_run.sh
#   TASK=gsm8k MAX_STEPS=200 bash torchao/prototype/qat/temp_run.sh
#   RUN_TAG=my_experiment bash torchao/prototype/qat/temp_run.sh
#
# Everything is saved under ./logs/${RUN_TAG}/:
#   base/                       bf16 base model (no training)
#   sft/                        bf16 SFT checkpoint
#   sft-qat/                    bf16 SFT+QAT checkpoint
#   eval_base_bf16.log          base model, bf16 inference
#   eval_sft_bf16.log           SFT model, bf16 inference
#   eval_sft_nvfp4.log          SFT model, NVFP4 inference
#   eval_qat_nvfp4.log          SFT+QAT model, NVFP4 inference

set -euo pipefail

TASK="${TASK:-arc_challenge}"
MAX_STEPS="${MAX_STEPS:-100}"
QAT_IMPL="${QAT_IMPL:-tensor_subclass}"
MODEL_NAME="qwen3-30b-a3b"
RUN_TAG="${RUN_TAG:-${MODEL_NAME}_${TASK}_${MAX_STEPS}_${QAT_IMPL}}"

RUN_DIR="./logs/${RUN_TAG}"
BASE_DIR="${RUN_DIR}/base"
SFT_DIR="${RUN_DIR}/sft"
QAT_DIR="${RUN_DIR}/sft-qat"

mkdir -p "$RUN_DIR"

echo "Saving base model checkpoint (no training)..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps 0 --output-dir "$BASE_DIR" > "$RUN_DIR/finetune_base.log" 2>&1

echo "Fine-tuning with SFT..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps "$MAX_STEPS" --output-dir "$SFT_DIR" > "$RUN_DIR/finetune_sft.log" 2>&1

echo "Fine-tuning with SFT + QAT..."
python torchao/prototype/qat/temp_finetune.py --task "$TASK" --max-steps "$MAX_STEPS" --qat-impl "$QAT_IMPL" --output-dir "$QAT_DIR" > "$RUN_DIR/finetune_qat.log" 2>&1

echo "Running all 5 evals in parallel on GPUs 0-4..."
CUDA_VISIBLE_DEVICES=0 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" > "$RUN_DIR/eval_base_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" > "$RUN_DIR/eval_sft_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$QAT_DIR" > "$RUN_DIR/eval_qat_nvfp4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$BASE_DIR" --bf16 > "$RUN_DIR/eval_base_bf16.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 python torchao/prototype/qat/temp_eval.py --task "$TASK" --checkpoint "$SFT_DIR" --bf16 > "$RUN_DIR/eval_sft_bf16.log" 2>&1 &
wait
