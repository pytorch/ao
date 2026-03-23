#!/bin/bash
# End-to-end: SFT (with and without QAT) + NVFP4 eval on GSM8K.
#
# Usage:
#   bash torchao/prototype/qat/temp_run.sh
#
# Checkpoints:
#   ./checkpoints/qwen3-30b-a3b-base        bf16 base model (no training)
#   ./checkpoints/qwen3-30b-a3b-sft         bf16 SFT checkpoint
#   ./checkpoints/qwen3-30b-a3b-sft-qat     bf16 SFT+QAT checkpoint
#
# Evals (results in ./logs/):
#   eval_base_bf16.log          base model, bf16 inference
#   eval_sft_bf16.log           SFT model, bf16 inference
#   eval_sft_nvfp4.log          SFT model, NVFP4 inference
#   eval_qat_nvfp4.log          SFT+QAT model, NVFP4 inference

set -euo pipefail

BASE_DIR="./checkpoints/qwen3-30b-a3b-base"
SFT_DIR="./checkpoints/qwen3-30b-a3b-sft"
QAT_DIR="./checkpoints/qwen3-30b-a3b-sft-qat"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

#echo "Saving base model checkpoint (no training)..."
#python torchao/prototype/qat/temp_finetune.py --max-steps 0 --output-dir "$BASE_DIR" > "$LOG_DIR/finetune_base.log" 2>&1
#
#echo "Fine-tuning with SFT..."
#python torchao/prototype/qat/temp_finetune.py --max-steps 200 --output-dir "$SFT_DIR" > "$LOG_DIR/finetune_sft.log" 2>&1
#
#echo "Fine-tuning with SFT + QAT..."
#python torchao/prototype/qat/temp_finetune.py --max-steps 200 --qat --output-dir "$QAT_DIR" > "$LOG_DIR/finetune_qat.log" 2>&1

echo "Evaluating base model (bf16)..."
CUDA_VISIBLE_DEVICES=0 python torchao/prototype/qat/temp_eval.py --checkpoint "$BASE_DIR" --bf16 > "$LOG_DIR/eval_base_bf16.log" 2>&1 &

echo "Evaluating SFT model (bf16)..."
CUDA_VISIBLE_DEVICES=1 python torchao/prototype/qat/temp_eval.py --checkpoint "$SFT_DIR" --bf16 > "$LOG_DIR/eval_sft_bf16.log" 2>&1 &

echo "Evaluating SFT model (NVFP4)..."
CUDA_VISIBLE_DEVICES=2 python torchao/prototype/qat/temp_eval.py --checkpoint "$SFT_DIR" > "$LOG_DIR/eval_sft_nvfp4.log" 2>&1 &

echo "Evaluating SFT + QAT model (NVFP4)..."
CUDA_VISIBLE_DEVICES=3 python torchao/prototype/qat/temp_eval.py --checkpoint "$QAT_DIR" > "$LOG_DIR/eval_qat_nvfp4.log" 2>&1 &

wait
