# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
MODEL=vit_b_16
BATCH_SIZE=256

python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_B_16_Weights.IMAGENET1K_V1 --header
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_B_16_Weights.IMAGENET1K_V1 --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_B_16_Weights.IMAGENET1K_V1 --sparsity semi_structured
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_B_16_Weights.IMAGENET1K_V1 --sparsity semi_structured --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.80 --bsr 64 --weights-path checkpoints/$MODEL/sp0.80-ts64.pth
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.80 --bsr 64 --weights-path checkpoints/$MODEL/sp0.80-ts64.pth --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.84 --bsr 64 --weights-path checkpoints/$MODEL/sp0.84-ts64.pth
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.84 --bsr 64 --weights-path checkpoints/$MODEL/sp0.84-ts64.pth --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.90 --bsr 64 --weights-path checkpoints/$MODEL/sp0.90-ts64.pth
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.90 --bsr 64 --weights-path checkpoints/$MODEL/sp0.90-ts64.pth --quantization

MODEL=vit_h_14
BATCH_SIZE=128

python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 --header
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 --sparsity semi_structured
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --weights ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 --sparsity semi_structured --quantization
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.90 --bsr 64 --weights-path checkpoints/$MODEL/sp0.90-ts64.pth
python evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.90 --bsr 64 --weights-path checkpoints/$MODEL/sp0.90-ts64.pth --quantization
