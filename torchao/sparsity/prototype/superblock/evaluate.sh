MODEL=vit_b_16
BATCH_SIZE=256

torchrun --nproc_per_node=8 evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH
torchrun --nproc_per_node=8 evaluate.py --model $MODEL --batch-size $BATCH_SIZE --data-path $IMAGENET_PATH --sparsity bsr --sparsity-linear 0.80 --bsr 64 --sparsity bsr --weights-path checkpoints/sp0.8-ts64.pth
