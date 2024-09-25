MODEL=vit_h_14
BATCH_SIZE=256

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.8 --sp-linear-tile-size 64 --bsr 64 --sparsity bsr
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity semi_structured
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.8 --sp-linear-tile-size 64 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
