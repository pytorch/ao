MODEL=vit_h_14
BATCH_SIZE=256

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --header
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --quantization

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity semi_structured
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity semi_structured --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr --quantization

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params

MODEL=vit_b_16
BATCH_SIZE=256

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --header
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --quantization

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity semi_structured
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity semi_structured --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr --quantization
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr --quantization

python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.80 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.84 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
python benchmark.py --model $MODEL --batch-size $BATCH_SIZE --sparsity-linear 0.90 --bsr 64 --sparsity bsr --quantization  --tune-kernel-params
