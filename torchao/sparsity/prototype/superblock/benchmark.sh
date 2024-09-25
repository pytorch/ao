
python benchmark.py --model vit_h_14 --batch-size 256

python benchmark.py --model vit_h_14 --batch-size 256 --sparsity-linear 0.8 \
  --sp-linear-tile-size 64 \
  --bsr 64 \
  --sparsity bsr

python benchmark.py \
  --model vit_h_14 \
  --batch-size 256 \
  --sparsity semi_structured
