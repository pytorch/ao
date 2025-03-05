
# BSR benchmarks
export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder
export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result bsr_bench_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --sparsity semi-structured --precision float16 --write_result bsr_bench_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --sparsity semi-structured --precision float16 --write_result bsr_bench_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result bsr_bench_results.txt --sparsity bsr-0.8-32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result bsr_bench_results.txt --sparsity bsr-0.8-64
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result bsr_bench_results.txt --sparsity bsr-0.9-32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result bsr_bench_results.txt --sparsity bsr-0.9-64
