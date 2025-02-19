export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder
export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B

# dense baseline
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8192
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt

# bsr benchmarks
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8192 --sparsity bsr-0.9-64
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --sparsity bsr-0.9-32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --sparsity bsr-0.9-64

# Kernel search
BSR_AUTOTUNE=1 python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8192 --sparsity bsr-0.9-64
BSR_AUTOTUNE=1 python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --sparsity bsr-0.9-32
BSR_AUTOTUNE=1 python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --sparsity bsr-0.9-64
