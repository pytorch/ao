export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder
export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B

#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8192 --profile baseline_prefill
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8192 --sparsity bsr --profile bsr_prefill
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --profile baseline
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --sparsity bsr --profile bsr_padded_trition
