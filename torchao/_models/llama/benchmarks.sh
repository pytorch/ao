export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder


export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
# Runs on H100, float8 is not supported on CUDA arch < 8.9
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8wo --write_result benchmark_results.txt
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-tensor --write_result benchmark_results.txt
#python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --sparsity semi-structured --precision float16 --write_result benchmark_results.txt
