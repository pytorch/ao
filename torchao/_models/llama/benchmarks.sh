export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder


# export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
# # in readme
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt

# export MODEL_REPO=meta-llama/Meta-Llama-3-8B
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
# # in readme
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --kv_cache_quantization
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --max_new_tokens 2048
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --kv_cache_quantization --max_new_tokens 2048

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 1024
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 1024 --kv_cache_quantization
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 2048
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --max_new_tokens 2048 --kv_cache_quantization
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 4096
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --max_new_tokens 4096 --kv_cache_quantization

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 8192
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 8192 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 16384
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 16384 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 32768
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 32768 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 65536
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 65536 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 131072
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --max_new_tokens 131072 --kv_cache_quantization
