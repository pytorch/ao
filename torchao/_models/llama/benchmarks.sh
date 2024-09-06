export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder


export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
# in readme
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant-int4 --write_result benchmark_results.txt

# auto-round w/ quant_lm_head
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization  autoround
# auto-round w/o quant_lm_head
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization  autoround-cuda-0


export MODEL_REPO=meta-llama/Meta-Llama-3-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
# in readme
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant-int4 --write_result benchmark_results.txt
# sparse marlin (NOTE: float16)
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --precision float16 --write_result benchmark_results.txt
# auto-round w/ quant_lm_head
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization  autoround
# auto-round w/o quant_lm_head
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization  autoround-cuda-0



export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 8192
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 8192 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 8192 --kv_cache_quantization --linear_causal_mask
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 16384
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 16384 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 16384 --kv_cache_quantization --linear_causal_mask
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 32768
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 32768 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 32768 --kv_cache_quantization --linear_causal_mask
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 65536
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 65536 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 65536 --kv_cache_quantization --linear_causal_mask
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 131072
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 131072 --kv_cache_quantization
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --cache_size 131072 --kv_cache_quantization --linear_causal_mask
