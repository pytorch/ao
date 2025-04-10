# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization fp6 --write_result benchmark_results.txt --precision float16
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant-int4 --write_result benchmark_results.txt

export MODEL_REPO=meta-llama/Meta-Llama-3-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization fp6 --write_result benchmark_results.txt --precision float16
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant-int4 --write_result benchmark_results.txt

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int4wo-64 --write_result benchmark_results.txt
# Runs on H100, float8 is not supported on CUDA arch < 8.9
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8wo --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-tensor --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-wo --write_result benchmark_results.txt

# OTHER BENCHMARKS

# kv cache quantization
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

export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization fp6 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --sparsity semi-structured --precision float16 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization uintx-4-64 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization uintx-2-8 --write_result benchmark_results.txt

export MODEL_REPO=meta-llama/Meta-Llama-3-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision torch.float32 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization autoquant --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization fp6 --write_result benchmark_results.txt --precision float16
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --sparsity semi-structured --precision float16 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization uintx-4-64 --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization uintx-2-8 --write_result benchmark_results.txt

# Different Batch Size Benchmarks
export MODEL_REPO=meta-llama/Meta-Llama-3-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt --batch_size 1
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8dq --write_result benchmark_results.txt --batch_size 128

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt --batch_size 1
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization int8wo --write_result benchmark_results.txt --batch_size 128

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization autoquant --write_result benchmark_results.txt --batch_size 1
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization autoquant --write_result benchmark_results.txt --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization autoquant --write_result benchmark_results.txt --batch_size 128

# TTFT benchmarks
export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8dq --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8wo --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8dq --sparsity semi-structured --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization float8dq --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization float8wo --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int4wo-64 --write_result benchmark_results.txt --prefill_size 8000
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization sparse-marlin --write_result benchmark_results.txt --prefill_size 8000 --precision float16 --sparsity semi-structured

# gemlite benchmarks
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-4-64  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-4-64  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-4-None  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-4-None  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-8-None  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-8-None  --write_result benchmark_results.txt

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-4-64  --write_result benchmark_results.txt  --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-4-64  --write_result benchmark_results.txt  --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-4-None  --write_result benchmark_results.txt  --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-4-None  --write_result benchmark_results.txt --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-8-8-None  --write_result benchmark_results.txt --batch_size 32
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --quantization gemlite-32-8-None  --write_result benchmark_results.txt --batch_size 32

# 2:4 sparse model
export MODEL_REPO=nm-testing/SparseLlama-3-8B-pruned_50.2of4
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --precision float16 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --sparsity semi-structured --precision float16 --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization sparse-marlin --sparsity semi-structured --precision float16 --write_result benchmark_results.txt
