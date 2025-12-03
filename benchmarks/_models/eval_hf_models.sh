# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# This script uses the unified llm_eval.py for LLM evaluation
# For full options, run: python -m benchmarks._models.llm_eval --help

# For llama3.1-8B
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization float8dq-row --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization float8dq-tensor --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization float8wo --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization int4wo-128 --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization int8wo --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization int8dq --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization gemlitewo-4-128 --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.1-8B --quantization gemlitewo-8 --tasks wikitext hellaswag


# For llama3.2-3B
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization float8dq-row --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization float8dq-tensor --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization float8wo --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization int4wo-128 --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization int8wo --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization int8dq --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization gemlitewo-4-128 --tasks wikitext hellaswag
python -m benchmarks._models.llm_eval --model_id meta-llama/Llama-3.2-3B --quantization gemlitewo-8 --tasks wikitext hellaswag
