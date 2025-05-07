# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# For llama3.1-8B

python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization float8dq-row --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization float8dq-tensor --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int4wo-32 --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int8wo --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int8dq --tasks wikitext


# For llama3.2-3B

python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization float8dq-row --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization float8dq-tensor --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int4wo-32 --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int8wo --tasks wikitext
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int8dq --tasks wikitext
