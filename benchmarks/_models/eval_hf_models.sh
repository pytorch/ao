# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# For llama3.1-8B

python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization float8dq-row
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization float8dq-tensor
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int4wo-32
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int8wo
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization int8dq


# For llama3.2-3B

python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization float8dq-row
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization float8dq-tensor
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int4wo-32
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int8wo
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.2-3B --quantization int8dq
