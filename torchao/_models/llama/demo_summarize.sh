# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# grab moby dick prompt
wget -nc -O moby.txt https://gist.githubusercontent.com/jcaip/f319146bb543e92e23b2c76815b0f29f/raw/31a9cd12b0b59f323eb197c9534953bdac352986/gistfile1.txt

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B-Instruct

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8dq_prefill_wo_decode --prefill_size 8192 --max_new_tokens 256 --num_samples 1 --demo_summarize_prompt moby.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8wo --prefill_size 8192 --max_new_tokens 256 --num_samples 1 --demo_summarize_prompt moby.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --compile_prefill --quantization int8dq --prefill_size 8192 --max_new_tokens 256 --num_samples 1 --demo_summarize_prompt moby.txt
