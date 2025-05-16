# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf
python scripts/download.py --repo_id meta-llama/Meta-Llama-3-8B
python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B
python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B-Instruct
python scripts/download.py --repo_id meta-llama/Llama-3.2-3B
python scripts/download.py --repo_id nm-testing/SparseLlama-3-8B-pruned_50.2of4
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-3.2-3B
# neuralmagic doesn't come with tokenizer, so we need to copy it over
mkdir -p checkpoints/nm-testing/SparseLlama-3-8B-pruned_50.2of4/original && cp checkpoints/meta-llama/Meta-Llama-3-8B/original/tokenizer.model checkpoints/nm-testing/SparseLlama-3-8B-pruned_50.2of4/original/tokenizer.model
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/nm-testing/SparseLlama-3-8B-pruned_50.2of4
