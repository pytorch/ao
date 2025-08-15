# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# quantize and upload checkpoint
python quantize_and_upload.py --model_id Qwen/Qwen3-8B-Base --quant float8
