# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

check_torch() {
  if ! pip show torch > /dev/null 2>&1; then
    echo "Error: torch package is NOT installed. please install with `pip install torch`" >&2
    exit 1
  fi
}

check_vllm() {
  # Check if VLLM_DIR is set
  if [ -z "$VLLM_DIR" ]; then
    echo "Error: VLLM_DIR environment variable is not set. Please set it before running this script."
    exit 1
  fi
  if ! pip show vllm > /dev/null 2>&1; then
    echo "Error: vllm package is NOT installed. please install from source: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-python-only-build-without-compilation" >&2
    exit 1
  fi
}

check_lm_eval() {
  if ! pip show lm_eval > /dev/null 2>&1; then
    echo "Error: lm_eval package is NOT installed. please install with `pip install lm_eval`" >&2
    exit 1
  fi
}
