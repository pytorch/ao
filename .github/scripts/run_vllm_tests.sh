#!/bin/bash

set -eux


# vLLM docker image is using CUDA 12.8 and python 3.12
uv pip install --pre fbgemm-gpu-genai --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install -r dev-requirements.txt
# Build and install ao
uv pip install .

pushd vllm
pytest test/integration --verbose -s
popd
