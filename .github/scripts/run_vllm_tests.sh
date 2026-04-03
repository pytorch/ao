#!/bin/bash

set -eux

# vLLM docker image is using CUDA 12.8 and python 3.12
pip install --pre fbgemm-gpu-genai --index-url https://download.pytorch.org/whl/cu128
pip install -r dev-requirements.txt
# Build and install ao
pip install .

pytest test/integration --verbose -s
