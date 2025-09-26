#!/bin/bash

conda create -yn xpu_ao_ci python=3.10
source activate xpu_ao_ci

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCCACHE_DISABLE=1

python3 -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchao && python3 setup.py install && cd ..

python3 -c "import torch; import torchao; print(f'Torch version: {torch.__version__}')"

pip install pytest expecttest parameterized accelerate hf_transfer 'modelscope!=1.15.0'

pytest -v -s torchao/test/quantization/quantize_/workflows/int4/test_int4_plain_int32_tensor.py 
