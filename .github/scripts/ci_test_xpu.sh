#!/bin/bash

python3 -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchao && python3 setup.py install

pip install pytest expecttest parameterized accelerate hf_transfer 'modelscope!=1.15.0'

pytest -v -s test/quantization/quantize_/workflows/int4/test_int4_plain_int32_tensor.py 
