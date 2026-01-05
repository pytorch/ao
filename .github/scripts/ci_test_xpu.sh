#!/bin/bash

conda create -yn xpu_ao_ci python=3.10
source activate xpu_ao_ci

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCCACHE_DISABLE=1

python3 -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchao && pip install . --no-build-isolation && cd ..

python3 -c "import torch; import torchao; print(f'Torch version: {torch.__version__}')"

pip install pytest expecttest parameterized accelerate hf_transfer 'modelscope!=1.15.0' transformers tabulate fire

# pytest -v -s torchao/test/quantization/ \
#         torchao/test/dtypes/ \
#         torchao/test/float8/ \
#         torchao/test/integration/test_integration.py \
#         torchao/test/prototype/ \
#         torchao/test/test_ao_models.py
pytest -v -s torchao/test/quantization/test_qat.py::TestQAT::test_fake_quantize_per_channel_group
pytest -v -s torchao/test/prototype/test_quantized_training.py::TestQuantizedTraining::test_int8_weight_only_training_compile_True_device_cpu
pytest -v -s torchao/test/dtypes/test_nf4.py::TestNF4Linear::test_quantize_api_compile_True
