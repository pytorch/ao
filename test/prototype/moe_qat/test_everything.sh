#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
pytest test/prototype/moe_qat/test_wrapper_tensor.py -s -v
pytest test/prototype/moe_qat/test_moe_qat_config.py -s -v
pytest test/prototype/moe_qat/test_moe_qat_transform.py -s -v
pytest test/prototype/moe_qat/test_training.py -s -v
./test/prototype/moe_qat/test_distributed.sh
fi

echo "all tests successful"
