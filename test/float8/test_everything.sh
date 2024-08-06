#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)

pytest test/float8/test_base.py
pytest test/float8/test_compile.py
# pytest test/float8/test_inference_flows.py
pytest test/float8/test_numerics_integration.py

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
./test/float8/test_fsdp.sh
./test/float8/test_fsdp_compile.sh
./test/float8/test_dtensor.sh
pytest test/float8/test_fsdp2/test_fsdp2.py
fi

echo "all tests successful"
