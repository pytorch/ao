# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)

pytest test/float8/test_base.py
pytest test/float8/test_compile.py
pytest test/float8/test_numerics_integration.py

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
./test/float8/test_fsdp.sh
./test/float8/test_fsdp_compile.sh
./test/float8/test_dtensor.sh
python test/float8/test_fsdp2/test_fsdp2.py
fi

echo "all tests successful"
