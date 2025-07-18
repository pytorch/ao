# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e

pytest test/float8/test_base.py --verbose -s
pytest test/float8/test_compile.py --verbose -s
pytest test/float8/test_numerics_integration.py --verbose -s
pytest test/float8/test_auto_filter.py --verbose -s

echo "all float8 single gpu tests successful"
