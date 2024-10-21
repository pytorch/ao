# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="torchao_mps_ops",
    version="1.0",
    ext_modules=[
        CppExtension(
            name="torchao_mps_ops",
            sources=["register.mm"],
            include_dirs=[os.getenv("TORCHAO_ROOT")],
            extra_compile_args=["-DATEN=1"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
