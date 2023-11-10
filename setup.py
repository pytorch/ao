# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name='torchao',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    description='Package for applying ao techniques to GPU models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pytorch-labs/ao',
)
