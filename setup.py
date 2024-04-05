# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime

from setuptools import find_packages, setup

current_date = datetime.now().strftime("%Y.%m.%d")


def read_requirements(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()


# Determine the package name based on the presence of an environment variable
package_name = "torchao-nightly" if os.environ.get("TORCHAO_NIGHTLY") else "torchao"

# Version is year.month.date if using nightlies
version = current_date if package_name == "torchao-nightly" else "0.1"


setup(
    name=package_name,
    version=version,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "torchao.kernel.configs": ["*.pkl"],
    },
    install_requires=read_requirements("requirements.txt"),
    description="Package for applying ao techniques to GPU models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/ao",
)
