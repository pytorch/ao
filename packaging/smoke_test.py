#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import torchao


def main():
    """
    Run torchao binary smoke tests like importing and performing simple ops
    """
    print(dir(torchao))


if __name__ == "__main__":
    main()
