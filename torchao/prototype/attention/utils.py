# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib


def _is_fa3_available() -> bool:
    try:
        importlib.import_module("flash_attn_interface")
        return True
    except ModuleNotFoundError:
        return False


def _is_fa4_available() -> bool:
    try:
        importlib.import_module("flash_attn.cute.interface")
        return True
    except ModuleNotFoundError:
        return False
