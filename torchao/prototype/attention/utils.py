# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch


def _is_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


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
