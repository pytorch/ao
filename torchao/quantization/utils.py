# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = [
    "find_multiple",
    "compute_error",
    "_apply_logging_hook",
    "get_model_size_in_bytes",
]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# basic SQNR
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# logger for fqn + op + shape
# note: not safe for any kind of multithreading
_cur_fqn: Optional[str] = None


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _get_logging_hook(fqn):
    # pyre-fixme[53]: Captured variable `fqn` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward_hook(module, input):
        global _cur_fqn
        _cur_fqn = fqn

    return forward_hook


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _apply_logging_hook(model):
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(_get_logging_hook(name))


# collections.defaultdict printing is weird with lambdas, so hand writing for now
_fqn_to_op_to_shape_to_count: Dict[
    Optional[str], Dict[Optional[str], Dict[Optional[str], int]]
] = {}


class LoggingTensorMode(TorchDispatchMode):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        global _cur_fqn
        op_name: str = f"{func.__module__}.{func.__name__}"
        shape_str = ""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape_str += str(list(arg.shape)) + ", "
        if shape_str != "":
            shape_str = shape_str[:-2]

        if _cur_fqn not in _fqn_to_op_to_shape_to_count:
            _fqn_to_op_to_shape_to_count[_cur_fqn] = {}
        if op_name not in _fqn_to_op_to_shape_to_count[_cur_fqn]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name] = {}
        if shape_str not in _fqn_to_op_to_shape_to_count[_cur_fqn][op_name]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] = 0
        _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] += 1

        return rs


# https://discuss.pytorch.org/t/finding-model-size/130275
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_model_size_in_bytes(model):
    s = 0
    for p in model.parameters():
        s += p.nelement() * p.element_size()
    for b in model.buffers():
        s += b.nelement() * b.element_size()
    return s
