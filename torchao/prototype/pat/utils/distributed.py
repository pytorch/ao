# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor
from torch import distributed as dist

try:
    from torch.distributed.tensor import DTensor

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


def is_main_process():
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return rank == 0


def is_dtensor(x):
    return HAS_DTENSOR and isinstance(x, DTensor)


class NoopHandle:
    def wait(self):
        pass


def _maybe_async_aggregate(
    handle_buf: List[tuple[Tensor, dist.Work | NoopHandle]], input_tensor: Tensor
) -> None:
    if dist.is_initialized() and not is_dtensor(input_tensor):
        handle = dist.reduce(input_tensor, dst=0, async_op=True)
        handle_buf.append((input_tensor, handle))
    else:
        if is_dtensor(input_tensor):
            input_tensor = input_tensor.full_tensor()
        if is_main_process():
            handle_buf.append((input_tensor, NoopHandle()))


def _sum_async_streams(handle_buf: List[tuple[Tensor, dist.Work | NoopHandle]]) -> int:
    assert isinstance(handle_buf, list), (
        f"Expected a list of async handles but got {type(handle_buf)}"
    )
    output = 0
    for input_tensor, handle in handle_buf:
        handle.wait()
        output += input_tensor.item()
    handle_buf.clear()
    return output
