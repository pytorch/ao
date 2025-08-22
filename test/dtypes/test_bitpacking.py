# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch.utils._triton import has_triton

from torchao.dtypes.uintx.bitpacking import pack, pack_cpu, unpack, unpack_cpu

from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()


bit_widths = (1, 2, 3, 4, 5, 6, 7)
dimensions = (0, -1, 1)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset()  # reset cache between tests


@pytest.mark.parametrize("bit_width", bit_widths)
@pytest.mark.parametrize("dim", dimensions)
def test_CPU(bit_width, dim):
    test_tensor = torch.randint(
        0, 2**bit_width, (32, 32, 32), dtype=torch.uint8, device="cpu"
    )
    packed = pack_cpu(test_tensor, bit_width, dim=dim)
    unpacked = unpack_cpu(packed, bit_width, dim=dim)
    assert unpacked.allclose(test_tensor)



@pytest.mark.parametrize("bit_width", bit_widths)
@pytest.mark.parametrize("dim", dimensions)
def test_GPU(bit_width, dim):
    test_tensor = torch.randint(0, 2**bit_width, (32, 32, 32), dtype=torch.uint8).to(_DEVICE)
    packed = pack(test_tensor, bit_width, dim=dim)
    unpacked = unpack(packed, bit_width, dim=dim)
    assert unpacked.allclose(test_tensor)



@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("bit_width", bit_widths)
@pytest.mark.parametrize("dim", dimensions)
def test_compile(bit_width, dim):
    torch._dynamo.config.specialize_int = True
    torch.compile(pack, fullgraph=True)
    torch.compile(unpack, fullgraph=True)
    test_tensor = torch.randint(0, 2**bit_width, (32, 32, 32), dtype=torch.uint8).to(_DEVICE)
    packed = pack(test_tensor, bit_width, dim=dim)
    unpacked = unpack(packed, bit_width, dim=dim)
    assert unpacked.allclose(test_tensor)


# these test cases are for the example pack walk through in the bitpacking.py file

def test_pack_example():
    test_tensor = torch.tensor(
        [0x30, 0x29, 0x17, 0x5, 0x20, 0x16, 0x9, 0x22], dtype=torch.uint8
    ).to(_DEVICE)
    shard_4, shard_2 = pack(test_tensor, 6)
    print(shard_4, shard_2)
    assert torch.tensor([0, 105, 151, 37], dtype=torch.uint8).to(_DEVICE).allclose(shard_4)
    assert torch.tensor([39, 146], dtype=torch.uint8).to(_DEVICE).allclose(shard_2)
    unpacked = unpack([shard_4, shard_2], 6)
    assert unpacked.allclose(test_tensor)


def test_pack_example_CPU():
    test_tensor = torch.tensor(
        [0x30, 0x29, 0x17, 0x5, 0x20, 0x16, 0x9, 0x22], dtype=torch.uint8
    )
    shard_4, shard_2 = pack(test_tensor, 6)
    print(shard_4, shard_2)
    assert torch.tensor([0, 105, 151, 37], dtype=torch.uint8).allclose(shard_4)
    assert torch.tensor([39, 146], dtype=torch.uint8).allclose(shard_2)
    unpacked = unpack([shard_4, shard_2], 6)
    assert unpacked.allclose(test_tensor)
