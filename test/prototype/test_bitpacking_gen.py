import pytest
import torch

from torchao.prototype.dtypes.uintgen import (
    pack_uint2,
    pack_uint3,
    pack_uint4,
    pack_uint5,
    pack_uint6,
    pack_uint7,
    unpack_uint2,
    unpack_uint3,
    unpack_uint4,
    unpack_uint5,
    unpack_uint6,
    unpack_uint7,
)


@pytest.mark.parametrize(
    "pack_fn, unpack_fn, bit_count",
    [
        (pack_uint2, unpack_uint2, 2),
        (pack_uint3, unpack_uint3, 3),
        (pack_uint4, unpack_uint4, 4),
        (pack_uint5, unpack_uint5, 5),
        (pack_uint6, unpack_uint6, 6),
        (pack_uint7, unpack_uint7, 7),
    ],
)
def test_uint_packing(pack_fn, unpack_fn, bit_count):
    x = torch.arange(0, 256, dtype=torch.uint8)
    y = pack_fn(x)
    z = unpack_fn(y)
    k = z.view(-1, 2**bit_count)
    check = torch.arange(0, 2**bit_count, dtype=torch.uint8).repeat(k.size(0), 1)
    assert torch.all(k == check), f"Failed for {bit_count}-bit packing"


if __name__ == "__main__":
    pytest.main(__file__)
