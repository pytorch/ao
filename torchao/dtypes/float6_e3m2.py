import torch
from torch import Tensor
from torch.utils._triton import has_triton
from torchao.ops import to_float6_e3m2_packed_cpu, to_float6_e3m2_unpacked_cpu, from_float6_e3m2_packed_cpu, from_float6_e3m2_unpacked_cpu


# some useful constants
FLOAT6_E3M2_MAX = 28.0
FLOAT6_E3M2_SMALLEST_SUBNORMAL = 0.0625


if has_triton():
    import triton
    from triton import language as tl

    # see _to_float6_e3m2_pt() for explanation
    @triton.jit
    def _triton_float32_to_float6_e3m2(x: tl.tensor):
        x = x.to(tl.float32)
        x = x * 2.0 ** (-127 + 3)
        bits = x.to(tl.int32, bitcast=True)

        sign = ((bits >> 31) & 0x1) << 5
        exp_and_man = (bits >> 21) & 0x1F
        result = sign | exp_and_man

        remainder = bits & 0x1F_FFFF
        do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
        result = tl.where(do_round_up, result + 1, result)
        return result.to(tl.uint8)

    @triton.jit
    def _to_float6_e3m2_triton_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        # strided memory read. there will be uncoalesced memory access
        val0 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4, mask))
        val1 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 1, mask))
        val2 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 2, mask))
        val3 = _triton_float32_to_float6_e3m2(tl.load(in_ptr + offsets * 4 + 3, mask))

        # bit packing
        bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
        bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
        bits2 = (val2 << 6) | (val3);      # 2233 3333

        # strided memory write. there will be uncoalesced memory access
        tl.store(out_ptr + offsets * 3, bits0, mask)
        tl.store(out_ptr + offsets * 3 + 1, bits1, mask)
        tl.store(out_ptr + offsets * 3 + 2, bits2, mask)

    def _to_float6_e3m2_triton(tensor: Tensor) -> Tensor:
        out_shape = tensor.shape[:-1] + (tensor.shape[-1] // 4 * 3,)
        output = torch.empty(out_shape, device=tensor.device, dtype=torch.uint8)

        n = tensor.numel()
        grid_size = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"] * 4),)
        _to_float6_e3m2_triton_kernel[grid_size](tensor, output, n, BLOCK_SIZE=256)

        return output

else:
    _to_float6_e3m2_triton = None


# NOTE: This implementation requires FP32 denormal numbers to be handled correctly.
# On CPU, denormal numbers might be flushed to zero for performance gain (FTZ and DAZ flags).
def _to_float6_e3m2_pt(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    tensor = tensor.float()

    # correct exponent bias. this also handles subnormal numbers correctly
    tensor = tensor * 2.0 ** (-127 + 3)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 5
    exp_and_man = (bits >> 21) & 0x1F
    result = sign | exp_and_man

    # round to nearest even
    remainder = bits & 0x1F_FFFF  # truncated mantissa bits
    do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
    result = torch.where(do_round_up, result + 1, result)
    result = result.to(torch.uint8)

    if no_bit_packing:
        return result

    # bit packing
    val0, val1, val2, val3 = result.unflatten(-1, (-1, 4)).unbind(-1)
    bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
    bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
    bits2 = (val2 << 6) | (val3);      # 2233 3333
    return torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)


def to_float6_e3m2(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    """Convert input tensor to FP6. This particular FP6 format has 3 exponent bits and 2 mantissa
    bits. By default, bit packing is performed: every 4 FP6 values are packed as 3 uint8 values
    (4 x 6 bits = 3 x 8 bits).

    Args:
      tensor: Input tensor. The last dimension must be divisible by 4 (unless ``no_bit_packing=False``)
      no_bit_packing: Whether to not perform bit packing. Setting this to ``True`` can be useful for
        observing the bit patterns and debugging.

    Returns:
      :class:`torch.Tensor`: FP6 tensor, stored as uint8 data. If ``no_bit_packing=False``, the last
      dimension of output tensor is 3/4 of that of input tensor.

    Note:
      This FP6 format does not represent +/-inf and NaN. Thus, make sure that input tensor does
      not have +/-inf or NaN values, and no values with magnitude >= 30 (largest number in FP6 is 28.
      All numbers >= 28 and < 30 will be rounded down to 28, while >= 30 will overflow).

      See also :func:`from_float6_e3m2`
    """
    if not no_bit_packing:
        assert tensor.shape[-1] % 4 == 0, "Last dim must be divisible by 4"

    if tensor.is_cpu:
      if no_bit_packing:
        return to_float6_e3m2_unpacked_cpu(tensor)
      
      *leading_dims, last_dim = tensor.shape
      return to_float6_e3m2_packed_cpu(tensor.view(-1, last_dim)).view(*leading_dims, -1)

    # torch.compile() cannot generate fused bit-packing triton kernel,
    # thus we write custom triton kernel for this specific case.
    if tensor.is_cuda and not no_bit_packing and _to_float6_e3m2_triton is not None:
        return _to_float6_e3m2_triton(tensor)

    else:
        return _to_float6_e3m2_pt(tensor, no_bit_packing=no_bit_packing)


# NOTE: This implementation requires FP32 denormal numbers to be handled correctly.
# On CPU, denormal numbers might be flushed to zero for performance gain (FTZ and DAZ flags).
def _pt_float6_e3m2_to_float32(tensor: Tensor) -> Tensor:
    bits = tensor.to(torch.int32)  # bit extension
    sign = bits >> 5 << 31
    exp_and_man = (bits & 0x1F) << 21
    results = sign | exp_and_man

    results = results.view(torch.float32)
    return results * 2.0 ** (127 - 3)  # exponent bias correction


def from_float6_e3m2(tensor: Tensor, no_bit_packing: bool = False, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert an FP6 tensor (created by :func:`to_float6_e3m2`) to FP32.

    Args:
      tensor: FP6 tensor, stored as uint8 data. If ``no_bit_packing=False``, the last dimension must
        be divisible by 3.
      no_bit_packing: whether the input does not have bit packing.
      dtype: returned dtype.

    Returns:
      :class:`torch.Tensor`: FP32 tensor. If ``no_bit_packing=False``, the last dimension of output
      tensor is 4/3 of that of input tensor.
    """
    assert tensor.dtype == torch.uint8
    if no_bit_packing:
        if tensor.is_cpu:
          return from_float6_e3m2_unpacked_cpu(tensor, dtype)

        return _pt_float6_e3m2_to_float32(tensor).to(dtype)

    assert tensor.shape[-1] % 3 == 0, "Last dim must be divisible by 3"
    if tensor.is_cpu:
        return from_float6_e3m2_packed_cpu(tensor, dtype)

    bits0, bits1, bits2 = tensor.unflatten(-1, (-1, 3)).unbind(-1)
    val0 = _pt_float6_e3m2_to_float32(bits0 >> 2).to(dtype)
    val1 = _pt_float6_e3m2_to_float32(((bits0 & 0x3) << 4) | (bits1 >> 4)).to(dtype)
    val2 = _pt_float6_e3m2_to_float32(((bits1 & 0xF) << 2) | (bits2 >> 6)).to(dtype)
    val3 = _pt_float6_e3m2_to_float32(bits2 & 0x3F).to(dtype)
    return torch.stack([val0, val1, val2, val3], dim=-1).flatten(-2)
