import torch
from torch import Tensor
from torch.utils._triton import has_triton


if has_triton():
    import triton
    from triton import language as tl

    @triton.jit
    def _triton_fp32_to_fp6(x: tl.tensor):
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
    def _to_fp6_triton_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        # strided memory read. there will be uncoalesced memory access
        val0 = _triton_fp32_to_fp6(tl.load(in_ptr + offsets * 4, mask))
        val1 = _triton_fp32_to_fp6(tl.load(in_ptr + offsets * 4 + 1, mask))
        val2 = _triton_fp32_to_fp6(tl.load(in_ptr + offsets * 4 + 2, mask))
        val3 = _triton_fp32_to_fp6(tl.load(in_ptr + offsets * 4 + 3, mask))

        bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
        bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
        bits2 = (val2 << 6) | (val3);      # 2233 3333

        # strided memory write. there will be uncoalesced memory access
        tl.store(out_ptr + offsets * 3, bits0, mask)
        tl.store(out_ptr + offsets * 3 + 1, bits1, mask)
        tl.store(out_ptr + offsets * 3 + 2, bits2, mask)

    def _to_fp6_triton(tensor: Tensor) -> Tensor:
        out_shape = tensor.shape[:-1] + (tensor.shape[-1] // 4 * 3,)
        output = torch.empty(out_shape, device=tensor.device, dtype=torch.uint8)

        n = tensor.numel()
        grid_size = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"] * 4),)
        _to_fp6_triton_kernel[grid_size](tensor, output, n, BLOCK_SIZE=256)

        return output

else:
    _to_fp6_triton = None


def _to_fp6_pt(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    tensor = tensor.float()
    tensor = tensor * 2.0 ** (-127 + 3)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 5
    exp_and_man = (bits >> 21) & 0x1F
    result = sign | exp_and_man

    remainder = bits & 0x1F_FFFF
    do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
    result = torch.where(do_round_up, result + 1, result)
    result = result.to(torch.uint8)

    if no_bit_packing:
        return result

    val0, val1, val2, val3 = result.unflatten(-1, (-1, 4)).unbind(-1)
    bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
    bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
    bits2 = (val2 << 6) | (val3);      # 2233 3333
    return torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)


def to_fp6(tensor: Tensor, no_bit_packing: bool = False) -> Tensor:
    if not no_bit_packing:
        assert tensor.shape[-1] % 4 == 0, "Last dim must be divisible by 4"

    # torch.compile() cannot generate fused bit-packing triton kernel,
    # thus we write custom triton kernel for this specific case.
    if tensor.is_cuda and not no_bit_packing and _to_fp6_triton is not None:
        return _to_fp6_triton(tensor)

    else:
        return _to_fp6_pt(tensor, no_bit_packing=no_bit_packing)
