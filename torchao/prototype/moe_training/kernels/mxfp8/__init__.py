from torchao.prototype.moe_training.kernels.mxfp8 import quant as _quant
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    fused_pad_token_groups_cuda,  # noqa: F401
    fused_unpad_token_groups_cuda,  # noqa: F401
    mx_block_rearrange_2d_M_groups_cuda,  # noqa: F401
    mxfp8_quantize_2d_1x32_32x1_cutedsl,  # noqa: F401
    mxfp8_quantize_2d_1x32_cutedsl,  # noqa: F401
    mxfp8_quantize_2d_1x32_flydsl,  # noqa: F401
    mxfp8_quantize_2d_32x1_cutedsl,  # noqa: F401
    mxfp8_quantize_2d_32x1_flydsl,  # noqa: F401
    mxfp8_quantize_3d_flydsl,  # noqa: F401
    mxfp8_quantize_cuda_3d,  # noqa: F401
    torch_pad_token_groups,  # noqa: F401
    torch_to_blocked_2d_K_groups,  # noqa: F401
    torch_to_blocked_2d_M_groups,  # noqa: F401
    torch_to_blocked_per_group_3d,  # noqa: F401
    torch_unpad_token_groups,  # noqa: F401
    triton_mx_block_rearrange_2d_K_groups,  # noqa: F401
    triton_mx_block_rearrange_2d_M_groups,  # noqa: F401
    triton_mx_block_rearrange_per_group_3d,  # noqa: F401
)


def __getattr__(name: str):
    # The device flags are computed on first access (see `quant._LAZY_FLAGS`).
    if name in _quant._LAZY_FLAGS:
        return getattr(_quant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
