from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

__all__ = [
    "fp8_blockwise_grouped_mm",
    "prepare_fp8_blockwise_grouped_mm_plan",
    "_to_mxfp8_then_scaled_grouped_mm",
    "_to_fp8_rowwise_then_scaled_grouped_mm",
]


def __getattr__(name: str):
    if name in (
        "fp8_blockwise_grouped_mm",
        "prepare_fp8_blockwise_grouped_mm_plan",
    ):
        from torchao.prototype.moe_training.blockwise_fp8 import (
            fp8_blockwise_grouped_mm,
            prepare_fp8_blockwise_grouped_mm_plan,
        )

        return {
            "fp8_blockwise_grouped_mm": fp8_blockwise_grouped_mm,
            "prepare_fp8_blockwise_grouped_mm_plan": (
                prepare_fp8_blockwise_grouped_mm_plan
            ),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
