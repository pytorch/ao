from torchao.quantization.linear_quant_modules import (
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightLinear,
    Int8DynActInt4WeightQuantizer,
    WeightOnlyInt4Linear,
    # for BC
    _replace_linear_8da4w,  # noqa: F401
    _replace_linear_int4,  # noqa: F401
)

from .GPTQ import Int4WeightOnlyGPTQQuantizer, MultiTensor, MultiTensorInputRecorder

__all__ = [
    "Int4WeightOnlyGPTQQuantizer",
    "MultiTensorInputRecorder",
    "MultiTensor",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "WeightOnlyInt4Linear",
    "Int8DynActInt4WeightLinear",
]
