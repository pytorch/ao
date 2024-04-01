from . import dtypes
from .quantization.quant_api import apply_dynamic_quant
from .quantization.quant_api import apply_weight_only_int8_quant

__all__ = [
        "dtypes",
        "apply_dynamic_quant",
]
