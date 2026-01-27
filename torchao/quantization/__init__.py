import importlib

from torchao.kernel import (
    int_scaled_matmul,
    safe_int_mm,
)

# Lazy imports to avoid CUDA initialization at import time
_lazy_imports = {
    "ALL_AUTOQUANT_CLASS_LIST": ".autoquant",
    "DEFAULT_AUTOQUANT_CLASS_LIST": ".autoquant",
    "DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST": ".autoquant",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST": ".autoquant",
    "DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST": ".autoquant",
    "GEMLITE_INT4_AUTOQUANT_CLASS_LIST": ".autoquant",
    "OTHER_AUTOQUANT_CLASS_LIST": ".autoquant",
    "autoquant": ".autoquant",
}


def __getattr__(name):
    if name in _lazy_imports:
        module_path = _lazy_imports[name]
        module = importlib.import_module(module_path, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
    MultiTensor,
    MultiTensorInputRecorder,
)
from .granularity import (
    Granularity,
    PerAxis,
    PerBlock,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from .linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)
from .linear_quant_modules import (
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightLinear,
    Int8DynActInt4WeightQuantizer,
)
from .observer import (
    AffineQuantizedMinMaxObserver,
    AffineQuantizedObserverBase,
)
from .quant_api import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Float8MMConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FqnToConfig,
    GemliteUIntXWeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    PlainLayout,
    TensorCoreTiledLayout,
    UIntXWeightOnlyConfig,
    fqn_matches_fqn_config,
    intx_quantization_aware_training,
    quantize_,
    swap_conv2d_1x1_to_linear,
)
from .quant_primitives import (
    MappingType,
    TorchAODType,
    ZeroPointDomain,
    choose_qparams_affine,
    choose_qparams_affine_with_min_max,
    dequantize_affine,
    quantize_affine,
)
from .quantize_.workflows import (
    Float8Tensor,
    Int4PlainInt32Tensor,
    Int4PreshuffledTensor,
    Int4Tensor,
    Int4TilePackedTo4dTensor,
    Int8Tensor,
    IntxOpaqueTensor,
    IntxUnpackedToInt8Tensor,
)
from .transform_module import register_quantize_module_handler
from .unified import Quantizer, TwoStepQuantizer
from .utils import (
    compute_error,
)

# TODO: remove after migration of APIs are done
AOPerModuleConfig = ModuleFqnToConfig

__all__ = [
    # top level API - auto
    "autoquant",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "GEMLITE_INT4_AUTOQUANT_CLASS_LIST",
    "DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
    "ALL_AUTOQUANT_CLASS_LIST",
    # top level API - manual
    "quantize_",
    "intx_quantization_aware_training",
    "fqn_matches_fqn_config",
    "swap_conv2d_1x1_to_linear",
    "Int8DynamicActivationInt4WeightConfig",
    "Int8DynamicActivationInt8WeightConfig",
    "Int8DynamicActivationIntxWeightConfig",
    "Int8StaticActivationInt8WeightConfig",
    "Int4WeightOnlyConfig",
    "Float8DynamicActivationInt4WeightConfig",
    "Int8WeightOnlyConfig",
    "Float8WeightOnlyConfig",
    "Float8DynamicActivationFloat8WeightConfig",
    "Float8StaticActivationFloat8WeightConfig",
    "Float8DynamicActivationFloat8SemiSparseWeightConfig",
    "UIntXWeightOnlyConfig",
    "IntxWeightOnlyConfig",
    "GemliteUIntXWeightOnlyConfig",
    "AOPerModuleConfig",
    "FqnToConfig",
    "ModuleFqnToConfig",
    # tensor subclasses
    "Int8Tensor",
    "Int4Tensor",
    "Int4PlainInt32Tensor",
    "Int4PreshuffledTensor",
    "IntxOpaqueTensor",
    "IntxUnpackedToInt8Tensor",
    "Int4TilePackedTo4dTensor",
    "Float8Tensor",
    "compute_error",
    # building blocks
    "to_linear_activation_quantized",
    "to_weight_tensor_with_linear_activation_scale_metadata",
    "AffineQuantizedMinMaxObserver",
    "AffineQuantizedObserverBase",
    # quant primitive ops
    "choose_qparams_affine",
    "choose_qparams_affine_with_min_max",
    "quantize_affine",
    "dequantize_affine",
    # operators/kernels
    "safe_int_mm",
    "int_scaled_matmul",
    # registration of module transforms for quantize_
    "register_quantize_module_handler",
    # dataclasses and types
    "MappingType",
    "ZeroPointDomain",
    "TorchAODType",
    "Granularity",
    "PerTensor",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerToken",
    "LinearActivationQuantizedTensor",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightLinear",
    "TwoStepQuantizer",
    "Quantizer",
    # Layouts for quant_api
    "PlainLayout",
    "TensorCoreTiledLayout",
    "Float8MMConfig",
    # GPTQ
    "Int4WeightOnlyGPTQQuantizer",
    "MultiTensor",
    "MultiTensorInputRecorder",
]
