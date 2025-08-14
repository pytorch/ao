from torchao.kernel import (
    int_scaled_matmul,
    safe_int_mm,
)

from .autoquant import (
    ALL_AUTOQUANT_CLASS_LIST,
    DEFAULT_AUTOQUANT_CLASS_LIST,
    DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,
    DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
    DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST,
    GEMLITE_INT4_AUTOQUANT_CLASS_LIST,
    OTHER_AUTOQUANT_CLASS_LIST,
    autoquant,
)
from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
    MultiTensor,
    MultiTensorInputRecorder,
)
from .granularity import (
    PerAxis,
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
    CutlassInt4PackedLayout,
    FbgemmConfig,
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Float8MMConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FPXWeightOnlyConfig,
    GemliteUIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    PlainLayout,
    TensorCoreTiledLayout,
    UIntXWeightOnlyConfig,
    float8_dynamic_activation_float8_weight,
    float8_static_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
    gemlite_uintx_weight_only,
    int4_dynamic_activation_int4_weight,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    intx_quantization_aware_training,
    quantize_,
    swap_conv2d_1x1_to_linear,
    uintx_weight_only,
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
    Int4MarlinSparseTensor,
    Int4PreshuffledTensor,
    Int4Tensor,
    IntxTilePackedTensor,
    IntxUnpackedTensor,
)
from .smoothquant import (
    SmoothFakeDynamicallyQuantizedLinear,
    SmoothFakeDynQuantMixin,
    get_scale,
    set_smooth_fq_attribute,
    smooth_fq_linear_to_inference,
    swap_linear_with_smooth_fq_linear,
)
from .subclass import *  # noqa: F403
from .transform_module import register_quantize_module_handler
from .unified import Quantizer, TwoStepQuantizer
from .utils import (
    compute_error,
)
from .weight_only import WeightOnlyInt8QuantLinear

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
    "int4_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int8_weight",
    "int8_dynamic_activation_int8_semi_sparse_weight",
    "int4_weight_only",
    "int8_weight_only",
    "intx_quantization_aware_training",
    "float8_weight_only",
    "float8_dynamic_activation_float8_weight",
    "float8_static_activation_float8_weight",
    "uintx_weight_only",
    "fpx_weight_only",
    "gemlite_uintx_weight_only",
    "swap_conv2d_1x1_to_linear",
    "Int4DynamicActivationInt4WeightConfig",
    "Int8DynamicActivationInt4WeightConfig",
    "Int8DynamicActivationInt8WeightConfig",
    "Int8DynamicActivationIntxWeightConfig",
    "Int4WeightOnlyConfig",
    "Float8DynamicActivationInt4WeightConfig",
    "Int8WeightOnlyConfig",
    "Float8WeightOnlyConfig",
    "Float8DynamicActivationFloat8WeightConfig",
    "Float8StaticActivationFloat8WeightConfig",
    "Float8DynamicActivationFloat8SemiSparseWeightConfig",
    "UIntXWeightOnlyConfig",
    "IntxWeightOnlyConfig",
    "FPXWeightOnlyConfig",
    "GemliteUIntXWeightOnlyConfig",
    "AOPerModuleConfig",
    "ModuleFqnToConfig",
    "FbgemmConfig",
    # tensor subclasses
    "Int4Tensor",
    "Int4PreshuffledTensor",
    "Int4MarlinSparseTensor",
    "IntxUnpackedTensor",
    "Float8Tensor",
    "IntxTilePackedTensor",
    "IntxUnpackedTensor",
    # smooth quant - subject to change
    "get_scale",
    "SmoothFakeDynQuantMixin",
    "SmoothFakeDynamicallyQuantizedLinear",
    "swap_linear_with_smooth_fq_linear",
    "smooth_fq_linear_to_inference",
    "set_smooth_fq_attribute",
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
    "PerTensor",
    "PerAxis",
    "PerGroup",
    "PerRow",
    "PerToken",
    "LinearActivationQuantizedTensor",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightLinear",
    "WeightOnlyInt8QuantLinear",
    "TwoStepQuantizer",
    "Quantizer",
    # Layouts for quant_api
    "PlainLayout",
    "TensorCoreTiledLayout",
    "CutlassInt4PackedLayout",
    "Float8MMConfig",
    # GPTQ
    "Int4WeightOnlyGPTQQuantizer",
    "MultiTensor",
    "MultiTensorInputRecorder",
]
