.. _api_quantization:

====================
torchao.quantization
====================

.. currentmodule:: torchao.quantization

Main Quantization APIs
----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    quantize_
    autoquant

Inference APIs for quantize\_
-------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Int4WeightOnlyConfig
    Float8ActivationInt4WeightConfig
    Float8DynamicActivationFloat8WeightConfig
    Float8WeightOnlyConfig
    Float8StaticActivationFloat8WeightConfig
    Int8DynamicActivationInt4WeightConfig
    GemliteUIntXWeightOnlyConfig
    Int8WeightOnlyConfig
    Int8DynamicActivationInt8WeightConfig
    UIntXWeightOnlyConfig
    FPXWeightOnlyConfig

.. currentmodule:: torchao.quantization

Quantization Primitives
-----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    choose_qparams_affine
    choose_qparams_affine_with_min_max
    quantize_affine
    dequantize_affine
    safe_int_mm
    int_scaled_matmul
    MappingType
    ZeroPointDomain
    TorchAODType

..
  TODO: delete these?

Other
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    to_linear_activation_quantized
    swap_linear_with_smooth_fq_linear
    smooth_fq_linear_to_inference
