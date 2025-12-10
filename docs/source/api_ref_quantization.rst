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

Inference APIs for quantize\_
-------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Int4WeightOnlyConfig
    Float8DynamicActivationInt4WeightConfig
    Float8DynamicActivationFloat8WeightConfig
    Float8WeightOnlyConfig
    Int8DynamicActivationInt4WeightConfig
    Int8WeightOnlyConfig
    Int8DynamicActivationInt8WeightConfig

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
    TorchAODType
