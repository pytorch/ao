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
    Float8DynamicActivationFloat8WeightConfig
    Float8WeightOnlyConfig
    Float8StaticActivationFloat8WeightConfig
    Int8DynamicActivationInt4WeightConfig
    GemliteUIntXWeightOnlyConfig
    Int8WeightOnlyConfig
    Int8DynamicActivationInt8WeightConfig
    UIntXWeightOnlyConfig
    FPXWeightOnlyConfig

.. currentmodule:: torchao.quantization.qat

QAT APIs
----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    IntXQuantizationAwareTrainingConfig
    FromIntXQuantizationAwareTrainingConfig
    FakeQuantizeConfig
    Int4WeightOnlyQATQuantizer
    Int8DynActInt4WeightQATQuantizer
    Int4WeightOnlyEmbeddingQATQuantizer
    ComposableQATQuantizer
    initialize_fake_quantizers

.. currentmodule:: torchao.quantization

Quantization Primitives
-----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    choose_qparams_affine
    choose_qparams_affine_with_min_max
    choose_qparams_affine_floatx
    quantize_affine
    quantize_affine_floatx
    dequantize_affine
    dequantize_affine_floatx
    choose_qparams_and_quantize_affine_hqq
    fake_quantize_affine
    fake_quantize_affine_cachemask
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
