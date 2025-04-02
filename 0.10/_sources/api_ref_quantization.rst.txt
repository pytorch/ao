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

Quantization APIs for quantize_
-------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    int4_weight_only
    int8_weight_only
    int8_dynamic_activation_int4_weight
    int8_dynamic_activation_int8_weight
    uintx_weight_only
    gemlite_uintx_weight_only
    intx_quantization_aware_training
    from_intx_quantization_aware_training
    float8_weight_only
    float8_dynamic_activation_float8_weight
    float8_static_activation_float8_weight
    fpx_weight_only

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
