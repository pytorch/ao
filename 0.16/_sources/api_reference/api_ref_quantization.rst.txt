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
    AOBaseConfig
    FqnToConfig

Workflow Configs
----------------

float8 weight configs
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Float8DynamicActivationFloat8WeightConfig
    Float8WeightOnlyConfig

int8 weight configs
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Int8DynamicActivationInt8WeightConfig
    Int8WeightOnlyConfig

int4 weight configs
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Int4WeightOnlyConfig
    Float8DynamicActivationInt4WeightConfig

intx weight configs
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    IntxWeightOnlyConfig
    Int8DynamicActivationIntxWeightConfig

.. currentmodule:: torchao.prototype.mx_formats

mx weight configs (prototype)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    MXDynamicActivationMXWeightConfig

nvfp4 weight configs (prototype)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    NVFP4DynamicActivationNVFP4WeightConfig
    NVFP4WeightOnlyConfig
