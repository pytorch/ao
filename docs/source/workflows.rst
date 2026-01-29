Workflows
=========

This page provides an overview of the quantization and training workflows available in torchao.

Stable Workflows
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 15 10 15 20

   * - recommended hardware
     - weight
     - activation
     - quantized training
     - QAT
     - PTQ data algorithms
     - quantized inference
   * - H100, B200 GPUs
     - float8 rowwise
     - float8 rowwise
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/float8>`__
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling>`__
   * - Intel BMG GPUs
     - float8 tensor/rowwise
     - float8 tensor/rowwise
     - planned
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling>`__
   * - H100 GPUs
     - int4
     - float8 rowwise
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - planned
     - stable `(link) <https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L1296>`__
   * - A100 GPUs
     - int4
     - bfloat16
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - prototype: `HQQ <https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq/README.md>`__, `AWQ <https://github.com/pytorch/ao/tree/main/torchao/prototype/awq>`__
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization>`__
   * - Intel BMG GPUs
     - int4
     - float16/bfloat16
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - prototype: `AWQ <https://github.com/pytorch/ao/tree/main/torchao/prototype/awq>`__
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization>`__
   * - A100 GPUs
     - int8
     - bfloat16
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w8-int8-weightonly-quantization>`__
   * - A100 GPUs
     - int8
     - int8
     - prototype `(link) <https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training>`__
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization>`__
   * - Intel BMG GPUs
     - int8
     - int8
     - planned
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization>`__
   * - edge
     - intx (1..7)
     - bfloat16
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__
     - not supported
     - stable `(link) <https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L2267>`__

Prototype Workflows
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 15 10 15 20

   * - recommended hardware
     - weight
     - activation
     - quantized training
     - QAT
     - PTQ data algorithms
     - quantized inference
   * - B200, MI350x GPUs
     - mxfp8
     - mxfp8
     - prototype `(dense) <https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-training>`__, `(moe) <https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training>`__
     - not supported
     - not supported
     - prototype `(link) <https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference>`__
   * - B200 GPUs
     - nvfp4
     - nvfp4
     - planned
     - prototype `(link) <https://github.com/pytorch/ao/blob/main/torchao/prototype/qat/nvfp4.py>`__
     - not supported
     - prototype `(link) <https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference>`__
   * - B200, MI350x GPUs
     - mxfp4
     - mxfp4
     - not supported
     - planned
     - planned
     - prototype `(link) <https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference>`__
   * - H100
     - float8 128x128 (blockwise)
     - float8 1x128
     - planned
     - not supported
     - not supported
     - prototype

Other Resources
---------------

* `Quantization-Aware Training (QAT) README.md <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat/README.md>`__
* `Post-Training Quantization (PTQ) README.md <https://github.com/pytorch/ao/tree/main/torchao/quantization/README.md>`__
* `Sparsity README.md <https://github.com/pytorch/ao/tree/main/torchao/sparsity/README.md>`__ - includes different techniques such as 2:4 sparsity and block sparsity
* `The prototype folder <https://github.com/pytorch/ao/tree/main/torchao/prototype>`__ for other prototype features
