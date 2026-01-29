Welcome to the torchao Documentation
====================================

PyTorch-Native Training-to-Serving Model Optimization
-----------------------------------------------------

- Pre-train Llama-3.1-70B **1.5x faster** with float8 training
- Recover **67% of quantized accuracy degradation** on Gemma3-4B with QAT
- Quantize Llama-3-8B to int4 for **1.89x faster** inference with **58% less memory**

`torchao <https://github.com/pytorch/ao>`__ is a library for custom data types and optimizations.
Quantize and sparsify weights, gradients, optimizers, and activations for inference and training
using native PyTorch. Please checkout torchao `README <https://github.com/pytorch/ao#torchao-pytorch-architecture-optimization>`__
for an overall introduction to the library and recent highlight and updates.

Quick Start
-----------

First, install TorchAO. We recommend installing the latest stable version:

.. code:: bash

    pip install torchao

Quantize your model weights to int4!

.. code:: python

    import torch
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    if torch.cuda.is_available():
      # quantize on CUDA
      quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq"))
    elif torch.xpu.is_available():
      # quantize on XPU
      quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="plain_int32"))

See our `first quantization example <eager_quantization/first_quantization_example.html>`__ for more details.

Installation
------------

To install the latest stable version:

.. code:: bash

    pip install torchao

Other installation options:

.. code:: bash

    # Nightly
    pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128

    # Different CUDA versions
    pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
    pip install torchao --index-url https://download.pytorch.org/whl/cu129  # CUDA 12.9
    pip install torchao --index-url https://download.pytorch.org/whl/xpu    # XPU
    pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

    # For developers
    # Note: the --no-build-isolation flag is required.
    USE_CUDA=1 pip install -e . --no-build-isolation
    USE_XPU=1 pip install -e . --no-build-isolation
    USE_CPP=0 pip install -e . --no-build-isolation

Please see the `torchao compatibility table <https://github.com/pytorch/ao/issues/2919>`__ for version requirements for dependencies.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   api_reference/index

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Eager Quantization Tutorials

   eager_quantization/index

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Notes

   developer_notes/index

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PT2E Quantization Tutorials

   pt2e_quantization/index
