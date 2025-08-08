Quick Start Guide
-----------------

In this quick start guide, we will explore how to perform basic quantization using torchao.
First, install the latest stable torchao release::

  pip install torchao

If you prefer to use the nightly release, you can install torchao using the following
command instead::

  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121

torchao is compatible with the latest 3 major versions of PyTorch, which you will also
need to install (`detailed instructions <https://pytorch.org/get-started/locally/>`__)::

  pip install torch


First Quantization Example
==========================

The main entry point for quantization in torchao is the `quantize_ <https://pytorch.org/ao/stable/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API.
This function mutates your model inplace to insert the custom quantization logic based
on what the user configures. All code in this guide can be found in this `example script <https://github.com/pytorch/ao/blob/main/scripts/quick_start.py>`__.
First, let's set up our toy model:

.. code:: py

  import copy
  import torch

  class ToyLinearModel(torch.nn.Module):
      def __init__(self, m: int, n: int, k: int):
          super().__init__()
          self.linear1 = torch.nn.Linear(m, n, bias=False)
          self.linear2 = torch.nn.Linear(n, k, bias=False)

      def forward(self, x):
          x = self.linear1(x)
          x = self.linear2(x)
          return x

  model = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to("cuda")

  # Optional: compile model for faster inference and generation
  model = torch.compile(model, mode="max-autotune", fullgraph=True)
  model_bf16 = copy.deepcopy(model)

Now we call our main quantization API to quantize the linear weights
in the model to int4 inplace. More specifically, this applies uint4
weight-only asymmetric per-group quantization, leveraging the
`tinygemm int4mm CUDA kernel <https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/cuda/int4mm.cu#L1097>`__
for efficient mixed dtype matrix multiplication:

.. code:: py

  # torch 2.4+ only
  from torchao.quantization import Int4WeightOnlyConfig, quantize_
  quantize_(model, Int4WeightOnlyConfig(group_size=32))

The quantized model is now ready to use! Note that the quantization
logic is inserted through tensor subclasses, so there is no change
to the overall model structure; only the weights tensors are updated,
but `nn.Linear` modules stay as `nn.Linear` modules:

.. code:: py

  >>> model.linear1
  Linear(in_features=1024, out_features=1024, weight=AffineQuantizedTensor(shape=torch.Size([1024, 1024]), block_size=(1, 32), device=cuda:0, _layout=TensorCoreTiledLayout(inner_k_tiles=8), tensor_impl_dtype=torch.int32, quant_min=0, quant_max=15))

  >>> model.linear2
  Linear(in_features=1024, out_features=1024, weight=AffineQuantizedTensor(shape=torch.Size([1024, 1024]), block_size=(1, 32), device=cuda:0, _layout=TensorCoreTiledLayout(inner_k_tiles=8), tensor_impl_dtype=torch.int32, quant_min=0, quant_max=15))

First, verify that the int4 quantized model is roughly a quarter of
the size of the original bfloat16 model:

.. code:: py

  >>> import os
  >>> torch.save(model, "/tmp/int4_model.pt")
  >>> torch.save(model_bf16, "/tmp/bfloat16_model.pt")
  >>> int4_model_size_mb = os.path.getsize("/tmp/int4_model.pt") / 1024 / 1024
  >>> bfloat16_model_size_mb = os.path.getsize("/tmp/bfloat16_model.pt") / 1024 / 1024

  >>> print("int4 model size: %.2f MB" % int4_model_size_mb)
  int4 model size: 1.25 MB

  >>> print("bfloat16 model size: %.2f MB" % bfloat16_model_size_mb)
  bfloat16 model size: 4.00 MB

Next, we demonstrate that not only is the quantized model smaller,
it is also much faster!

.. code:: py

  from torchao.utils import (
      TORCH_VERSION_AT_LEAST_2_5,
      benchmark_model,
      unwrap_tensor_subclass,
  )

  # Temporary workaround for tensor subclass + torch.compile
  # Only needed for torch version < 2.5
  if not TORCH_VERSION_AT_LEAST_2_5:
      unwrap_tensor_subclass(model)

  num_runs = 100
  torch._dynamo.reset()
  example_inputs = (torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda"),)
  bf16_time = benchmark_model(model_bf16, num_runs, example_inputs)
  int4_time = benchmark_model(model, num_runs, example_inputs)

  print("bf16 mean time: %0.3f ms" % bf16_time)
  print("int4 mean time: %0.3f ms" % int4_time)
  print("speedup: %0.1fx" % (bf16_time / int4_time))

On a single A100 GPU with 80GB memory, this prints::

  bf16 mean time: 30.393 ms
  int4 mean time: 4.410 ms
  speedup: 6.9x

PyTorch 2 Export Quantization
=============================
PyTorch 2 Export Quantization is a full graph quantization workflow mostly for static quantization. It targets hardwares that requires both input and output activation and weight to be quantized and relies of recognizing an operator pattern to make quantization decisions (such as linear - relu). PT2E quantization produces a pattern with quantize and dequantize ops inserted around the operators and during lowering quantized operator patterns will be fused into real quantized ops. Currently there are two typical lowering paths, 1. torch.compile through inductor lowering 2. ExecuTorch through delegation

Here we show an example with X86InductorQuantizer

API Example::

  import torch
  from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
  from torch.export import export
  from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
      X86InductorQuantizer,
      get_default_x86_inductor_quantization_config,
  )

  class M(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(5, 10)

     def forward(self, x):
         return self.linear(x)

  # initialize a floating point model
  float_model = M().eval()

  # define calibration function
  def calibrate(model, data_loader):
      model.eval()
      with torch.no_grad():
          for image, target in data_loader:
              model(image)

  # Step 1. program capture
  m = export(m, *example_inputs).module()
  # we get a model with aten ops

  # Step 2. quantization
  # backend developer will write their own Quantizer and expose methods to allow
  # users to express how they
  # want the model to be quantized
  quantizer = X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

  # or prepare_qat_pt2e for Quantization Aware Training
  m = prepare_pt2e(m, quantizer)

  # run calibration
  # calibrate(m, sample_inference_data)
  m = convert_pt2e(m)

  # Step 3. lowering
  # lower to target backend

  # Optional: using the C++ wrapper instead of default Python wrapper
  import torch._inductor.config as config
  config.cpp_wrapper = True

  with torch.no_grad():
      optimized_model = torch.compile(converted_model)

      # Running some benchmark
      optimized_model(*example_inputs)


Please follow these tutorials to get started on PyTorch 2 Export Quantization:

Modeling Users:

- `PyTorch 2 Export Post Training Quantization <tutorials_source/pt2e_quant_ptq.html>`__
- `PyTorch 2 Export Quantization Aware Training <tutorials_source/pt2e_quant_qat.html>`__
- `PyTorch 2 Export Post Training Quantization with X86 Backend through Inductor <tutorials_source/pt2e_quant_x86_inductor.html>`__
- `PyTorch 2 Export Post Training Quantization with XPU Backend through Inductor <tutorials_source/pt2e_quant_xpu_inductor.html>`__
- `PyTorch 2 Export Quantization for OpenVINO torch.compile Backend <tutorials_source/pt2e_quant_openvino_inductor.html>`__


Backend Developers (please check out all Modeling Users docs as well):

- `How to Write a Quantizer for PyTorch 2 Export Quantization <tutorials_source/pt2e_quantizer.html>`_


Next Steps
==========

In this quick start guide, we learned how to quantize a simple model with
torchao. To learn more about the different workflows supported in torchao,
see our main `README <https://github.com/pytorch/ao/blob/main/README.md>`__.
For a more detailed overview of quantization in torchao, visit
`this page <quantization_overview.html>`__.

Finally, if you would like to contribute to torchao, don't forget to check
out our `contributor guide <contributor_guide.html>`__ and our list of
`good first issues <https://github.com/pytorch/ao/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`__ on Github!
