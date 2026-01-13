Quick Start Guide
-----------------

In this quick start guide, we will explore how to perform basic quantization using torchao.

Follow `torchao installation and compatibility guide <https://github.com/pytorch/ao#-installation>`__ to install torchao and compatible pytorch.

First Quantization Example
==========================

The main entry point for quantization in torchao is the `quantize_ <https://pytorch.org/ao/stable/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API.
This function mutates your model inplace based on the quantization config you provide.
All code in this guide can be found in this `example script <https://github.com/pytorch/ao/blob/main/scripts/quick_start.py>`__.

Setting Up the Model
~~~~~~~~~~~~~~~~~~~~~

First, let's create a simple model:

.. code:: py

  import copy
  import torch

  class ToyLinearModel(torch.nn.Module):
      def __init__(
          self,
          input_dim,
          hidden_dim,
          output_dim,
          dtype,
          device,
          has_bias=False,
      ):
          super().__init__()
          self.dtype = dtype
          self.device = device
          self.linear1 = torch.nn.Linear(
              input_dim, hidden_dim, bias=has_bias, dtype=dtype, device=device
          )
          self.linear2 = torch.nn.Linear(
              hidden_dim, output_dim, bias=has_bias, dtype=dtype, device=device
          )

      # Note: Tiny-GEMM kernel only uses BF16 inputs
      def example_inputs(self, batch_size=1):
          return (
              torch.randn(
                  batch_size,
                  self.linear1.in_features,
                  dtype=self.dtype,
                  device=self.device,
              ),
          )

      def forward(self, x):
          x = self.linear1(x)
          x = self.linear2(x)
          return x

  model = ToyLinearModel(1024, 1024, 1024, device="cpu", dtype=torch.bfloat16).eval()
  # Optional: compile model for faster inference and generation
  model = torch.compile(model, mode="max-autotune", fullgraph=True)
  model_bf16 = copy.deepcopy(model)

W8A8-INT: 8-bit Dynamic Activation and Weight Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic quantization quantizes both weights and activations at runtime.
This provides better accuracy than weight-only while still offering speedup:

.. code:: py

  from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

  quantize_(model_bf16, Int8DynamicActivationInt8WeightConfig())

The model structure remains unchanged - only weight tensors are quantized. You can verify quantization by checking the weight tensor type:

  >>> print(type(model_bf16.linear1.weight).__name__)
  'Int8Tensor'

For real model, you can use

.. code:: py

  from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

  # Quantize a model with W8A8
  model_w8a8 = AutoModelForCausalLM.from_pretrained(
      meta-llama/Llama-3.1-8B,
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
  quantize_(model_w8a8, Int8DynamicActivationInt8WeightConfig())

  # Save the quantized model
  model_w8a8.save_pretrained("./Llama-3.1-8B-int8")

Both weights and activations are quantized to int8, providing better accuracy
than weight-only quantization while still reducing model size by ~2x. For comprehensive benchmark results and detailed evaluation workflows on production models,
see the `quantization benchmarks <https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md>`__.

Evaluating on Real Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

For production use, you'll want to quantize real LLMs from HuggingFace.
Here's a complete workflow using ``transformers`` and ``lm_eval``:

.. code:: py

  from transformers import AutoModelForCausalLM, AutoTokenizer
  from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

  # Load model from HuggingFace
  model_id = "meta-llama/Llama-3.1-8B"
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  # Quantize the model
  quantize_(model, Int8DynamicActivationInt8WeightConfig())

  # Save the quantized model
  model.save_pretrained("./Llama-3.1-8B-w8a8-int")
  tokenizer.save_pretrained("./Llama-3.1-8B-w8a8-int")

Evaluate accuracy with ``lm_eval``:

.. code:: bash

  lm_eval --model hf \
    --model_args "pretrained=./llama-3.2-1B-w8a8-int" \
    --tasks wikitext,winogrande \
    --device cuda:0 \
    --batch_size 1

Serve and benchmark with ``vllm``:

.. code:: bash

  # Serve the quantized model
  vllm serve ./Llama-3.1-8B-w8a8-int

  # Benchmark throughput
  vllm bench throughput \
    --model ./Llama-3.1-8B-w8a8-int \
    --num-prompts 32 \
    --input-len 4096 \
    --output-len 32


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
