First Quantization Example
==========================

The main entry point for quantization in torchao is the `quantize_ <https://pytorch.org/ao/stable/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API.
This function mutates your model inplace based on the quantization config you provide.
All code in this guide can be found in this `example script <https://github.com/pytorch/ao/blob/main/scripts/quick_start.py>`__.

Setting Up the Model
--------------------

First, let's create a simple model:

.. code:: py

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


    model = ToyLinearModel(
        1024, 1024, 1024, device="cuda", dtype=torch.bfloat16
    ).eval()
    model_w16a16 = copy.deepcopy(model)
    model_w8a8 = copy.deepcopy(model)  # We will quantize in next chapter!

W8A8-INT: 8-bit Dynamic Activation and Weight Quantization
----------------------------------------------------------

Dynamic quantization quantizes both weights and activations at runtime.
This provides better accuracy than weight-only while still offering speedup:

.. code:: py

    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    quantize_(model_w8a8, Int8DynamicActivationInt8WeightConfig())

The model structure remains unchanged - only weight tensors are quantized. You can verify quantization by checking the weight tensor type:

  >>> print(type(model_w8a8.linear1.weight).__name__)
  'Int8Tensor'

The quantized model is now ready to use! Note that the quantization
logic is inserted through tensor subclasses, so there is no change
to the overall model structure; only the weights tensors are updated.

Model Size Comparison
~~~~~~~~~~~~~~~~~~~~~

The int8 quantized model achieves approximately 2x size reduction compared to the original bfloat16 model.
You can verify this by saving both models to disk and comparing file sizes:

.. code:: py

    import os

    # Save models
    torch.save(model_w16a16.state_dict(), "model_w16a16.pth")
    torch.save(model_w8a8.state_dict(), "model_w8a8.pth")

    # Compare file sizes
    original_size = os.path.getsize("model_w16a16.pth") / 1024**2
    quantized_size = os.path.getsize("model_w8a8.pth") / 1024**2
    print(
        f"Size reduction: {original_size / quantized_size:.2f}x ({original_size:.2f}MB -> {quantized_size:.2f}MB)"
    )

Output::

  Size reduction: 2.00x (4.00MB -> 2.00MB)

Speedup Comparison
~~~~~~~~~~~~~~~~~~

Let's demonstrate that not only is the quantized model smaller, but it is also faster.

.. code:: py

    import time

    # Optional: compile model for faster inference and generation
    model_w16a16 = torch.compile(model, mode="max-autotune", fullgraph=True)
    model_w8a8 = torch.compile(model_w8a8, mode="max-autotune", fullgraph=True)

    # Get example inputs
    example_inputs = model_w8a8.example_inputs(batch_size=128)

    # Warmup
    for _ in range(10):
        _ = model_w8a8(*example_inputs)
        _ = model_w16a16(*example_inputs)
    torch.cuda.synchronize()

    # Throughput: original model
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model_w16a16(*example_inputs)
    torch.cuda.synchronize()
    original_time = time.time() - start

    # Throughput: Quantized (W8A8-INT) model
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model_w8a8(*example_inputs)
    torch.cuda.synchronize()
    quantized_time = time.time() - start

    print(f"Speedup: {original_time / quantized_time:.2f}x")

Output::

    Speedup: 1.03x

.. note::
   The speedup results can vary significantly based on hardware and model. We recommend CUDA-enabled GPUs and models larger than 8B for best performance.

Both weights and activations are quantized to int8, reducing model size by ~2x. Speedup is not enough in small toy model because it requires dynamic overhead. For comprehensive benchmark results and detailed evaluation workflows on production models,
see the `quantization benchmarks <../workflows/inference_quantization.html>`__.


We will address how to evaluate quantized model using `vLLM` and `lm-eval` in `model serving <serving.html>`__ post.

Next Steps
----------

In this tutorial, we learned how to quantize a simple model with
torchao. To learn more about the different workflows supported in torchao,
see our main `README <https://github.com/pytorch/ao/blob/main/README.md>`__.
For a more detailed overview of quantization in torchao, visit
`this page <../developer_notes/quantization_overview.html>`__.

Finally, if you would like to contribute to torchao, don't forget to check
out our `contributor guide <../developer_notes/contributor_guide.html>`__ and our list of
`good first issues <https://github.com/pytorch/ao/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`__ on Github!
