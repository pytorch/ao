Static Quantization
--------------------

Static quantization refers to using a fixed quantization range for all inputs during inference. Unlike dynamic quantization, which recomputes quantization ranges for each new input batch, static quantization typically results in more efficient computation, potentially at the cost of lower quantized accuracy since we cannot adapt to changes in the input distribution on-the-fly.

In static quantization, this fixed quantization range is typically calibrated on similar inputs before quantizing the model. During the calibration phase, we determine what scales and zero points to use, then lock them in for all future inference.

In this tutorial, we walk through an example of how to achieve this in torchao using ``Int8StaticActivationInt8WeightConfig`` and ``quantize_``. Let's start with our toy linear model:

.. code:: python

    from collections import OrderedDict
    import copy
    import torch

    from torchao.quantization import (
        AffineQuantizedMinMaxObserver,
        FqnToConfig,
        Int8StaticActivationInt8WeightConfig,
        MappingType,
        PerRow,
        PerTensor,
        quantize_,
    )

    class ToyLinearModel(torch.nn.Module):
        def __init__(self, m=64, n=32, k=64):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, k, bias=False)
            self.linear2 = torch.nn.Linear(k, n, bias=False)

        def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
            return (
                torch.randn(
                    batch_size, self.linear1.in_features, dtype=dtype, device=device
                ),
            )

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x


Calibration Phase
~~~~~~~~~~~~~~~~~

The goal of calibration is to determine fixed activation quantization parameters for each linear layer. We insert activation observers with forward pre-hooks, then run representative inputs through the original floating point model:

.. code:: python

    dtype = torch.bfloat16
    m = ToyLinearModel().eval().to(dtype).to("cuda")
    m_static = copy.deepcopy(m)

    activation_granularity = PerTensor()
    weight_granularity = PerRow()
    act_mapping_type = MappingType.SYMMETRIC

    activation_observers = OrderedDict()
    observer_handles = []

    def make_activation_observer():
        return AffineQuantizedMinMaxObserver(
            act_mapping_type,
            torch.int8,
            granularity=activation_granularity,
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int8,
            keepdim=True,
        )

    def observe_input(module, inputs, observer):
        observer(inputs[0])

    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            observer = make_activation_observer()
            activation_observers[name] = observer
            observer_handles.append(
                module.register_forward_pre_hook(
                    lambda module, inputs, observer=observer: observe_input(
                        module, inputs, observer
                    )
                )
            )

    with torch.no_grad():
        for _ in range(10):
            example_inputs = m.example_inputs(dtype=dtype, device="cuda")
            m(*example_inputs)

    for handle in observer_handles:
        handle.remove()

After calibration, each observer can compute the activation scale and zero point for its corresponding layer:

.. code:: python

    act_scale, act_zero_point = activation_observers["linear1"].calculate_qparams()


Quantization Phase
~~~~~~~~~~~~~~~~~~

Now we create one ``Int8StaticActivationInt8WeightConfig`` per linear layer and apply the configs with ``FqnToConfig``. This keeps the model structure unchanged and only replaces each linear weight with an ``Int8Tensor`` carrying the calibrated activation quantization parameters:

.. code:: python

    fqn_to_config = OrderedDict()
    for name, observer in activation_observers.items():
        act_scale, act_zero_point = observer.calculate_qparams()
        fqn_to_config[f"{name}.weight"] = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=act_scale,
            act_quant_zero_point=act_zero_point,
            granularity=[activation_granularity, weight_granularity],
            act_mapping_type=act_mapping_type,
        )

    quantize_(m_static, FqnToConfig(fqn_to_config), filter_fn=None)

Now, we will see that the linear layers in our model have fixed activation scales and quantized weights:

.. code::

    >>> m_static
    ToyLinearModel(
      (linear1): Linear(in_features=64, out_features=64, bias=False)
      (linear2): Linear(in_features=64, out_features=32, bias=False)
    )
    >>> type(m_static.linear1.weight)
    <class 'torchao.quantization.quantize_.workflows.int8.int8_tensor.Int8Tensor'>
    >>> m_static.linear1.weight.act_quant_scale  # fixed at calibration time
    tensor(..., device='cuda:0')

The model structure is unchanged. Only the weight tensors are replaced with quantized tensor subclasses carrying fixed activation scales. All subsequent forward passes will use the same scales, unlike dynamic quantization which recomputes them per batch.

Other Approaches
~~~~~~~~~~~~~~~~

The calibration phase can be customized:

- **Observers**: You can use lower-level observer APIs (``AffineQuantizedMinMaxObserver``) to record min/max statistics over calibration data and compute scales yourself. This gives full control over scale computation (e.g., moving averages, histograms).
- **AWQ / SmoothQuant**: These algorithms provide a ``prepare`` -> calibrate -> ``convert`` flow via ``quantize_()`` that integrates activation pre-scaling and smoothing with static quantization. They work with any tensor subclass that implements the relevant protocols — ``SupportsActivationPreScaling`` (``act_pre_scale`` attribute) for AWQ, and both ``IsStaticQuantizationConfig`` and ``SupportsActivationPreScaling`` for SmoothQuant. See the ``torchao.prototype.smoothquant`` and ``torchao.prototype.awq`` modules.
- **Float8 static quantization**: ``Float8StaticActivationFloat8WeightConfig`` (in ``torchao.prototype.quantization``) supports a built-in observer-based prepare/convert flow for float8 dtypes.

For a list of all available quantization configs, see the :doc:`API Reference <../api_reference/api_ref_quantization>`.

In this tutorial, we walked through how to perform integer static quantization in torchao using the current ``quantize_`` API with ``Int8StaticActivationInt8WeightConfig``.
