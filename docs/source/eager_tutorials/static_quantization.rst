Static Quantization
--------------------

Static quantization refers to using a fixed quantization range for all inputs during inference or generation. Unlike dynamic quantization, which dynamically computes new quantization ranges for each new input batch, static quantization typically results in more efficient computation, potentially at the cost of lower quantized accuracy since we cannot adapt to changes in the input distribution on-the-fly.

In static quantization, this fixed quantization range is typically calibrated on similar inputs before quantizing the model. During the calibration phase, we first insert observers into the model to "observe" the distribution of the inputs to be quantized, and use this distribution to decide what scales and zero points to ultimately use when quantizing the model.

In this tutorial, we walk through an example of how to achieve this in torchao. Let's start with our toy linear model:

.. code:: py

   import copy
   import torch

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

   dtype = torch.bfloat16
   m = ToyLinearModel().eval().to(dtype).to("cuda")
   m = torch.compile(m, mode="max-autotune")


Calibration Phase
~~~~~~~~~~~~~~~~~

torchao comes with a a simple observer implementation, `AffineQuantizedMinMaxObserver`, that records the min and max values that have flowed through the observer during the calibration phase. Users are welcome to implement their own desired, more advanced observation techniques, such as those relying on moving averages or histograms, and these may be added to torchao in the future.

.. code:: py

   from torchao.quantization.granularity import PerAxis, PerTensor
   from torchao.quantization.observer import AffineQuantizedMinMaxObserver
   from torchao.quantization.quant_primitives import MappingType

   # per tensor input activation asymmetric quantization
   act_obs = AffineQuantizedMinMaxObserver(
       MappingType.ASYMMETRIC,
       torch.uint8,
       granularity=PerTensor(),
       eps=torch.finfo(torch.float32).eps,
       scale_dtype=torch.float32,
       zero_point_dtype=torch.float32,
   )

   # per channel weight asymmetric quantization
   weight_obs = AffineQuantizedMinMaxObserver(
       MappingType.ASYMMETRIC,
       torch.uint8,
       granularity=PerAxis(axis=0),
       eps=torch.finfo(torch.float32).eps,
       scale_dtype=torch.float32,
       zero_point_dtype=torch.float32,
   )

Next, we define our observed linear that we will swap our `torch.nn.Linear` with. This is a high precision (e.g. fp32) linear module with the above observers inserted to record the input activation and weight values during calibration:

.. code:: py

   import torch.nn.functional as F

   class ObservedLinear(torch.nn.Linear):
       def __init__(
           self,
           in_features: int,
           out_features: int,
           act_obs: torch.nn.Module,
           weight_obs: torch.nn.Module,
           bias: bool = True,
           device=None,
           dtype=None,
       ):
           super().__init__(in_features, out_features, bias, device, dtype)
           self.act_obs = act_obs
           self.weight_obs = weight_obs

       def forward(self, input: torch.Tensor):
           observed_input = self.act_obs(input)
           observed_weight = self.weight_obs(self.weight)
           return F.linear(observed_input, observed_weight, self.bias)

       @classmethod
       def from_float(cls, float_linear, act_obs, weight_obs):
           observed_linear = cls(
               float_linear.in_features,
               float_linear.out_features,
               act_obs,
               weight_obs,
               False,
               device=float_linear.weight.device,
               dtype=float_linear.weight.dtype,
           )
           observed_linear.weight = float_linear.weight
           observed_linear.bias = float_linear.bias
           return observed_linear

To actually insert these observers into our toy model:

.. code:: py

   from torchao.quantization.quant_api import (
       _replace_with_custom_fn_if_matches_filter,
   )

   def insert_observers_(model, act_obs, weight_obs):
       _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)

       def replacement_fn(m):
           copied_act_obs = copy.deepcopy(act_obs)
           copied_weight_obs = copy.deepcopy(weight_obs)
           return ObservedLinear.from_float(m, copied_act_obs, copied_weight_obs)

       _replace_with_custom_fn_if_matches_filter(model, replacement_fn, _is_linear)

   insert_observers_(m, act_obs, weight_obs)

Now we are ready to calibrate the model, which populates the observers we inserted with statistics recorded during the calibration. We can do this simply by feeding some example inputs to our "observed" model:

.. code:: py

   for _ in range(10):
       example_inputs = m.example_inputs(dtype=dtype, device="cuda")
       m(*example_inputs)


Quantization Phase
~~~~~~~~~~~~~~~~~~

There are multiple ways to actually quantize the model. Here we walk through the simpler alternative, which is to define a `QuantizedLinear` class that we will swap our `ObservedLinear` to.

.. code:: py

   from torchao.quantization import Int8Tensor
   from torchao.quantization import PerRow, PerTensor

   class QuantizedLinear(torch.nn.Module):
       def __init__(
           self,
           in_features: int,
           out_features: int,
           act_obs: torch.nn.Module,
           weight_obs: torch.nn.Module,
           weight: torch.Tensor,
           bias: torch.Tensor,
       ):
           super().__init__()
           self.act_scale, self.act_zero_point = act_obs.calculate_qparams()
           weight_scale, weight_zero_point = weight_obs.calculate_qparams()
           self.bias = bias
           self.qweight = Int8Tensor.from_hp(
               weight, granularity=PerRow(),
               scale=weight_scale, zero_point=weight_zero_point,
           )

       def forward(self, input: torch.Tensor):
           qinput = Int8Tensor.from_hp(
               input,
               granularity=PerTensor(),
               scale=self.act_scale,
               zero_point=self.act_zero_point,
           )
           return F.linear(qinput, self.qweight, self.bias)

       @classmethod
       def from_observed(cls, observed_linear, target_dtype):
           quantized_linear = cls(
               observed_linear.in_features,
               observed_linear.out_features,
               observed_linear.act_obs,
               observed_linear.weight_obs,
               observed_linear.weight,
               observed_linear.bias,
               target_dtype,
           )
           return quantized_linear

This linear class computes the scales and zero points for both input activations and weights in the beginning, effectively fixing the quantization range for future forward calls. Now, to actually quantize the model using this linear class, we can define the following config and pass it to torchao's main `quantize_` API:

.. code:: py

   from dataclasses import dataclass

   from torchao.core.config import AOBaseConfig
   from torchao.quantization import quantize_
   from torchao.quantization.transform_module import (
       register_quantize_module_handler,
   )

   @dataclass
   class StaticQuantConfig(AOBaseConfig):
       target_dtype: torch.dtype

   @register_quantize_module_handler(StaticQuantConfig)
   def _apply_static_quant(
       module: torch.nn.Module,
       config: StaticQuantConfig,
   ):
       """
       Define a transformation associated with `StaticQuantConfig`.
       This is called by `quantize_`, not by the user directly.
       """
       return QuantizedLinear.from_observed(module, config.target_dtype)

   # filter function to identify which modules to swap
   is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)

   # perform static quantization
   quantize_(m, StaticQuantConfig(torch.uint8), is_observed_linear)

Now, we will see that the linear layers in our model are swapped to our `QuantizedLinear` class, with a fixed input activation scale and a fixed quantized weight:

.. code:: py

   >>> m
   OptimizedModule(
     (_orig_mod): ToyLinearModel(
       (linear1): QuantizedLinear()
       (linear2): QuantizedLinear()
     )
   )
   >>> m.linear1.act_scale
   tensor([0.0237], device='cuda:0')
   >>> m.linear1.qweight  # quantized weight tensor with scale and zero_point
   IntxUnpackedToInt8Tensor(...)  # actual repr depends on quantization config

In this tutorial, we walked through a basic example of how to perform integer static quantization in torchao.
