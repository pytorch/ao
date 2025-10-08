Writing Your Own Quantized Tensor
---------------------------------

Quantization in torchao is built on the foundation of tensor subclasses.
They are the main extension point for torchao to provide flexible
inference and training support using low precision computation, while
composing with important PyTorch features such as torch.compile,
autograd, and distributed primitives.

In this tutorial, we will highlight the benefits of leveraging tensor
subclasses compared to module swaps, and walk through a simple example
of how to express quantization using this approach.

What are Tensor Subclasses?
===========================

Tensor subclasses are simply classes that inherit from `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`__.
They allow users to interpose their custom computation logic between existing
ops in their models, such that functions in the top-level torch
namespace like torch.add will continue to work seamlessly.

An obvious alternative to the tensor subclass approach is module swaps:
simply swap all nn.Linear modules in your model with your custom
Int8QuantizedLinear modules, for example. There are a few important
benefits of using tensor subclasses compared to this approach:

1. **Finer-grained integration point.** Module swaps intercept
   computation at the module level and so will not work for models that
   rely on torch functions or variants of native modules (e.g. slightly
   modified versions of nn.Linear). In contrast, since tensor subclasses
   intercept computation at the function/op level, we will be able to
   quantize the model as long as the same function/op is used.

2. **Better composability.** Composing multiple features using module
   swaps is clunky. For example, combining two existing
   Int8QuantizedLinear and DistributedLinear modules would require users
   to create another linear class that duplicates these functionalities.
   Tensor subclasses bypass this problem by simply wrapping one subclass
   in another. This can also offer performance benefits if the outer
   tensor (e.g. `DTensor <https://pytorch.org/docs/stable/distributed.tensor.html>`__)
   is aware that the inner tensor is quantized, and so can perform
   expensive allgather operations using less network and memory
   bandwidth.

3. **Reusing PyTorch components.** It is natural to express quantization
   using tensor subclasses since the quantized tensors are simply
   torch.Tensors with different dtypes. The model structure does not
   change (nn.Linears stay as nn.Linears), and so subsequent
   optimization passes can also stay exactly the same as before.

|
In the rest of the tutorial, we will walk through an example of how to
express quantization using both approaches. For further reading on
tensor subclasses, please refer to:

-  `Tensor subclass documentation <https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-like-type>`__
-  `Tensor subclass zoo <https://github.com/albanD/subclass_zoo>`__
-  `Tensor subclass podcast by Edward Yang <https://podcasts.apple.com/us/podcast/tensor-subclasses-and-pt2/id1566080008?i=1000646728968>`__

Quantization with Module Swaps
==============================

We begin with a simple example of how to implement int8 symmetric weight
only quantization using module swaps. All code can be found in this
`example script <https://github.com/pytorch/ao/tree/main/tutorials/examples/quantized_module_swap.py>`__.
We will use the following function for quantizing float32 tensors into
int8 tensors:

.. code:: py

   from typing import Tuple
   import torch

   def int8_symmetric_quantize(
       fp32_tensor: torch.Tensor,
   ) -> Tuple[torch.Tensor, torch.Tensor]:
       """
       Symmetrically quantize the torch.float32 tensor into torch.int8.
       Return a 2-tuple of (quantized value, scale).

       input: dimensions=[M, N], dtype=torch.float32
       output: dimensions=[M, N], dtype=torch.int8
       scale: dimensions=[M, 1], dtype=torch.float32
       """
       quant_min = -128
       quant_max = 127
       min_val = torch.amin(fp32_tensor, dim=[1], keepdim=False)
       max_val = torch.amax(fp32_tensor, dim=[1], keepdim=False)
       min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
       max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
       max_val_pos = torch.max(-min_val_neg, max_val_pos)
       scale = max_val_pos / (float(quant_max - quant_min) / 2)
       scale = scale.view(fp32_tensor.shape[0], -1)
       out = torch.round(fp32_tensor * (1.0 / scale))
       out = torch.clamp(out, quant_min, quant_max).to(torch.int8)
       return out, scale

Next, we will create a new QuantizedLinear module that calls this
function to dynamically quantize the weights:

.. code:: py

   class QuantizedLinear(torch.nn.Linear):
       """
       Linear module that performs dynamic and symmetric weight-only
       int8 quantization.
       """
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           w_int8, scale = int8_symmetric_quantize(self.weight)
           return torch.matmul(x, w_int8.t().to(x.dtype)) * scale.t()

       @classmethod
       def from_float(cls, mod: torch.nn.Linear):
           new_linear = cls(mod.in_features, mod.out_features, mod.bias)
           new_linear.weight = mod.weight
           return new_linear

Then, the only thing that’s left is to swap all `nn.Linear` modules in the
model with our new QuantizedLinear. Let’s use the following toy model
for demonstration purposes:

.. code:: py

   import copy

   class ToyModel(torch.nn.Module):
       def __init__(self, m: int, n: int, k: int):
           super().__init__()
           self.linear1 = torch.nn.Linear(m, n, bias=False)
           self.linear2 = torch.nn.Linear(n, k, bias=False)

       def forward(self, x):
           x = self.linear1(x)
           x = self.linear2(x)
           return x

   float_model = ToyModel(64, 128, 32).cuda()
   quantized_model = copy.deepcopy(float_model)

   # Swap torch.nn.Linear with QuantizedLinear
   for name, child in quantized_model.named_children():
       if type(child) == torch.nn.Linear:
           new_linear = QuantizedLinear.from_float(child)
           setattr(quantized_model, name, new_linear)

Verify that the model now uses our QuantizedLinear module. This model is
now ready to use!

.. code:: py

   >>> print(float_model)
   ToyModel(
     (linear1): Linear(in_features=64, out_features=128, bias=False)
     (linear2): Linear(in_features=128, out_features=32, bias=False)
   )

   >>> print(quantized_model)
   ToyModel(
     (linear1): QuantizedLinear(in_features=64, out_features=128, bias=False)
     (linear2): QuantizedLinear(in_features=128, out_features=32, bias=False)
   )

An important drawback of this simple approach is flexibility. Currently
this only works for native PyTorch modules, but what if the model has
slightly modified linear modules that, for example, support distributed
training? It also won’t work with models that directly call the functional
version of linear (`torch.nn.functional.linear`) instead.

Further, suppose we want to compose this feature with distribution,
which is also implemented through module swaps. There is no clean way to
do this except to create yet another module that combines both features.
These limitations can be solved with tensor subclasses, which is a more
elegant way to interpose custom computation such as quantization in your
model.

Quantization with Tensor Subclasses
===================================

Here we are going to re-implement the above quantization technique,
using a `__torch_dispatch__`-based tensor subclass.

Tensor subclasses (which often utilize `__torch_dispatch__`) are a pretty
powerful/flexible extension point in pytorch. They serve two main
purposes as an extension point:

1) Tensor subclasses allow you to override the **implementation** of
   (almost) every PyTorch API, and are used quite a bit to implement
   other PyTorch offerings
2) Tensor subclasses allow you to **couple** your tensor data with
   additional metadata. A few examples

   1) [distributed] metadata on how a tensor is sharded across ranks
      (`DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py#L217>`__,
      `docs <https://pytorch.org/docs/stable/distributed.tensor.html#pytorch-dtensor-distributed-tensor>`__)
   2) [quantization] scale/zero_point metadata
      (`AffineQuantizedTensor <https://github.com/pytorch/ao/blob/v0.8.0/torchao/dtypes/affine_quantized_tensor.py#L46>`__)
   3) [raggedness] metadata on ragged structure
      (`NestedTensor <https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/nested_tensor.py#L53>`__,
      `docs <https://pytorch.org/tutorials/prototype/nestedtensor.html#getting-started-with-nested-tensors>`__)

Some other resources on tensor subclasses for those who are interested:

1) \__torch_dispatch_\_ docs
   (`link <https://pytorch.org/docs/stable/notes/extending.html#extending-torch-native-api>`__)
2) What (and why) is \__torch_dispatch_\_
   (`link <https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557>`__)
3) Google collab that implements a FlopCounter and MemoryTracker using
   \__torch_dispatch_\_
   (`link <https://colab.research.google.com/drive/1zjAisRrc8R6uixKsrs1DRm3lwz5MWN68?usp=sharing>`__)

With that out of the way, let’s start by defining our bare-bones tensor
subclass for symmetric quantization:

.. code:: py

  class Int8SymmetricTensor(torch.Tensor):
      """
      Our subclass represents a tensor that has been quantized to int8
      It will hold two inner tensors:
        int_data: int8[M, N]
        scale: fp32[M, 1]
      """

      @staticmethod
      @torch._dynamo.disable
      def __new__(cls, int_data: torch.Tensor, scale: torch.Tensor):
          return torch.Tensor._make_wrapper_subclass(
              cls,
              int_data.shape,
              strides=int_data.stride(),
              storage_offset=int_data.storage_offset(),
              dtype=scale.dtype,
              device=int_data.device,
          )
  
      @torch._dynamo.disable
      def __init__(self, int_data: torch.Tensor, scale: torch.Tensor):
          # inner data expected to be quantized already
          assert int_data.dtype is torch.int8
          # we could do more work to support ndim > 2!
          assert int_data.ndim == 2
          assert scale.ndim == 2
          self.int_data = int_data
          self.scale = scale
  
      def __tensor_flatten__(self) -> Tuple[List[str], Any]:
          """
          Returns a tuple of:
            names of all inner tensor attributes (two in our case)
            any other additional, non-tensor metadata.

          Needed for PT2 support.
          """
          return ["int_data", "scale"], None
  
      @classmethod
      def __tensor_unflatten__(cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None):
          """
           __tensor_unflatten__ should effectively undo __tensor_flatten__.

          inputs:
            a dict mapping names of inner tensor attributes back to the tensors
            the constant metadata from __tensor_flatten__
          output:
            a new instance of your subclass

          Needed for PT2 support.
          """
          assert extra_metadata is None
          int_data = tensor_data_dict["int_data"]
          scale = tensor_data_dict["scale"]
          return Int8SymmetricTensor(int_data, scale)
  
      def __repr__(self):
          return f'Int8SymmetricTensor(int_data={repr(self.int_data)}, scale={repr(self.scale)})'
  
      @staticmethod
      def from_float(float_tensor):
          """
          Actually performs the symmetric quantization.
          In our simple inference example we will quantize weights "ahead-of-time",
          although later in a training example we can quantize/dequantize
          during model execution, inside of our __torch_dispatch__

          input:
            float32 torch.Tensor
          output:
            Int8SymmetricTensor
          """
          int8_tensor, scale = int8_symmetric_quantize(float_tensor)
          return Int8SymmetricTensor(int8_tensor, scale)
  
      @classmethod
      def __torch_dispatch__(cls, func, types, args, kwargs):
          """
          Called for each ATen operator that our subclass is passed as an input to.
          We need to define our own implementation for every operator here.
          """
          if kwargs is None:
              kwargs = {}
          if func not in op_implementations_dict:
              raise AssertionError(f'Int8SymmetricTensor does not yet support op: {str(func)}')
          return op_implementations_dict[func](func, *args, **kwargs)
  

  # Convenience function for registering our own implementation
  # to every ATen operator in PyTorch
  op_implementations_dict = {}
  def register_op(ops: List[torch._ops.OpOverload]):
      def impl_decorator(op_impl):
          global op_implementations_dict
          for op in ops:
              op_implementations_dict[op] = op_impl
          return op_impl
  
      return impl_decorator

In the above code, we have done a few things:

1) Defined a basic “wrapper” tensor subclass - it is effectively a
   container object, that holds some inner data (in particular, two
   tensors that correspond to our int8 data and scales)
2) Defined a `__torch_dispatch__` implementation, which will be called
   for every ATen operator our model calls on any of our subclass inputs
3) (For PT2 support) Defined a `__tensor_flatten__`/`__tensor_unflatten__`
   method. This is the largest of a few requirements we have in order for
   our subclass to work with torch.compile (more on this later). It
   effectively tells `torch.compile` how to “desugar” our subclass into
   its inner components.
4) (For PT2 support) Added a `torch._dynamo.disable` decorator to both
   constructor methods (`__new__` and `__init__`) (more on this later).

Which operators should we implement?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch has a pretty large operator surface. Instead of trying to give
our new tensor subclass 100% coverage, let’s just focus on the ops we
need for our toy model above.

Which operators are called in our model though, so we know what to
implement first? The brute force way is to repeatedly run the model
to see what ops error in your subclass. A more elegant way is to log
every operator that your model sees during execution. This can be
achieved through another `LoggingTensor` subclass as in `this example <https://github.com/pytorch/ao/tree/main/tutorials/examples/logging_subclass.py>`__.

Let's implement the necessary ops below:

.. code:: py

   from torch.utils._python_dispatch import return_and_correct_aliasing

   @register_op([torch.ops.aten.mm.default])
   def int8_mm(func, x, weight):
       assert isinstance(weight, Int8SymmetricTensor), "Int8SymmetricTensor: matmul currently only supports the weight in low precision, not the input!"
       return torch.mm(x, weight.int_data.to(x.dtype)) * weight.scale

   @register_op([
       torch.ops.aten.detach.default,
       torch.ops.aten.t.default,
   ])
   def int8_view_ops(func, *args, **kwargs):
       assert isinstance(args[0], Int8SymmetricTensor)
       out_data = func(args[0].int_data, *args[1:], **kwargs)
       out_scale = func(args[0].scale, *args[1:], **kwargs)
       out = Int8SymmetricTensor(out_data, out_scale)
       return return_and_correct_aliasing(func, args, kwargs, out)

One thing you’ll notice quickly is: our model itself consists of a few
linear layers, but we see a few operations like `aten.t` and `aten.mm`
hitting our subclass. Some background:

-  We have a number of op decompositions that live in C++, that run
   “above” tensor subclasses. `linear` is one such op (the decomp
   lives `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LinearAlgebra.cpp#L2006>`__)
-  Decompositions can be good in the sense that they shrink the size of
   the API that you as a subclass author have to implement. But they can
   be painful if you would rather override the “higher level” operator
   than the underlying operations in its decomposition.
-  If you would prefer to override some operations (like Linear) at a
   higher level, you can do so using `__torch_function__`
   (`example <https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/nested_tensor.py#L336>`__).
   It’s worth noting that if you want autograd support, then any
   overrides you perform at the `__torch_function__` layer need to be
   written in a way that is differentiable, while any overrides you
   perform in `__torch_dispatch__` will be automatically differentiable.

There are a few nuances in our implementations worth pointing out:

1) You’ll notice that we no longer had to transpose our weight / scales
   inside of our mm implementation. That’s because the transposition
   “already happened” before we got to the `aten.mm` op.
2) Our `aten.mm` implementation does **not** return a tensor subclass
   output. In that sense, the “propagation” of our quantized subclass
   ends with matmuls. This maps to the fact that our weights are in low
   precision, but we need to perform the matmuls themselves in high
   precision. In general, subclass authors are free to choose for which
   ops their subclasses do-or-do-not propagate. If you wanted every
   function in your model to be quantized (including all pointwise and
   reduction operations), you could write your subclass implementation
   to quantize the output of every op and always return a subclass.
3) We were able to re-use the same implementation for 4 view operations.
   In general, many ops might work with a pretty generic implementation:
   unwrap any subclass inputs, run the underlying operator on the inner
   tensor, and wrap the output back into a subclass.

   - Whether you can always re-use an implementation, though, depends
     on what you are trying to do. For example, we implemented
     `transpose(dim0, dim1)` on our subclass by calling the same
     transpose on our inner data and inner scale tensor. This wouldn’t
     work if our scale and data tensors had a different number of
     dimensions, so transposition in that case would require a custom
     implementation.


Comparing the Outputs
=====================

And with all of that out of the way, let’s run our model with both
versions of quantization and confirm that they give the same output!

.. code:: py

   float_model = ToyModel(64, 128, 32).cuda()
   quantized_model_module_swap = copy.deepcopy(float_model)
   quantized_model_subclass = copy.deepcopy(float_model)

   # Swap torch.nn.Linear with QuantizedLinear
   for name, child in quantized_model_module_swap.named_children():
       if type(child) == torch.nn.Linear:
           new_linear = QuantizedLinear.from_float(child)
           setattr(quantized_model_module_swap, name, new_linear)

   # Swap torch.nn.Linear weights with Int8SymmetricTensor subclasses
   for name, child in quantized_model_subclass.named_children():
       if type(child) == torch.nn.Linear:
           subclass_param = Int8SymmetricTensor.from_float(child.weight)
           child.weight = torch.nn.Parameter(subclass_param, requires_grad=True)

   with torch.no_grad():
       x = torch.randn(64, 64, 64, device='cuda')
       out_module_swap = quantized_model_module_swap(x)
       out = quantized_model_subclass(x)
       print(torch.allclose(out, out_module_swap))  # prints True

       # We can also use torch.compile to fuse some of our quantized logic
       out_compiled = torch.compile(quantized_model_subclass)(x)
       print(torch.allclose(out, out_compiled))  # prints True


Next Steps
==========

In this tutorial, we demonstrated how to build a simple quantized tensor
subclass. This is part one of two tutorials in this series. The
`next post <subclass_advanced.html>`__ will discuss how to add more advanced
features to your tensor subclass, such as making it trainable, composing
with DTensors, and adding tensor parallelism support. For a more detailed
example of how `AffineQuantizedTensor` in torchao was built using tensor
subclasses, also check out `this example <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_dtype_tensor_subclass.py>`__.

If you have any questions while implementing your subclass, feel free to
file an issue `here <https://github.com/pytorch/ao/issues>`__.
