Writing Your Own Quantized Tensor (advanced)
--------------------------------------------

In the previous tutorial, we covered the basics of implementing a quantized tensor 
subclass with Int8SymmetricTensor. Here, we'll explore more advanced features by 
implementing tensor parallelism support in our quantized tensor subclass. We'll see 
how tensor subclasses can elegantly compose with PyTorch's distributed primitives.

What is Tensor Parallelism?
===========================

Tensor parallelism is a technique for distributing large tensors (typically model 
weights) across multiple devices to enable training and inference of large models 
that wouldn't fit on a single device. Unlike data parallelism which replicates 
the entire model across devices, tensor parallelism splits individual tensors.

For example, consider a linear layer with weight matrix W of shape [2048, 1024]. 
With 2-way tensor parallelism, we could:

1. Split W column-wise into [2048, 512] chunks on each device (column parallel)
2. Split W row-wise into [1024, 1024] chunks on each device (row parallel)

The choice of splitting strategy affects both memory usage and communication 
patterns during forward/backward passes.

Benefits of tensor parallelism include:

1. **Memory efficiency**: Each device only stores a portion of the full tensor
2. **Computation parallelism**: Matrix operations can run in parallel across devices
3. **Bandwidth optimization**: Reduced communication compared to model parallelism

Why Use Tensor Subclasses for Parallelism?
=========================================

PyTorch provides DTensor (Distributed Tensor) as a tensor subclass for distributed 
computing. By composing our quantized tensor with DTensor, we get:

1. **Clean composition**: No need to create new modules that duplicate both 
   quantization and distribution logic
2. **Bandwidth savings**: DTensor can be aware that inner tensors are quantized, 
   enabling communication of compressed data
3. **Flexibility**: Works with any model using standard PyTorch ops, not just 
   specific modules

Implementing Tensor Parallel Support
==================================

Let's extend our Int8SymmetricTensor from the previous tutorial to support tensor 
parallelism. First, we'll import the necessary dependencies:

.. code:: python

    import os
    from typing import Sequence
    import torch
    import torch.distributed as dist
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
    from torch.utils._python_dispatch import return_and_correct_aliasing

Creating the Parallel Tensor Class
--------------------------------

We'll create a new Int8SymmetricTensorTP class that inherits from Int8SymmetricTensor:

.. code:: python

    class Int8SymmetricTensorTP(Int8SymmetricTensor):
        """
        A tensor subclass that supports both quantization and tensor parallelism.
        Builds on Int8SymmetricTensor by adding support for DTensor operations.
        """
        pass

    implements = Int8SymmetricTensorTP.implements
    aten = torch.ops.aten

Key Operations
-------------

We need to implement several key operations to support tensor parallelism:

1. **Basic Operations**: Copy and clone operations needed for DTensor

.. code:: python

    @implements([aten._to_copy.default, aten.clone.default])
    def _(func, types, args, kwargs):
        # Need to clone both int8 data and scale
        new_int8_data = torch.clone(args[0].int_data)
        new_scale = torch.clone(args[0].scale)
        out = Int8SymmetricTensorTP(new_int8_data, new_scale)
        return return_and_correct_aliasing(func, args, kwargs, out)

2. **Split Operations**: Required for sharding tensors

.. code:: python

    @implements([aten.split.Tensor])
    def _(func, types, args, kwargs):
        # Split both int8 data and scale
        int8_splits = func(args[0].int_data, *args[1:], **kwargs)
        scale_splits = func(args[0].scale, *args[1:], **kwargs)
        out = [
            Int8SymmetricTensorTP(int8_data, scale)
            for int8_data, scale in zip(int8_splits, scale_splits)
        ]
        return out

3. **View/Reshape Operations**: Required for DTensor's internal reshaping

.. code:: python

    @implements(aten.view.default)
    def _(func, types, args, kwargs):
        x, shape = args
        
        if tuple(x.shape) == tuple(shape):
            return x

        if len(shape) == 1 and shape[0] == -1:
            # Flatten both int8 data and scale
            flat_int8 = x.int_data.view(-1)
            flat_scale = x.scale.view(-1, 1)
            return Int8SymmetricTensorTP(flat_int8, flat_scale)

        raise ValueError(
            f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]"
        )

4. **Matrix Operations**: Handle sharded matrix multiplication

.. code:: python

    @implements(aten.mm.default)
    def _(func, types, args, kwargs):
        input_tensor, weight_tensor = args[0], args[1]
        if isinstance(weight_tensor, Int8SymmetricTensorTP):
            # Dequantize weight tensor for the matmul
            fp32_weight = weight_tensor.int_data.to(torch.float32) * weight_tensor.scale
            return torch.mm(input_tensor, fp32_weight)
        return torch.mm(input_tensor, weight_tensor)

Sharding Helper Functions
-----------------------

To simplify sharding operations, we'll implement helper functions:

.. code:: python

    def shard(
        tensor: Int8SymmetricTensorTP,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
    ) -> DTensor:
        """
        Shard a quantized tensor into a DTensor based on indicated placements.
        """
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        shape, offset = compute_local_shape_and_global_offset(
            tensor.shape, device_mesh, placements
        )
        slices = [
            slice(cur_offset, cur_offset + cur_shape)
            for cur_shape, cur_offset in zip(shape, offset)
        ]
        # Shard both int8 data and scale
        local_int8 = tensor.int_data[slices]
        local_scale = tensor.scale[slices[0]:slices[0].stop]
        local_tensor = Int8SymmetricTensorTP(local_int8, local_scale)
        return DTensor.from_local(local_tensor, device_mesh, placements)

    def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer weights column-wise (input features).
        Assumes the weight is already quantized as Int8SymmetricTensorTP.
        """
        assert isinstance(m.linear.weight, Int8SymmetricTensorTP), \
            "Weight must be quantized before sharding"
        # Column-wise is wrt to A^T, so for A it is row-wise
        dtensor = shard(m.linear.weight, mesh, [Shard(0)])
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m

    def rowwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer weights row-wise (output features).
        Assumes the weight is already quantized as Int8SymmetricTensorTP.
        """
        assert isinstance(m.linear.weight, Int8SymmetricTensorTP), \
            "Weight must be quantized before sharding"
        # Row-wise is wrt to A^T, so for A it is column-wise
        dtensor = shard(m.linear.weight, mesh, [Shard(1)])
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m

Using the Parallel Tensor
========================

Here's a complete example showing how to use our parallel tensor with a simple model:

.. code:: python

    class M(torch.nn.Module):
        def __init__(self, in_features, out_features, **kwargs) -> None:
            super().__init__(**kwargs)
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def main():
        # Initialize distributed environment
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # Create model
        proj_up = M(1024, 2048).to(device)
        proj_dn = M(2048, 1024).to(device)
        
        # Create example input
        example_input = torch.randn(128, 1024, device=device)

        # Quantize the models
        up_quant = quantize(proj_up)
        dn_quant = quantize(proj_dn)

        # Initialize distributed
        dist.init_process_group(backend="nccl")
        mesh = dist.init_device_mesh("cuda", (world_size,))

        # Shard the models
        up_dist = colwise_shard(up_quant, mesh)
        dn_dist = rowwise_shard(dn_quant, mesh)

        # Convert input to DTensor
        input_dtensor = DTensor.from_local(example_input, mesh, [Replicate()])

        # Run forward pass
        y_d = dn_dist(up_dist(input_dtensor))

        # Works with torch.compile too!
        up_compiled = torch.compile(up_dist)
        dn_compiled = torch.compile(dn_dist)
        y_c = dn_compiled(up_compiled(input_dtensor))

The choice between row-wise and column-wise sharding depends on your model 
architecture and performance requirements. Column-wise sharding requires 
communication before the matrix multiply, while row-wise requires communication 
after.

Next Steps
==========

This tutorial demonstrated how to extend a basic quantized tensor subclass to 
support tensor parallelism. For more examples:

1. Check the full implementation in the `tensor_parallel.py example <https://github.com/pytorch/ao/tree/main/tutorials/developer_api_guide/tensor_parallel.py>`__
2. Learn about DTensor in the `PyTorch documentation <https://pytorch.org/docs/stable/distributed.tensor.html>`__
3. Explore the `AffineQuantizedTensor implementation <https://github.com/pytorch/ao/blob/main/torchao/dtypes/affine_quantized_tensor.py>`__

If you have questions while implementing your parallel tensor subclass, feel free 
to file an issue `here <https://github.com/pytorch/ao/issues>`__.


