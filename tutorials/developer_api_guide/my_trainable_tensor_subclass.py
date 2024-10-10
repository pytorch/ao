"""
This is an example for a tensor subclass representing a simple dtype
that can be used in training.

We extend our previous example of `MyDTypeTensor` with a few extra steps
needed to ensure proper gradient updates during training:

  1. Define a differentiable constructor
  2. Define backward pass for ops of interest (e.g. torch.nn.functional.linear)
  3. Handle special ops used by the optimizer (e.g. aten.add, aten.add_)
"""

import torch

from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_primitives import choose_qparams_affine, MappingType
from torchao.dtypes.utils import Layout, PlainLayout
from my_dtype_tensor_subclass import MyDTypeLayout, MyDTypeTensor

aten = torch.ops.aten


##############################
# Tensor Subclass Definition #
##############################

class MyTrainableDTypeTensor(MyDTypeTensor):
    """
    Example tensor subclass that extends `MyDTypeTensor` to support training.
    """

    @classmethod
    def _quantize(
        cls,
        input_float: torch.Tensor,
        _layout: Layout,
    ) -> MyDTypeLayout:
        """
        Convert from a floating point tensor (fp32/fp16/bf16) to the desired dtype.
        """
        mapping_type = MappingType.SYMMETRIC
        block_size = input_float.shape
        dtype = torch.int16
        scale, _ = choose_qparams_affine(input_float, mapping_type, block_size, dtype)
        int_data = (input_float / scale).to(torch.int8)
        tensor_impl_ctr = cls.get_tensor_impl_constructor(type(_layout))
        return tensor_impl_ctr(int_data, scale, _layout)

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        _layout: Layout = PlainLayout(),
    ) -> "MyTrainableDTypeTensor":
        """
        Main entry point for creating a `MyTrainableDTypeTensor`.

        This instantiates the tensor subclass in a differentiable constructor
        to ensure gradients are passed to the tensor subclass properly during training.
        """
        return _ToMyTrainableDTypeTensor.apply(input_float, _layout)

class _ToMyTrainableDTypeTensor(torch.autograd.Function):
    """
    Differentiable constructor for `MyTrainableDTypeTensor`.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_float: torch.Tensor,
        _layout: Layout,
    ) -> "MyTrainableDTypeTensor":
        tensor_impl = MyTrainableDTypeTensor._quantize(input_float, _layout)
        return MyTrainableDTypeTensor(
            tensor_impl,
            input_float.shape,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx, gy):
        return gy, None

to_my_trainable_dtype = MyTrainableDTypeTensor.from_float


#####################################################
# torch functional and aten operator implementation #
#####################################################

implements = MyTrainableDTypeTensor.implements

class _QuantizedLinearOp(torch.autograd.Function):
    """
    Forward and backward definition for linear with quantized weights.
    Weights are dequantized during both the forward and the backward passes.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(weight_tensor, MyTrainableDTypeTensor)
        ctx.save_for_backward(input_tensor, weight_tensor)
        weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight_tensor = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight_tensor.dequantize())
        grad_weight = torch.matmul(
            grad_output.view(-1, weight_tensor.shape[0]).T,
            input_tensor.view(-1, weight_tensor.shape[1]),
        )
        return grad_input, grad_weight

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    """
    Handle the linear op with quantized weights.
    For simplicity, we run both the forward and backward passes entirely in float.
    """
    assert isinstance(args[1], MyTrainableDTypeTensor)
    if len(args) > 2 and args[2] is not None:
        raise NotImplementedError("linear bias not yet supported")
    return _QuantizedLinearOp.apply(args[0], args[1])

@implements(aten.add_.Tensor)
def _(func, types, args, kwargs):
    """
    Handle the in-place add op, called by the optimizer to update
    the quantized weight during training.
    """
    assert len(args) == 2
    assert isinstance(args[0], MyTrainableDTypeTensor)
    assert args[0].tensor_impl.int_data.dtype == torch.int8
    float0 = args[0].dequantize()
    float1 = args[1].dequantize() if isinstance(args[1], MyTrainableDTypeTensor) else args[1]
    new_value = torch.add(float0, float1, **kwargs)
    new_tensor_impl = MyTrainableDTypeTensor._quantize(
        new_value,
        args[0].tensor_impl.get_layout(),
    )
    args[0].tensor_impl = new_tensor_impl
    return return_and_correct_aliasing(func, args, kwargs, args[0])

@implements(aten.add.Tensor)
def _(func, types, args, kwargs):
    """Handle the add op, called by the optimizer during training."""
    assert len(args) == 2
    assert not isinstance(args[0], MyTrainableDTypeTensor)
    assert isinstance(args[1], MyTrainableDTypeTensor)
    out = torch.add(args[0], args[1].dequantize(), **kwargs)
    return return_and_correct_aliasing(func, args, kwargs, out)


########
# Test #
########

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(512, 1024, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def main():
    m = M().cuda()
    NUM_TRAIN_STEPS = 10
    VERBOSE = True

    # Convert weights to quantized weights
    m.linear.weight = torch.nn.Parameter(
        to_my_trainable_dtype(m.linear.weight), requires_grad=True,
    )

    # Dummy training loop
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(NUM_TRAIN_STEPS):
        example_inputs = (torch.randn(512).cuda(),)
        target = torch.randn(1024).cuda()
        output = m(*example_inputs)
        loss = loss_fn(output, target)
        loss.backward()
        if VERBOSE:
            weight = m.linear.weight.tensor_impl.int_data.flatten()[:3]
            weight_grad = m.linear.weight.grad.flatten()[:3]
            print(" * step %s: weight grad = %s, weight value = %s" % (i, weight_grad, weight))
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()
