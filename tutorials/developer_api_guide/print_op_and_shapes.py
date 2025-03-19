# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch

PRINT_ARGS = False
linear_shapes = []
from torch.overrides import TorchFunctionMode


class TorchFunctionLoggingMode(TorchFunctionMode):
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.nn.functional.linear:
            input_tensor, weight_tensor, _ = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            flattened_input_tensor = input_tensor.view(-1, input_tensor.shape[-1])
            M, K = flattened_input_tensor.shape[0], flattened_input_tensor.shape[1]
            assert K == weight_tensor.shape[1]
            N = weight_tensor.shape[0]
            print(f"TORCH_FUNC {func=} (M, K, N):", M, K, N)
            linear_shapes.append((M, K, N))
        else:
            arg_shape = (
                args[0].shape
                if len(args) > 0 and isinstance(args[0], torch.Tensor)
                else None
            )
            if PRINT_ARGS:
                print(
                    f"TORCH_FUNC {func=}, {types=}, {args=}, {kwargs=}, args[0] shape: {arg_shape}"
                )
            else:
                print(f"TORCH_FUNC {func=}, {types=}, args[0] shape: {arg_shape}")
        return func(*args, **kwargs)


from torch.utils._python_dispatch import TorchDispatchMode


class TorchDispatchLoggingMode(TorchDispatchMode):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        arg_shape = (
            args[0].shape
            if len(args) > 0 and isinstance(args[0], torch.Tensor)
            else None
        )
        if PRINT_ARGS:
            print(
                f"ATEN_FUNC {func=}, {types=}, {args=}, {kwargs=}, args[0] shape: {arg_shape}"
            )
        else:
            print(f"ATEN_FUNC {func=}, {types=}, args[0] shape: {arg_shape}")

        return func(*args, **kwargs)


# NOTE: Modify this with your own model
from torchvision import models

m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
example_inputs = (torch.randn(1, 3, 224, 224),)

with TorchFunctionLoggingMode():
    m(*example_inputs)

print()
print("all linear shapes (M, K, N):", linear_shapes)

# check all aten ops that's called in the model
# with TorchDispatchLoggingMode():
#     m(*example_inputs)
