# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py

import torch

from torchao.ops import lib
from torchao.prototype.spinquant._hadamard_matrices import (
    get_had12,
    get_had20,
    get_had28,
    get_had36,
    get_had40,
    get_had44,
    get_had52,
    get_had60,
    get_had108,
    get_had140,
    get_had156,
    get_had172,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform

    def matmul_hadU(X, hadK, K):
        if X.is_cuda:
            return matmul_hadU_fast(X, hadK, K)
        else:
            return matmul_hadU_slow(X, hadK, K)

except ImportError:
    print(
        "NOTE: Using slow Hadamard transform for SpinQuant. "
        "For better performance on GPU, install `fast_hadamard_transform`: "
        "`pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git`"
    )

    def matmul_hadU(X, hadK, K):
        return matmul_hadU_slow(X, hadK, K)


def register_custom_op_impl(name):
    def decorator(func):
        if TORCH_VERSION_AT_LEAST_2_4:
            return torch.library.custom_op(f"{name}", mutates_args=())(func)
        else:
            lib.define("hadamard_transform(Tensor x, float scale = 0.0) -> Tensor")
            return torch.library.impl(f"{name}", "cuda")(func)

    return decorator


def register_custom_op_abstract(name):
    def decorator(func):
        if TORCH_VERSION_AT_LEAST_2_4:
            return torch.library.register_fake(f"{name}")(func)
        else:
            return torch.library.impl_abstract(f"{name}")(func)

    return decorator


@register_custom_op_impl("torchao::hadamard_transform")
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.

    Source: https://github.com/Dao-AILab/fast-hadamard-transform
    """
    return _fast_hadamard_transform(x, scale)


@register_custom_op_abstract("torchao::hadamard_transform")
def _(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    torch._check(
        x.dim() >= 1, lambda: f"input should be at least a 1D tensor, got {x.dim()}D"
    )
    return torch.empty_like(x)


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return _fast_hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return _fast_hadamard_transform(grad)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)

        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)

        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)

        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)

        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)

        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)

        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert is_pow2(n // 36)

        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        assert is_pow2(n // 28)

        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 44 == 0:
        assert is_pow2(n // 44)

        K = 44
        hadK = get_had44().T if transpose else get_had44()
    elif n % 40 == 0:
        assert is_pow2(n // 40)

        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert is_pow2(n // 20)

        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert is_pow2(n // 12)

        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert is_pow2(n)
        hadK = torch.FloatTensor([[1]])
        K = 1

    return hadK, K


def matmul_hadU_slow(X, hadK, K):
    n = X.shape[-1]
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadU_fast(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return (
            torch.ops.torchao.hadamard_transform.default(X.contiguous())
            / torch.tensor(n).sqrt()
        )
    input = X.view(-1, K, n // K)
    input = (
        torch.ops.torchao.hadamard_transform.default(input.contiguous())
        / torch.tensor(n).sqrt()
    )
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def random_hadamard_matrix(size, device, seed=0):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Q = torch.randint(low=0, high=2, size=(size,), generator=gen).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    hadK, K = get_hadK(size)
    return matmul_hadU_slow(Q, hadK, K).to(device)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    hadK, K = get_hadK(size)
    return matmul_hadU_slow(Q, hadK, K).to(device)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W = module.weight.data
    dtype_orig = W.dtype
    W = W.float()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W = matmul_hadU(W.t(), had_K.to(W.device), K).t()
        else:
            had_K, K = get_hadK(in_features)
            W = matmul_hadU(W, had_K.to(W.device), K)
    else:
        if R2 is not None:
            hadK = R2.to(torch.float64)
        else:
            hadK = hadamard_matrix(had_dim, W.device).to(torch.float64)

        if output:
            W = W.t()

        shape = W.shape
        temp = W.reshape(-1, shape[-1] // had_dim, had_dim)
        temp = temp.to(torch.float64) @ hadK
        W = temp.reshape(shape)

        if output:
            W = W.t()

    module.weight.data = W.to(dtype=dtype_orig)
