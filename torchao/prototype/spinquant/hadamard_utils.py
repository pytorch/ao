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

try:
    """
    To install the fast_hadamard_transform package:
    ```
        git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
        cd fast-hadamard-transform
        pip install .
    ```
    """
    from fast_hadamard_transform import hadamard_transform
except:
    pass

from torchao.quantization._hadamard_matrices import get_had172, get_had156, get_had140, get_had108, get_had60, get_had52, get_had36, get_had28, get_had44, get_had40, get_had20, get_had12, get_had60, get_had52, get_had36, get_had28, get_had44, get_had40, get_had20, get_had12


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad)


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

        K = 1

    return hadK, K


def matmul_hadU(X, hadK, K):
    if X.device == torch.device("cpu"):
        return matmul_hadU_cpu(X, hadK, K)
    else:
        return matmul_hadU_cuda(X, hadK, K)


def matmul_hadU_cpu(X, hadK, K):
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


def matmul_hadU_cuda(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(n).sqrt()
    input = X.view(-1, K, n // K)
    input = HadamardTransform.apply(input.contiguous()) / torch.tensor(n).sqrt()
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    hadK, K = get_hadK(size)
    return matmul_hadU_cpu(Q, hadK, K).to(device)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    hadK, K = get_hadK(size)
    return matmul_hadU_cpu(Q, hadK, K).to(device)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    device_orig = W_.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_shape = W_.shape
    W_ = W_.float().to(device=device)

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, device).to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=device_orig, dtype=dtype)