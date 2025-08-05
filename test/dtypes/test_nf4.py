# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import io
import logging
import math
import unittest
from collections import OrderedDict
from typing import Tuple, Union

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal import common_utils
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if common_utils.SEED is None:
    common_utils.SEED = 1234

import torchao
from packaging import version
from torchao.dtypes._nf4tensor_api import nf4_weight_only
from torchao.dtypes.nf4tensor import (
    _INNER_TENSOR_NAMES_FOR_SHARDING,
    NF4Tensor,
    linear_nf4,
    to_nf4,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

bnb_available = False

try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def _build_input_weight(embed_dim: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=dtype)
    input_weight.normal_(0, 1)
    return input_weight


def _build_bnb_linear(input_weight, device):
    assert bnb_available, "Needs bitsandbytes support"
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(
        device
    )
    bnb_linear = bnb.nn.LinearNF4(
        input_weight.size(0), input_weight.size(1), bias=False
    )
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


class TestNF4Linear(TestCase):
    class TestMod(nn.Module):
        def __init__(self, tensor, block_size, scaler_block_size):
            super().__init__()
            self.param = torch.nn.Parameter(
                to_nf4(tensor, block_size, scaler_block_size)
            )

    def save_state_dict_to_buffer(self, state_dict: OrderedDict):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        return buffer

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_register_nf4_as_param(self, dtype: torch.dtype):
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=dtype))

        # Would raise if nn.Parameter registration fails, such as no detach()
        # impl when calling __torch_dispatch__
        param = torch.nn.Parameter(nf4_tensor, requires_grad=False)
        assert not param.requires_grad

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_output_dtype_match(self, dtype: torch.dtype):
        # Test to ensure W4 A16 produces A16
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=dtype))
        out = linear_nf4(input=inp, weight=nf4_tensor)
        assert out.dtype == dtype

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_backward_dtype_match(self, dtype: torch.dtype):
        # Test to ensure backward pass gives activation a bf16 gradient and no gradient
        # to the linear's weight, as it is frozen.
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=dtype))
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        linear_nf4(inp, nf4_tensor).sum().backward()
        assert inp.grad is not None and inp.grad.dtype == dtype
        assert nf4_tensor.grad is None

    @unittest.skipIf(not bnb_available, "Need bnb availble")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        TORCH_VERSION_AT_LEAST_2_7, reason="Failing in CI"
    )  # TODO: fix this
    @skip_if_rocm("ROCm enablement in progress")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_reconstruction_qlora_vs_bnb(self, dtype: torch.dtype):
        # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65C1-L81C47
        torch.manual_seed(0)
        device = "cuda"
        embed_dim = 512
        input_weight = _build_input_weight(embed_dim, device, dtype)
        nf4_weight = to_nf4(input_weight)
        bnb_linear = _build_bnb_linear(input_weight, device)
        bnb_reconstruction = bnb_linear(
            torch.eye(embed_dim, embed_dim, dtype=dtype, device=device)
        )
        bnb_diff = (bnb_reconstruction.T - input_weight).abs().max()
        nugs_diff = (nf4_weight.get_original_weight() - input_weight).abs().max()
        # Since we are subtle different we assume that we both reconstruct with
        # a similar precision
        assert bnb_diff < 1
        assert nugs_diff < 1
        assert (nugs_diff - bnb_diff).abs() < 2e-1

    @unittest.skipIf(not bnb_available, "Need bnb availble")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skip_if_rocm("ROCm enablement in progress")
    @unittest.skipIf(
        TORCH_VERSION_AT_LEAST_2_7, reason="Failing in CI"
    )  # TODO: fix this
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_nf4_bnb_linear(self, dtype: torch.dtype):
        """
        This test ensures that nf4_linear is "no worse" than BNB by ensuring the
        error compared to a bf16 linear is not more than BNB's implementation.
        """
        torch.manual_seed(0)
        dim = 512
        device = "cuda"
        input_weight = _build_input_weight(dim, device, dtype)
        nf4_weight = to_nf4(input_weight)
        bnb_linear = _build_bnb_linear(input_weight, device)

        inp = torch.randn(2, 512, dtype=dtype, device="cuda")

        out_nf4 = linear_nf4(inp, nf4_weight).sum()
        out_bnb = bnb_linear(inp).sum()
        out_ref = F.linear(inp, input_weight).sum()

        err_bnb = (out_bnb - out_ref).abs().max()
        err_native = (out_nf4 - out_ref).abs().max()
        assert err_native < 0.5 * dim
        assert err_bnb < 0.5 * dim

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_load_from_state_dicts(self, dtype: torch.dtype):
        """Tests loading to and from different module state dicts"""
        input_tensor = torch.rand(64, device="cuda", dtype=dtype)
        base_mod = self.TestMod(input_tensor, 32, 2)

        dummy_dict = {"param": input_tensor}
        base_mod.load_state_dict(dummy_dict)

        assert base_mod.param.block_size == 32
        assert base_mod.param.scaler_block_size == 2

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_load_from_nf4_same_meta(self, dtype: torch.dtype):
        """Tests loading to and from different module state dicts"""
        input_tensor = torch.rand(64, dtype=dtype)
        base_mod = self.TestMod(input_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(input_tensor, 32, 2)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 32
        assert other_mod.param.scaler_block_size == 2

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_load_from_nf4_diff_meta(self, dtype: torch.dtype):
        """Tests loading to and from different module state dicts"""
        input_tensor = torch.rand(128, dtype=dtype)
        base_mod = self.TestMod(input_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(input_tensor, 64, 1)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 64
        assert other_mod.param.scaler_block_size == 1

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_to_copy(self, dtype: torch.dtype):
        input_tensor = torch.rand(128, device="cpu")
        input_tensor_nf4 = to_nf4(input_tensor, 32, 2)
        nf4_to_dtype = input_tensor_nf4.to(dtype)
        torch.testing.assert_allclose(input_tensor, nf4_to_dtype, atol=0.13, rtol=0.13)

        if torch.cuda.is_available():
            input_tensor = torch.rand(128, device="cuda")
            input_tensor_nf4 = to_nf4(input_tensor, 32, 2)
            nf4_to_dtype = input_tensor_nf4.to(dtype)
            torch.testing.assert_allclose(
                input_tensor, nf4_to_dtype, atol=0.13, rtol=0.13
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    def test_to_copy_device(self):
        input_tensor = torch.rand(128, device="cpu")
        t = to_nf4(input_tensor, 32, 2)
        assert t.device == torch.device("cpu")
        z = t.cuda()
        assert z.device.type == "cuda"  # Because the device could be cuda:0
        x = z.cpu()
        assert x.device == torch.device("cpu")

        input_tensor = torch.rand(128, device="cuda")
        t = to_nf4(input_tensor, 32, 2)
        assert t.device.type == "cuda"

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_to_dtype(self, dtype: torch.dtype):
        input_tensor = torch.rand(128, dtype=dtype)
        input_tensor_nf4 = to_nf4(input_tensor, 32, 2)
        assert type(input_tensor_nf4) is not torch.Tensor
        assert type(input_tensor_nf4.to(dtype)) is torch.Tensor
        assert input_tensor_nf4.to(dtype).dtype is dtype

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_smoketest_linear(self, dtype: torch.dtype):
        a = torch.randn(32, 32, dtype=dtype, device="cuda")
        a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)
        inp = torch.randn(2, 32, 32, dtype=a.dtype, device=a.device)
        _ = torch.nn.functional.linear(inp, a)
        _ = torch.nn.functional.linear(inp, a_nf4)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_smoketest_linear_compile(self, dtype: torch.dtype):
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() < (8, 0)
            and dtype == torch.bfloat16
        ):
            self.skipTest("test requires SM capability of at least (8, 0).")
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest("test requires 2.3.0 and above for tracing NF4Tensor")
        a = torch.randn(32, 32, dtype=dtype, device="cuda")
        a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)
        inp = torch.randn(2, 32, 32, dtype=a.dtype, device=a.device)
        _ = torch.compile(torch.nn.functional.linear, mode="max-autotune")(inp, a_nf4)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @parametrize("shape", [(16, 16), (32, 16)])
    @parametrize("chunk_size", [8, 16, 32])
    def test_chunk_size_equivalence(self, dtype: torch.dtype, shape, chunk_size):
        a = torch.randn(shape, device="cuda", dtype=dtype)
        with unittest.mock.patch("torchao.dtypes.nf4tensor.CHUNK_SIZE", chunk_size):
            nf4_patched = to_nf4(a, 16, 2)
        # This will be essentially no chunking since the numel is alot smaller than default chunk_size
        nf4_base = to_nf4(a, 16, 2)

        torch.testing.assert_close(nf4_patched.quantized_data, nf4_base.quantized_data)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_empty_like(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.rand(input_size, device="cuda"))
        new_tensor = torch.empty_like(nf4_tensor, device="cpu")
        self.assertTrue(isinstance(new_tensor, NF4Tensor))
        self.assertEqual(new_tensor.get_device(), -1)  # that it's on CPU
        self.assertEqual(new_tensor.size(), nf4_tensor.size())

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("compile", [False, True])
    def test_quantize_api(self, compile):
        nf4_linear = nn.Linear(512, 512, device="cuda")
        torchao.quantize_(nf4_linear, nf4_weight_only())
        assert isinstance(nf4_linear.weight, NF4Tensor)

        ref_linear = copy.deepcopy(nf4_linear)
        ref_linear.weight.data = ref_linear.weight.get_original_weight()  # dequantize

        if compile:
            nf4_linear.compile()
            ref_linear.compile()

        nf4_x = torch.randn(2, 512, device="cuda").requires_grad_()
        ref_x = nf4_x.detach().clone().requires_grad_()

        nf4_out = nf4_linear(nf4_x)
        ref_out = ref_linear(ref_x)
        self.assertEqual(nf4_out, ref_out)

        grad_out = torch.randn(2, 512, device="cuda")
        nf4_out.backward(grad_out)
        ref_out.backward(grad_out)
        self.assertEqual(nf4_x.grad, ref_x.grad)


class TestFSDPOps(TestCase):
    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_torch_chunk_valid(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        nf4_tensor = to_nf4(torch.randn(input_size))
        chunks = list(torch.chunk(nf4_tensor, num_chunks))
        self.assertEqual(len(chunks), num_chunks)
        if isinstance(input_size, int):
            expected_size0 = input_size // num_chunks
        else:
            expected_size0 = input_size[0] // num_chunks
        for chunk in chunks:
            self.assertEqual(chunk.size(0), expected_size0)

    @parametrize("input_size", [511 * 512, (511 * 512,), (511, 512)])
    def test_torch_chunk_invalid_divide(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        with self.assertRaisesRegex(
            AssertionError, "Number of scalers must be divisible by scaler block size"
        ):
            nf4_tensor = to_nf4(torch.randn(input_size))
            torch.chunk(nf4_tensor, num_chunks)

    @parametrize("input_size", [(512, 512, 512)])
    def test_torch_chunk_invalid_3d(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        with self.assertRaisesRegex(AssertionError, "expect input tensor dim <= 2"):
            nf4_tensor = to_nf4(torch.randn(input_size))
            torch.chunk(nf4_tensor, num_chunks)

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_new_zeros_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        nf4_tensor_zeros = nf4_tensor.new_zeros(input_size)
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(nf4_tensor_zeros, attr)
            self.assertEqual(torch.count_nonzero(inner_tensor), 0)
        expected_size = input_size if not isinstance(input_size, int) else (input_size,)
        self.assertEqual(nf4_tensor_zeros.size(), torch.Size(expected_size))

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_new_zeros_invalid(self, input_size: Union[Tuple[int], int]):
        if isinstance(input_size, int):
            new_size = input_size + 1
        elif len(input_size) == 1:
            new_size = (input_size[0] + 1,)
        else:
            new_size = (input_size[0] + 1, input_size[1])
        nf4_tensor = to_nf4(torch.randn(input_size))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.new_zeros\\(NF4Tensor\\) with new size"
        ):
            _ = nf4_tensor.new_zeros(new_size)

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_slice_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        orig_attrs, _ = nf4_tensor.__tensor_flatten__()
        orig_sizes = dict(
            [(attr, getattr(nf4_tensor, attr).size()) for attr in orig_attrs]
        )
        end_idx = input_size if isinstance(input_size, int) else input_size[0]
        sliced_tensor = nf4_tensor[:end_idx]
        self.assertEqual(nf4_tensor.size(), sliced_tensor.size())
        attrs, _ = sliced_tensor.__tensor_flatten__()
        for attr in attrs:
            orig_storage = getattr(nf4_tensor, attr).untyped_storage().data_ptr()
            sliced_tensor_inner = getattr(sliced_tensor, attr)
            self.assertEqual(
                sliced_tensor_inner.untyped_storage().data_ptr(), orig_storage
            )
            self.assertEqual(sliced_tensor_inner.size(), orig_sizes[attr])

    def test_tensor_slice_1d_invalid(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with customized step"
        ):
            nf4_tensor[..., ::2]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"
        ):
            nf4_tensor[1:]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with end"
        ):
            nf4_tensor[:2]

    def test_tensor_slice_2d_invalid(self):
        nf4_tensor = to_nf4(torch.randn((512, 512)))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with dim"
        ):
            nf4_tensor[:, :511]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"
        ):
            nf4_tensor[1:]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with end"
        ):
            nf4_tensor[:2]

    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_tensor_view_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        viewed_tensor = nf4_tensor.view(-1)
        self.assertEqual(viewed_tensor.dim(), 1)
        self.assertEqual(viewed_tensor.numel(), math.prod(input_size))
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(viewed_tensor, attr)
            self.assertEqual(inner_tensor.size(0), inner_tensor.numel())

    @parametrize("input_size", [(512, 512)])
    def test_tensor_2d_view_valid(self, input_size: Tuple[int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        viewed_tensor = nf4_tensor.view(input_size)
        self.assertEqual(viewed_tensor.dim(), 2)
        self.assertEqual(viewed_tensor.numel(), math.prod(input_size))
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(viewed_tensor, attr)
            self.assertEqual(inner_tensor.size(0), inner_tensor.numel())

    @parametrize("input_size", [(512 * 512,)])
    def test_tensor_view_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        if len(input_size) == 1:
            with self.assertRaisesRegex(
                NotImplementedError, "aten.view\\(NF4Tensor\\) with size"
            ):
                nf4_tensor.view(input_size)

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_as_strided_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        nf4_tensor_strided = torch.as_strided(
            nf4_tensor,
            nf4_tensor.size(),
            nf4_tensor.stride(),
            nf4_tensor.storage_offset(),
        )
        self.assertEqual(nf4_tensor_strided.size(), nf4_tensor.size())
        self.assertEqual(nf4_tensor_strided.stride(), nf4_tensor.stride())
        self.assertEqual(
            nf4_tensor_strided.storage_offset(), nf4_tensor.storage_offset()
        )
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor_orig = getattr(nf4_tensor, attr)
            inner_tensor_strided = getattr(nf4_tensor_strided, attr)
            self.assertEqual(inner_tensor_strided.size(), inner_tensor_orig.size())
            self.assertEqual(inner_tensor_strided.stride(), inner_tensor_orig.stride())
            self.assertEqual(
                inner_tensor_strided.storage_offset(),
                inner_tensor_orig.storage_offset(),
            )

    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_tensor_as_strided_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        if len(input_size) == 1:
            size = (input_size[0] - 1,)
        else:
            size = (input_size[0] - 1, input_size[1])
        with self.assertRaisesRegex(
            NotImplementedError, "aten.as_strided\\(NF4Tensor\\) different numel"
        ):
            torch.as_strided(
                nf4_tensor, size, nf4_tensor.stride(), nf4_tensor.storage_offset()
            )
        with self.assertRaisesRegex(
            NotImplementedError,
            "aten.as_strided\\(NF4Tensor\\) only support original storage offset",
        ):
            torch.as_strided(nf4_tensor, nf4_tensor.size(), nf4_tensor.stride(), 1)

        if len(input_size) == 2:
            with self.assertRaisesRegex(
                NotImplementedError,
                "aten.as_strided\\(NF4Tensor\\) only support continuous stride",
            ):
                stride = (nf4_tensor.stride()[1], nf4_tensor.stride()[0])
                torch.as_strided(
                    nf4_tensor, nf4_tensor.size(), stride, nf4_tensor.storage_offset()
                )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_pin_memory(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertFalse(nf4_tensor.is_pinned())

        nf4_tensor = nf4_tensor.pin_memory()
        self.assertTrue(nf4_tensor.is_pinned())

        nf4_tensor = to_nf4(torch.randn(512 * 512, device="cuda"))
        self.assertFalse(nf4_tensor.is_pinned())

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_to_cuda(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda", non_blocking=True)
        self.assertEqual(nf4_tensor.device.type, "cuda")
        self.assertEqual(type(nf4_tensor), NF4Tensor)
        nf4_tensor.get_original_weight()  # make sure we can dequantize

        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda")
        self.assertEqual(nf4_tensor.device.type, "cuda")
        self.assertEqual(type(nf4_tensor), NF4Tensor)
        nf4_tensor.get_original_weight()

        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda", torch.bfloat16)
        self.assertEqual(nf4_tensor.device.type, "cuda")
        self.assertEqual(nf4_tensor.dtype, torch.bfloat16)
        self.assertEqual(type(nf4_tensor), torch.Tensor)  # dequantized

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_to_cpu(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512, device="cuda"))
        nf4_tensor = nf4_tensor.cpu()
        self.assertEqual(nf4_tensor.device.type, "cpu")
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(nf4_tensor, attr)
            self.assertEqual(inner_tensor.device.type, "cpu")
        nf4_tensor.get_original_weight()  # make sure we can dequantize

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_to_module(self):
        linear = nn.Linear(512, 512, bias=False)
        linear.weight = nn.Parameter(
            to_nf4(linear.weight.detach()), requires_grad=False
        )
        linear.cuda()
        self.assertEqual(linear.weight.device.type, "cuda")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cuda")

        linear.cpu()
        self.assertEqual(linear.weight.device.type, "cpu")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cpu")

        linear = nn.Linear(512, 512, bias=False)
        linear.weight = nn.Parameter(
            to_nf4(linear.weight.detach()), requires_grad=False
        )
        linear.to("cuda")
        self.assertEqual(linear.weight.device.type, "cuda")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cuda")

        linear.to("cpu")
        self.assertEqual(linear.weight.device.type, "cpu")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_deepcopy(self, input_size: Union[Tuple[int], int]):
        nf4_orig = to_nf4(torch.randn(input_size, device="cuda"))
        nf4_clone = copy.deepcopy(nf4_orig)
        self.assertEqual(
            nf4_clone.get_original_weight(), nf4_orig.get_original_weight()
        )


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.register_parameter("weight", nn.Parameter(to_nf4(weight)))
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = linear_nf4(input=x, weight=self.weight)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


class TestQLoRA(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @pytest.mark.skipif(
        version.parse(torch.__version__).base_version < "2.4.0",
        reason="torch >= 2.4 required",
    )
    @skip_if_lt_x_gpu(2)
    def test_qlora_fsdp2(self):
        from torch.distributed._composable.fsdp import CPUOffloadPolicy, OffloadPolicy

        self.run_subtests(
            {
                "enable_activation_checkpointing": [False, True],
                "offload_policy": [
                    OffloadPolicy(),
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
            },
            self._test_qlora_fsdp2,
        )

    def _test_qlora_fsdp2(
        self,
        enable_activation_checkpointing: bool,
        offload_policy: "OffloadPolicy",  # noqa: F821
    ):
        from torch.distributed._composable.fsdp import fully_shard
        from torch.testing._internal.distributed._tensor.common_dtensor import (
            ModelArgs,
            Transformer,
            TransformerBlock,
        )

        batch_size = 3
        lora_r = 8
        lora_alpha = 16
        vocab_size = 1024
        seq_len = 64
        model_args = ModelArgs(
            n_layers=3,
            n_heads=4,
            dim=1024,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dropout_p=0,
        )
        torch.manual_seed(42)
        with torch.device("cuda"):
            base_model = Transformer(model_args)
            for layer in base_model.layers:
                # attention with lora adapters
                for attr in ["wq", "wk", "wv", "wo"]:
                    orig_linear = getattr(layer.attention, attr)
                    setattr(
                        layer.attention,
                        attr,
                        LoRALinear(
                            orig_linear.weight.shape[1],
                            orig_linear.weight.shape[0],
                            orig_linear.weight,
                            lora_r,
                            lora_alpha,
                        ),
                    )
                for attr in ["w1", "w2"]:
                    orig_linear = getattr(layer.feed_forward, attr)
                    setattr(
                        layer.feed_forward,
                        attr,
                        LoRALinear(
                            orig_linear.weight.shape[1],
                            orig_linear.weight.shape[0],
                            orig_linear.weight,
                            lora_r,
                            lora_alpha,
                        ),
                    )
        for name, param in base_model.named_parameters():
            param.requires_grad_(
                name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
            )
        if enable_activation_checkpointing:
            apply_activation_checkpointing(
                base_model, auto_wrap_policy=ModuleWrapPolicy({TransformerBlock})
            )
        base_optim = torch.optim.AdamW(base_model.parameters(), lr=1e-2)

        fsdp_kwargs = {"offload_policy": offload_policy}
        fsdp_model = copy.deepcopy(base_model)
        for m in fsdp_model.modules():
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m, **fsdp_kwargs)
            else:
                if isinstance(m, TransformerBlock):
                    fully_shard(m, **fsdp_kwargs)
        fully_shard(fsdp_model, **fsdp_kwargs)
        fsdp_optim = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            inp = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            fsdp_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = fsdp_model(inp).sum()
            fsdp_loss.backward()
            fsdp_optim.step()

            base_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            base_loss = base_model(inp).sum()
            base_loss.backward()
            for param in base_model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG
                    )
            base_optim.step()
            self.assertEqual(fsdp_loss, base_loss)


class TestComm(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_comm(self):
        self.run_subtests(
            {"input_size": [512, 2048]},
            self._test_comm,
        )

    def _test_comm(self, input_size: int):
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed._tensor import distribute_tensor

        model = nn.Linear(input_size, input_size, device="cuda")
        origin_tensor = model.weight
        origin_nf4_tensor = to_nf4(origin_tensor)
        model = fully_shard(model)
        sharded_tensor = model.weight
        sharded_origin_nf4_tensor = distribute_tensor(
            origin_nf4_tensor,
            sharded_tensor.device_mesh,
            sharded_tensor.placements,
        )

        sharded_nf4_detach = sharded_origin_nf4_tensor.detach()
        resumed_full_tensor = sharded_nf4_detach.full_tensor()

        self.assertEqual(
            origin_nf4_tensor.get_original_weight(),
            resumed_full_tensor.get_original_weight(),
        )


instantiate_parametrized_tests(TestNF4Linear)
instantiate_parametrized_tests(TestFSDPOps)

if __name__ == "__main__":
    run_tests()
