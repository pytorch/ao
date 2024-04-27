import logging
import unittest
from packaging import version
import math

import torch
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.nf4tensor import linear_nf4, NF4Tensor, to_nf4
import torch.nn.functional as F
import io
from collections import OrderedDict
import torchao
from typing import Tuple, Union


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
    input_weight = torch.empty(
        embed_dim, embed_dim, device=device, dtype=dtype
    )
    input_weight.normal_(0, 1)
    return input_weight

def _build_bnb_linear(input_weight, device):
    assert bnb_available, "Needs bitsandbytes support"
    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type="nf4"
    ).cuda(device)
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
            self.param = torch.nn.Parameter(to_nf4(tensor, block_size, scaler_block_size))

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
    def test_output_dtype_match(self, dtype:torch.dtype):
        # Test to ensure W4 A16 produces A16
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=dtype))
        out = linear_nf4(input=inp, weight=nf4_tensor)
        assert out.dtype == dtype

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_backward_dtype_match(self, dtype:torch.dtype):
        # Test to ensure backward pass gives activation a bf16 gradient and no gradient
        # to the linear's weight, as it is frozen.
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=dtype))
        inp = torch.randn(2, 512, dtype=dtype, requires_grad=True)
        linear_nf4(inp, nf4_tensor).sum().backward()
        assert inp.grad is not None and inp.grad.dtype == dtype
        assert nf4_tensor.grad is None

    @unittest.skipIf(not bnb_available, "Need bnb availble")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
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
        inpt_tensor = torch.rand(64, device='cuda', dtype=dtype)
        base_mod = self.TestMod(inpt_tensor, 32, 2)

        dummy_dict = {"param": inpt_tensor}
        base_mod.load_state_dict(dummy_dict)

        assert base_mod.param.block_size == 32
        assert base_mod.param.scaler_block_size == 2

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_load_from_nf4_same_meta(self, dtype: torch.dtype):
        """Tests loading to and from different module state dicts"""
        inpt_tensor = torch.rand(64, device='cuda', dtype=dtype)
        base_mod = self.TestMod(inpt_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(inpt_tensor, 32, 2)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 32
        assert other_mod.param.scaler_block_size == 2

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_load_from_nf4_diff_meta(self, dtype: torch.dtype):
        """Tests loading to and from different module state dicts"""
        inpt_tensor = torch.rand(128, device='cuda', dtype=dtype)
        base_mod = self.TestMod(inpt_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(inpt_tensor, 64, 1)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 64
        assert other_mod.param.scaler_block_size == 1

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_to_copy(self, dtype: torch.dtype):
        inpt_tensor = torch.rand(128, device='cpu')
        inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
        nf4_to_dtype = inpt_tensor_nf4.to(dtype)
        torch.testing.assert_allclose(inpt_tensor, nf4_to_dtype, atol=0.13, rtol=0.13)

        if torch.cuda.is_available():
            inpt_tensor = torch.rand(128, device='cuda')
            inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
            nf4_to_dtype = inpt_tensor_nf4.to(dtype)
            torch.testing.assert_allclose(inpt_tensor, nf4_to_dtype, atol=0.13, rtol=0.13)

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    def test_to_copy_device(self):
        inpt_tensor = torch.rand(128, device='cpu')
        t = to_nf4(inpt_tensor, 32, 2)
        assert t.device == torch.device('cpu')
        z = t.cuda()
        assert z.device.type == "cuda" # Because the device could be cuda:0
        x = z.cpu()
        assert x.device == torch.device('cpu')

        inpt_tensor = torch.rand(128, device='cuda')
        t = to_nf4(inpt_tensor, 32, 2)
        assert t.device.type == "cuda"

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_to_dtype(self, dtype: torch.dtype):
        inpt_tensor = torch.rand(128, dtype=dtype)
        inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
        assert type(inpt_tensor_nf4) != torch.Tensor
        assert type(inpt_tensor_nf4.to(dtype)) == torch.Tensor
        assert inpt_tensor_nf4.to(dtype).dtype == dtype

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_smoketest_linear(self, dtype: torch.dtype):
        a = torch.randn(32, 32, dtype=dtype, device='cuda')
        a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)
        inp = torch.randn(2, 32, 32, dtype=a.dtype, device=a.device)
        out1 = torch.nn.functional.linear(inp, a)
        out2 = torch.nn.functional.linear(inp, a_nf4)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_smoketest_linear_compile(self, dtype: torch.dtype):
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0) and dtype == torch.bfloat16:
            self.skipTest("test requires SM capability of at least (8, 0).")
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest("test requires 2.3.0 and above for tracing NF4Tensor")
        a = torch.randn(32, 32, dtype=dtype, device='cuda')
        a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)
        inp = torch.randn(2, 32, 32, dtype=a.dtype, device=a.device)
        out3 = torch.compile(torch.nn.functional.linear, mode='max-autotune')(inp, a_nf4)


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
        with self.assertRaisesRegex(AssertionError, "Number of scalers must be divisible by scaler block size"):
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
        for attr in ["quantized_scalers", "quantization_factor", "quantized_data"]:
            inner_tensor = getattr(nf4_tensor_zeros, attr)
            self.assertEqual(torch.count_nonzero(inner_tensor), 0)
        expected_size = input_size if not isinstance(input_size, int) else (input_size, )
        self.assertEqual(nf4_tensor_zeros.size(), torch.Size(expected_size))

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_new_zeros_invalid(self, input_size: Union[Tuple[int], int]):
        if isinstance(input_size, int):
            new_size = input_size + 1
        elif len(input_size) == 1:
            new_size = (input_size[0] + 1, )
        else:
            new_size = (input_size[0] + 1, input_size[1])
        nf4_tensor = to_nf4(torch.randn(input_size))
        with self.assertRaisesRegex(NotImplementedError, "aten.new_zeros\\(NF4Tensor\\) with new size"):
            nf4_tensor_zeros = nf4_tensor.new_zeros(new_size)

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_slice_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        orig_attrs, _ = nf4_tensor.__tensor_flatten__()
        orig_sizes = dict([(attr, getattr(nf4_tensor, attr).size()) for attr in orig_attrs])
        end_idx = input_size if isinstance(input_size, int) else input_size[0]
        sliced_tensor = nf4_tensor[:end_idx]
        self.assertEqual(nf4_tensor.size(), sliced_tensor.size())
        attrs, _ = sliced_tensor.__tensor_flatten__()
        for attr in attrs:
            orig_storage = getattr(nf4_tensor, attr).untyped_storage().data_ptr()
            sliced_tensor_inner = getattr(sliced_tensor, attr)
            self.assertEqual(sliced_tensor_inner.untyped_storage().data_ptr(), orig_storage)
            self.assertEqual(sliced_tensor_inner.size(), orig_sizes[attr])

    def test_tensor_slice_1d_invalid(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with step"):
            nf4_tensor[..., ::2]
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"):
            nf4_tensor[1:]
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with end "):
            nf4_tensor[:2]

    def test_tensor_slice_2d_invalid(self):
        nf4_tensor = to_nf4(torch.randn((512, 512)))
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with dim"):
            nf4_tensor[:, :511]
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"):
            nf4_tensor[1:]
        with self.assertRaisesRegex(NotImplementedError, "aten.slice\\(NF4Tensor\\) with end"):
            nf4_tensor[:2]

    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_tensor_view_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        viewed_tensor = nf4_tensor.view(-1)
        self.assertEqual(viewed_tensor.dim(), 1)
        self.assertEqual(viewed_tensor.numel(), math.prod(input_size))
        for attr in ["quantized_scalers", "quantization_factor", "quantized_data"]:
            inner_tensor = getattr(viewed_tensor, attr)
            self.assertEqual(inner_tensor.size(0), inner_tensor.numel())

    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_tensor_view_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        with self.assertRaisesRegex(NotImplementedError, "aten.view\\(NF4Tensor\\) with size"):
            nf4_tensor.view(input_size)
        with self.assertRaisesRegex(NotImplementedError, "aten.view\\(NF4Tensor\\) with size"):
            nf4_tensor.view(input_size)

    @parametrize("input_size", [512 * 512, (512 * 512,), (512, 512)])
    def test_tensor_as_strided_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        nf4_tensor_strided = torch.as_strided(nf4_tensor, nf4_tensor.size(), nf4_tensor.stride(), nf4_tensor.storage_offset())
        self.assertEqual(nf4_tensor_strided.size(), nf4_tensor.size())
        self.assertEqual(nf4_tensor_strided.stride(), nf4_tensor.stride())
        self.assertEqual(nf4_tensor_strided.storage_offset(), nf4_tensor.storage_offset())
        for attr in ["quantized_scalers", "quantization_factor", "quantized_data"]:
            inner_tensor_orig = getattr(nf4_tensor, attr)
            inner_tensor_strided = getattr(nf4_tensor_strided, attr)
            self.assertEqual(inner_tensor_strided.size(), inner_tensor_orig.size())
            self.assertEqual(inner_tensor_strided.stride(), inner_tensor_orig.stride())
            self.assertEqual(inner_tensor_strided.storage_offset(), inner_tensor_orig.storage_offset())


    @parametrize("input_size", [(512 * 512,), (512, 512)])
    def test_tensor_as_strided_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        if len(input_size) == 1:
            size = (input_size[0] - 1, )
        else:
            size = (input_size[0] - 1, input_size[1])
        with self.assertRaisesRegex(NotImplementedError, "aten.as_strided\\(NF4Tensor\\) different numel"):
            torch.as_strided(nf4_tensor, size, nf4_tensor.stride(), nf4_tensor.storage_offset())
        with self.assertRaisesRegex(NotImplementedError, "aten.as_strided\\(NF4Tensor\\) only support original storage offset"):
            torch.as_strided(nf4_tensor, nf4_tensor.size(), nf4_tensor.stride(), 1)

        if len(input_size) == 2:
            with self.assertRaisesRegex(NotImplementedError, "aten.as_strided\\(NF4Tensor\\) only support continuous stride"):
                stride = (nf4_tensor.stride()[1], nf4_tensor.stride()[0])
                torch.as_strided(nf4_tensor, nf4_tensor.size(), stride, nf4_tensor.storage_offset())

    def test_pin_memory(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertFalse(nf4_tensor.is_pinned())

        nf4_tensor = nf4_tensor.pin_memory()
        self.assertTrue(nf4_tensor.is_pinned())

        nf4_tensor = to_nf4(torch.randn(512 * 512, device='cuda'))
        self.assertFalse(nf4_tensor.is_pinned())

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_to_cuda(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda", non_blocking=True)
        self.assertEqual(nf4_tensor.device.type, "cuda")

        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda")
        self.assertEqual(nf4_tensor.device.type, "cuda")

        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertEqual(nf4_tensor.device.type, "cpu")
        nf4_tensor = nf4_tensor.to("cuda", torch.bfloat16)
        self.assertEqual(nf4_tensor.device.type, "cuda")
        self.assertEqual(nf4_tensor.dtype, torch.bfloat16)

    def test_to_cpu(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512, device='cuda'))
        nf4_tensor.cpu()

instantiate_parametrized_tests(TestNF4Linear)
instantiate_parametrized_tests(TestFSDPOps)

if __name__ == "__main__":
    run_tests()
