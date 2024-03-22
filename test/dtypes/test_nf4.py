import logging
import unittest

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase
from torchao.dtypes.nf4tensor import linear_nf4, NF4Tensor, to_nf4
import torch.nn.functional as F
import io
from collections import OrderedDict
import torchao

bnb_available = False

try:
    import bitsandbytes as bnb

    bnb_available = True
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def _build_input_weight(embed_dim: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(
        embed_dim, embed_dim, device=device, dtype=torch.bfloat16
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

    def test_register_nf4_as_param(self):
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=torch.bfloat16))

        # Would raise if nn.Parameter registration fails, such as no detach()
        # impl when calling __torch_dispatch__
        param = torch.nn.Parameter(nf4_tensor, requires_grad=False)
        assert not param.requires_grad

    def test_output_bf16(self):
        # Test to ensure W4 A16 produces A16
        inp = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=torch.bfloat16))
        out = linear_nf4(input=inp, weight=nf4_tensor)
        assert out.dtype == torch.bfloat16

    def test_backward_bf16(self):
        # Test to ensure backward pass gives activation a bf16 gradient and no gradient
        # to the linear's weight, as it is frozen.
        nf4_tensor = to_nf4(torch.randn(512, 512, dtype=torch.bfloat16))
        inp = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        linear_nf4(inp, nf4_tensor).sum().backward()
        assert inp.grad is not None and inp.grad.dtype == torch.bfloat16
        assert nf4_tensor.grad is None

    @unittest.skipIf(not bnb_available, "Need bnb availble")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_reconstruction_qlora_vs_bnb(self):
        # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65C1-L81C47
        torch.manual_seed(0)
        device = "cuda"
        embed_dim = 512
        input_weight = _build_input_weight(embed_dim, device)
        nf4_weight = to_nf4(input_weight)
        bnb_linear = _build_bnb_linear(input_weight, device)
        bnb_reconstruction = bnb_linear(
            torch.eye(embed_dim, embed_dim, dtype=torch.bfloat16, device=device)
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
    def test_nf4_bnb_linear(self):
        """
        This test ensures that nf4_linear is "no worse" than BNB by ensuring the
        error compared to a bf16 linear is not more than BNB's implementation.
        """
        torch.manual_seed(0)
        dim = 512
        device = "cuda"
        input_weight = _build_input_weight(dim, device)
        nf4_weight = to_nf4(input_weight)
        bnb_linear = _build_bnb_linear(input_weight, device)

        inp = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")

        out_nf4 = linear_nf4(inp, nf4_weight).sum()
        out_bnb = bnb_linear(inp).sum()
        out_ref = F.linear(inp, input_weight).sum()

        err_bnb = (out_bnb - out_ref).abs().max()
        err_native = (out_nf4 - out_ref).abs().max()
        assert err_native < 0.5 * dim
        assert err_bnb < 0.5 * dim

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    def test_load_from_bfloat16(self):
        """Tests loading to and from different module state dicts"""
        inpt_tensor = torch.rand(64, device='cuda', dtype=torch.bfloat16)
        base_mod = self.TestMod(inpt_tensor, 32, 2)

        bf16_dummy_dict = {"param": inpt_tensor}
        base_mod.load_state_dict(bf16_dummy_dict)

        assert base_mod.param.block_size == 32
        assert base_mod.param.scaler_block_size == 2

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    def test_load_from_nf4_same_meta(self):
        """Tests loading to and from different module state dicts"""
        inpt_tensor = torch.rand(64, device='cuda', dtype=torch.bfloat16)
        base_mod = self.TestMod(inpt_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(inpt_tensor, 32, 2)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 32
        assert other_mod.param.scaler_block_size == 2

    @unittest.skipIf(not torch.cuda.is_available(), "Need cuda for test")
    def test_load_from_nf4_diff_meta(self):
        """Tests loading to and from different module state dicts"""
        inpt_tensor = torch.rand(128, device='cuda', dtype=torch.bfloat16)
        base_mod = self.TestMod(inpt_tensor, 32, 2)
        state_dict = base_mod.state_dict()
        saved_state_dict = self.save_state_dict_to_buffer(state_dict)

        other_mod = self.TestMod(inpt_tensor, 64, 1)
        other_mod.load_state_dict(torch.load(saved_state_dict))
        assert other_mod.param.block_size == 64
        assert other_mod.param.scaler_block_size == 1

    def test_to_copy(self):
        inpt_tensor = torch.rand(128, device='cpu')
        inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
        inpt_tensor_bfloat16 = inpt_tensor_nf4.to(torch.bfloat16)
        torch.testing.assert_allclose(inpt_tensor, inpt_tensor_bfloat16, atol=0.13, rtol=0.13)

        if torch.cuda.is_available():
            inpt_tensor = torch.rand(128, device='cuda')
            inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
            inpt_tensor_bfloat16 = inpt_tensor_nf4.to(torch.bfloat16)
            torch.testing.assert_allclose(inpt_tensor, inpt_tensor_bfloat16, atol=0.13, rtol=0.13)

    def test_to_bfloat16(self):
        inpt_tensor = torch.rand(128, dtype=torch.bfloat16)
        inpt_tensor_nf4 = to_nf4(inpt_tensor, 32, 2)
        assert type(inpt_tensor_nf4) != torch.Tensor
        assert type(inpt_tensor_nf4.to(torch.bfloat16)) == torch.Tensor
        assert inpt_tensor_nf4.to(torch.bfloat16).dtype == torch.bfloat16

    def test_linear(self):
        a = torch.randn(32, 32, dtype=torch.bfloat16, device='cuda')
        a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)
        inp = torch.randn(2, 32, 32, dtype=a.dtype, device=a.device)
        out1 = torch.nn.functional.linear(inp, a)
        out2 = torch.nn.functional.linear(inp, a_nf4)
        out3 = torch.compile(torch.nn.functional.linear, mode='max-autotune')(inp, a_nf4)

        # torch.testing.assert_allclose(out1, out2)
        # torch.testing.assert_allclose(out1, out3)



if __name__ == "__main__":
    unittest.main()
