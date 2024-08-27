import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

from torchao.prototype.low_bit_optim import _AdamW
from torchao.prototype.quantized_training import (
    int8_weight_only_quantized_training,
    int8_mixed_precision_training,
    quantize_int8_rowwise,
    Int8MixedPrecisionConfig,
)
from torchao.quantization.quant_api import quantize_
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Requires torch>=2.4", allow_module_level=True)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reset():
    # using TF32 will cause mixed mm to segfault with triton backend
    # fixed in nightly by https://github.com/pytorch/pytorch/pull/133173
    # also required for correctness check
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()


# we always use `quantize_(set_inductor_config=False)` to reduce compile time in CI.
class TestQuantizedTraining(TestCase):
    @parametrize("device", _DEVICES)
    def test_int8_stochastic_rounding(self, device):
        x = torch.randn(32, device=device)
        x_samples = x.view(1, -1).repeat(100_000, 1)

        x_int8, x_scale = quantize_int8_rowwise(x_samples, stochastic_rounding=True)
        x_dequant_samples = x_int8 * x_scale.view(-1, 1)
        x_dequant_mean = x_dequant_samples.mean(0)

        # a more rigorous test would be to do a hypothesis testing.
        # due to the statistical nature, this assertion may still fail, though very rarely.
        torch.testing.assert_close(x_dequant_mean, x, atol=1e-4, rtol=1e-4)

    @staticmethod
    def _forward_and_backward(module, input, grad):
        # clone input, since we want to inspect its gradient later
        input = input.detach().clone().requires_grad_(True)
        output = module(input)
        output.backward(grad)
        return input, output

    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_weight_only_correctness(self, leading_dims, bias, device):
        _reset()
        embed_dim = 32

        linear_fp32 = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        linear_int8 = copy.deepcopy(linear_fp32)
        quantize_(linear_int8, int8_weight_only_quantized_training(), set_inductor_config=False)
        linear_fp32.weight.data = linear_int8.weight.data.dequantize()

        input = torch.randn(leading_dims + (embed_dim,), device=device)
        grad = torch.randn(leading_dims + (embed_dim,), device=device)

        input_fp32, out_fp32 = self._forward_and_backward(linear_fp32, input, grad)
        input_int8, out_int8 = self._forward_and_backward(linear_int8, input, grad)

        torch.testing.assert_close(out_fp32, out_int8)
        torch.testing.assert_close(input_fp32.grad, input_int8.grad)
        torch.testing.assert_close(linear_fp32.weight.grad, linear_int8.weight.grad)
        if bias:
            torch.testing.assert_close(linear_fp32.bias.grad, linear_int8.bias.grad)

    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_weight_only_compile(self, leading_dims, bias, device):
        _reset()
        embed_dim = 128

        linear_eager = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        quantize_(linear_eager, int8_weight_only_quantized_training(), set_inductor_config=False)
        linear_compiled = copy.deepcopy(linear_eager)
        linear_compiled.compile()

        input = torch.randn(leading_dims + (embed_dim,), device=device) * 10
        grad = torch.randn(leading_dims + (embed_dim,), device=device)

        input_eager, out_eager = self._forward_and_backward(linear_eager, input, grad)
        input_compiled, out_compiled = self._forward_and_backward(linear_compiled, input, grad)

        torch.testing.assert_close(out_eager, out_compiled)
        torch.testing.assert_close(input_eager.grad, input_compiled.grad)
        torch.testing.assert_close(linear_eager.weight.grad, linear_compiled.weight.grad)
        if bias:
            torch.testing.assert_close(linear_eager.bias.grad, linear_compiled.bias.grad)

    @parametrize("compile", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_weight_only_training(self, compile, device):
        _reset()
        bsize = 4
        embed_dim = 32
        n_classes = 10

        model_fp32 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 2, n_classes),
        ).to(device)
        model_int8 = copy.deepcopy(model_fp32)
        quantize_(model_int8, int8_weight_only_quantized_training(), set_inductor_config=False)

        if compile:
            model_fp32.compile()
            model_int8.compile()

        optim_fp32 = _AdamW(model_fp32.parameters())
        optim_int8 = _AdamW(model_int8.parameters())

        for _ in range(5):
            inputs = torch.randn(bsize, embed_dim, device=device)
            labels = torch.randint(n_classes, size=(bsize,), device=device)
            loss_fp32 = F.cross_entropy(model_fp32(inputs), labels)
            loss_int8 = F.cross_entropy(model_int8(inputs), labels)

            rel_error = abs(loss_int8.item() - loss_fp32.item()) / abs(loss_fp32.item())
            assert rel_error < 2e-3, rel_error

            loss_fp32.backward()
            optim_fp32.step()
            optim_fp32.zero_grad()

            loss_int8.backward()
            optim_int8.step()
            optim_int8.zero_grad()

    @parametrize("compile", [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_int8_mixed_precision_training(self, compile):
        _reset()
        bsize = 4
        embed_dim = 32
        device = "cuda"
        config = Int8MixedPrecisionConfig(True, True, True)

        # only use 1 matmul shape to reduce triton autotune time
        model_ref = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)
        model_int8mp = copy.deepcopy(model_ref)
        quantize_(model_int8mp, int8_mixed_precision_training(config), set_inductor_config=False)

        if compile:
            model_ref.compile()
            model_int8mp.compile()

        optim_ref = torch.optim.AdamW(model_ref.parameters())
        optim_int8mp = torch.optim.AdamW(model_int8mp.parameters())

        for i in range(5):
            inputs = torch.randn(bsize, embed_dim, device=device)
            labels = torch.randint(embed_dim, size=(bsize,), device=device)
            loss_ref = F.cross_entropy(model_ref(inputs), labels)
            loss_int8mp = F.cross_entropy(model_int8mp(inputs), labels)

            rel_error = abs(loss_int8mp.item() - loss_ref.item()) / abs(loss_ref.item())
            assert rel_error < 3e-2, (i, rel_error)

            loss_ref.backward()
            optim_ref.step()
            optim_ref.zero_grad()

            loss_int8mp.backward()
            optim_int8mp.step()
            optim_int8mp.zero_grad()


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self):
        # due to stochastic rounding, use a pretty large tolerance here
        self.run_subtests(
            dict(),
            self._test_fsdp2,
            quantize_fn=int8_weight_only_quantized_training(),
            tolerance=0.05,
        )

        # triton autotune takes too long. apply INT8 matmul to forward pass only.
        self.run_subtests(
            dict(),
            self._test_fsdp2,
            quantize_fn=int8_mixed_precision_training(Int8MixedPrecisionConfig(True, False, False)),
            tolerance=1e-6,
        )

    def _test_fsdp2(self, quantize_fn, tolerance):
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import fully_shard
        from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs, Transformer

        _reset()
        batch_size = 3
        vocab_size = 32
        seq_len = 64
        model_args = ModelArgs(
            n_layers=2,
            n_heads=2,
            dim=128,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dropout_p=0,
            weight_tying=False,  # INT8 mixed-precision will fail if weight_tying=True
        )
        torch.manual_seed(42)
        base_model = Transformer(model_args).cuda()
        quantize_(base_model, quantize_fn, set_inductor_config=False)
        fsdp_model = copy.deepcopy(base_model)

        for layer in fsdp_model.layers:
            fully_shard(layer)
        fully_shard(fsdp_model)

        base_optim = torch.optim.Adam(base_model.parameters(), lr=1e-2, foreach=False, fused=False)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2, foreach=False, fused=False)

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
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            base_optim.step()

            rel_error = (fsdp_loss - base_loss).abs() / base_loss.abs()
            assert rel_error < tolerance, (iter_idx, rel_error)


instantiate_parametrized_tests(TestQuantizedTraining)


if __name__ == "__main__":
    run_tests()
