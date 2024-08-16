import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

from torchao.prototype.low_bit_optim import _AdamW
from torchao.prototype.quantized_training import Int8QTLinearWeight, int8_weight_only_quantized_training
from torchao.quantization.quant_api import quantize_
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

if not TORCH_VERSION_AFTER_2_3:
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

        x_int8, x_scale = Int8QTLinearWeight.quantize(x_samples, stochastic_rounding=True)
        x_dequant_samples = x_int8 * x_scale.view(-1, 1)
        x_dequant_mean = x_dequant_samples.mean(0)

        # a more rigorous test would be to do a hypothesis testing.
        # due to the statistical nature, this assertion may still fail, though very rarely.
        torch.testing.assert_close(x_dequant_mean, x, atol=1e-4, rtol=1e-4)

    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_linear(self, leading_dims, bias, device):
        _reset()
        embed_dim = 32

        linear_fp32 = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        linear_int8 = copy.deepcopy(linear_fp32)
        quantize_(linear_int8, int8_weight_only_quantized_training(), set_inductor_config=False)
        linear_fp32.weight.data = linear_int8.weight.data.dequantize()

        input_fp32 = torch.randn(leading_dims + (embed_dim,), device=device)
        input_int8 = input_fp32.clone()
        input_fp32.requires_grad_(True)
        input_int8.requires_grad_(True)

        # test forward
        out_fp32 = linear_fp32(input_fp32)
        out_int8 = linear_int8(input_int8)
        torch.testing.assert_close(out_fp32, out_int8)

        # test backward
        grad = torch.randn(leading_dims + (embed_dim,), device=device)
        out_fp32.backward(grad)
        out_int8.backward(grad)
        torch.testing.assert_close(input_fp32.grad, input_int8.grad)
        torch.testing.assert_close(linear_fp32.weight.grad, linear_int8.weight.grad)
        if bias:
            torch.testing.assert_close(linear_fp32.bias.grad, linear_int8.bias.grad)

    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_linear_compile(self, leading_dims, bias, device):
        _reset()
        embed_dim = 128

        linear_eager = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        quantize_(linear_eager, int8_weight_only_quantized_training(), set_inductor_config=False)
        linear_compiled = copy.deepcopy(linear_eager)
        linear_compiled.compile()

        input_eager = torch.randn(leading_dims + (embed_dim,), device=device) * 10
        input_compiled = input_eager.clone()
        input_eager.requires_grad_(True)
        input_compiled.requires_grad_(True)

        out_eager = linear_eager(input_eager)
        out_compiled = linear_compiled(input_compiled)
        torch.testing.assert_close(out_eager, out_compiled)

        grad = torch.randn(leading_dims + (embed_dim,), device=device)
        out_eager.backward(grad)
        out_compiled.backward(grad)
        torch.testing.assert_close(input_eager.grad, input_compiled.grad)
        torch.testing.assert_close(linear_eager.weight.grad, linear_compiled.weight.grad)
        if bias:
            torch.testing.assert_close(linear_eager.bias.grad, linear_compiled.bias.grad)

    @parametrize("compile", [False, True])
    @parametrize("device", _DEVICES)
    def test_int8_linear_training(self, compile, device):
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
        # don't set inductor flags to speed up CI time
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


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self):
        # FSDP2 + compiled quantized training fails with PyTorch 2.4
        compile_layer_choices = [False]
        if TORCH_VERSION_AFTER_2_4:
            compile_layer_choices.append(True)

        self.run_subtests(
            {"compile_layer": compile_layer_choices},
            self._test_fsdp2,
        )

    def _test_fsdp2(self, compile_layer):
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
        )
        torch.manual_seed(42)
        base_model = Transformer(model_args).cuda()
        quantize_(base_model, int8_weight_only_quantized_training(), set_inductor_config=False)
        fsdp_model = copy.deepcopy(base_model)

        if compile_layer:
            for layer in base_model.layers:
                layer.compile()

        for layer in fsdp_model.layers:
            if compile_layer:
                layer.compile()
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

            # due to stochastic rounding, use a pretty large tolerance here
            rel_error = (fsdp_loss - base_loss).abs() / base_loss.abs()
            assert rel_error < 0.05, rel_error


instantiate_parametrized_tests(TestQuantizedTraining)


if __name__ == "__main__":
    run_tests()
