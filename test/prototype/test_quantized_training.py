import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

from torchao.prototype.low_bit_optim import AdamW
from torchao.prototype.quantized_training import Int8QTLinearWeight, int8_weight_only_quantized_training
from torchao.quantization.quant_api import quantize_
from torchao.utils import TORCH_VERSION_AFTER_2_3

if not TORCH_VERSION_AFTER_2_3:
    pytest.skip("Requires torch>=2.4", allow_module_level=True)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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

    @parametrize("device", _DEVICES)
    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    def test_int8_linear_forward(self, leading_dims, bias, device):
        embed_dim = 32

        linear_fp32 = nn.Linear(embed_dim, embed_dim * 2, bias=bias, device=device)
        linear_int8 = copy.deepcopy(linear_fp32)
        quantize_(linear_int8, int8_weight_only_quantized_training())
        assert isinstance(linear_int8.weight, Int8QTLinearWeight)

        inputs = torch.randn(leading_dims + (embed_dim,), device=device)
        out_fp32 = linear_fp32(inputs)
        out_int8 = linear_int8(inputs)
        torch.testing.assert_close(out_fp32, out_int8, atol=1e-2, rtol=1e-2)

    @parametrize("device", _DEVICES)
    def test_int8_linear_backward(self, device):
        bsize = 4
        embed_dim = 32
        n_classes = 10

        model_fp32 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 2, n_classes),
        ).to(device)
        model_int8 = copy.deepcopy(model_fp32)
        quantize_(model_int8, int8_weight_only_quantized_training())

        inputs = torch.randn(bsize, embed_dim, device=device)
        labels = torch.randint(n_classes, size=(bsize,), device=device)
        F.cross_entropy(model_fp32(inputs), labels).backward()
        F.cross_entropy(model_int8(inputs), labels).backward()

        for p_fp32, p_int8 in zip(model_fp32.parameters(), model_int8.parameters()):
            torch.testing.assert_close(p_fp32.grad, p_int8.grad, atol=1e-3, rtol=1e-2)

    @parametrize("device", _DEVICES)
    def test_int8_linear_training(self, device):
        bsize = 4
        embed_dim = 32
        n_classes = 10

        model_fp32 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 2, n_classes),
        ).to(device)
        model_int8 = copy.deepcopy(model_fp32)
        quantize_(model_int8, int8_weight_only_quantized_training())

        optim_fp32 = AdamW(model_fp32.parameters())
        optim_int8 = AdamW(model_int8.parameters())

        for _ in range(2):
            inputs = torch.randn(bsize, embed_dim, device=device)
            labels = torch.randint(n_classes, size=(bsize,), device=device)
            F.cross_entropy(model_fp32(inputs), labels).backward()
            F.cross_entropy(model_int8(inputs), labels).backward()

            optim_fp32.step()
            optim_fp32.zero_grad()
            optim_int8.step()
            optim_int8.zero_grad()

            with torch.no_grad():
                for p_fp32, p_int8 in zip(model_fp32.parameters(), model_int8.parameters()):
                    torch.testing.assert_close(p_fp32, p_int8.dequantize(), atol=1e-2, rtol=1e-2)


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self):
        self.run_subtests(
            {
                "activation_checkpointing": [False, True],
                # "compile_layer": [False, True],
            },
            self._test_fsdp2,
        )

    def _test_fsdp2(self, activation_checkpointing, compile_layer):
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs, Transformer, TransformerBlock

        batch_size = 3
        vocab_size = 32
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
        base_model = Transformer(model_args).cuda()
        quantize_(base_model, int8_weight_only_quantized_training())
        if activation_checkpointing:
            policy = ModuleWrapPolicy({TransformerBlock})
            apply_activation_checkpointing(base_model, auto_wrap_policy=policy)
        fsdp_model = copy.deepcopy(base_model)

        if compile_layer:
            for layer in base_model.layers:
                layer.compile()

        for layer in fsdp_model.layers:
            if compile_layer:
                layer.compile()
            fully_shard(layer)
        fully_shard(fsdp_model)

        base_optim = AdamW(base_model.parameters(), lr=1e-2)
        fsdp_optim = AdamW(fsdp_model.parameters(), lr=1e-2)

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
