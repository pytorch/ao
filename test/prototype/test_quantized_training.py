import pytest

from torchao.utils import TORCH_VERSION_AT_LEAST_2_4, TORCH_VERSION_AT_LEAST_2_6

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Requires torch>=2.4", allow_module_level=True)

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

from torchao.prototype.low_bit_optim import _AdamW
from torchao.prototype.quantized_training import (
    Int8MixedPrecisionTrainingConfig,
    bitnet_training,
    int8_mixed_precision_training,
    int8_weight_only_quantized_training,
    quantize_int8_rowwise,
)
from torchao.quantization.quant_api import quantize_

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reset():
    # using TF32 will cause mixed mm to segfault with triton backend
    # fixed in nightly by https://github.com/pytorch/pytorch/pull/133173
    # also required for correctness check
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()


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
        quantize_(
            linear_int8,
            int8_weight_only_quantized_training(),
        )
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
        quantize_(
            linear_eager,
            int8_weight_only_quantized_training(),
        )
        linear_compiled = copy.deepcopy(linear_eager)
        linear_compiled.compile()

        input = torch.randn(leading_dims + (embed_dim,), device=device) * 10
        grad = torch.randn(leading_dims + (embed_dim,), device=device)

        input_eager, out_eager = self._forward_and_backward(linear_eager, input, grad)
        input_compiled, out_compiled = self._forward_and_backward(
            linear_compiled, input, grad
        )

        torch.testing.assert_close(out_eager, out_compiled)
        torch.testing.assert_close(input_eager.grad, input_compiled.grad)
        torch.testing.assert_close(
            linear_eager.weight.grad, linear_compiled.weight.grad
        )
        if bias:
            torch.testing.assert_close(
                linear_eager.bias.grad, linear_compiled.bias.grad
            )

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
        quantize_(model_int8, int8_weight_only_quantized_training())

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
    @parametrize(
        "config",
        [
            Int8MixedPrecisionTrainingConfig(),
            Int8MixedPrecisionTrainingConfig(output=False),
            Int8MixedPrecisionTrainingConfig(grad_input=False),
            Int8MixedPrecisionTrainingConfig(grad_weight=False),
        ],
    )
    @parametrize("module_swap", [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_int8_mixed_precision_training(self, compile, config, module_swap):
        _reset()
        bsize = 64
        embed_dim = 64
        device = "cuda"

        linear = nn.Linear(embed_dim, embed_dim, device=device)
        linear_int8mp = copy.deepcopy(linear)
        config.module_swap = module_swap
        apply_func = int8_mixed_precision_training(config)
        quantize_(linear_int8mp, apply_func)

        if compile:
            linear.compile()
            linear_int8mp.compile()

        inputs = torch.randn(bsize, embed_dim, device=device)
        grad_outputs = torch.randn(bsize, embed_dim, device=device)

        inputs_ref, outputs_ref = self._forward_and_backward(
            linear, inputs, grad_outputs
        )
        inputs_int8mp, outputs_int8mp = self._forward_and_backward(
            linear_int8mp, inputs, grad_outputs
        )

        def snr(ref, actual):
            error = actual - ref
            return 20 * torch.log10(ref.norm() / error.norm())

        assert snr(outputs_ref, outputs_int8mp) > 20
        assert snr(inputs_ref.grad, inputs_int8mp.grad) > 20
        assert snr(linear.weight.grad, linear_int8mp.weight.grad) > 20

    @pytest.mark.skip("Flaky on CI")
    @parametrize("compile", [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bitnet_training(self, compile):
        # reference implementation
        # https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
        # Figure 3
        class BitLinear(nn.Linear):
            def activation_quant(self, x):
                scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(
                    min=1e-5
                )
                return (x * scale).round().clamp_(-128, 127) / scale

            def weight_quant(self, x):
                scale = 1.0 / x.abs().mean().clamp_(min=1e-5)
                return (x * scale).round().clamp_(-1, 1) / scale

            def forward(self, x):
                w = self.weight
                x = x + (self.activation_quant(x) - x).detach()
                w = w + (self.weight_quant(w) - w).detach()
                return F.linear(x, w, self.bias)

        _reset()
        bsize = 4
        embed_dim = 32
        device = "cuda"

        # only use 1 matmul shape to reduce triton autotune time
        model_ref = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)
        model = copy.deepcopy(model_ref)
        quantize_(model, bitnet_training())

        # change model_ref to use BitLinear
        model_ref[0].__class__ = BitLinear
        model_ref[2].__class__ = BitLinear

        if compile:
            model_ref.compile()
            model.compile()

        optim_ref = torch.optim.AdamW(model_ref.parameters())
        optim = torch.optim.AdamW(model.parameters())

        for i in range(5):
            inputs = torch.randn(bsize, embed_dim, device=device)
            labels = torch.randint(embed_dim, size=(bsize,), device=device)
            loss_ref = F.cross_entropy(model_ref(inputs), labels)
            loss = F.cross_entropy(model(inputs), labels)

            torch.testing.assert_close(loss, loss_ref)

            loss_ref.backward()
            optim_ref.step()
            optim_ref.zero_grad()

            loss.backward()
            for p in model.parameters():
                assert p.grad is not None
            optim.step()
            optim.zero_grad()


_FSDP_WORLD_SIZE = 2


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return _FSDP_WORLD_SIZE

    @skip_if_lt_x_gpu(_FSDP_WORLD_SIZE)
    def test_fsdp2_correctness(self):
        mp_policy = MixedPrecisionPolicy()

        # quantize_fn, mp_policy, tolerance
        test_args = [
            # high tolerance due to stochastic rounding
            (int8_weight_only_quantized_training(), mp_policy, 0.05),
            (int8_mixed_precision_training(), mp_policy, 1e-6),
            (int8_mixed_precision_training(module_swap=True), mp_policy, 1e-6),
            (bitnet_training(), mp_policy, 1e-5),
        ]

        # FSDP2 mixed-precision requires https://github.com/pytorch/pytorch/pull/136129
        if TORCH_VERSION_AT_LEAST_2_6:
            # It's complicated (though possible) to simulate FSDP BF16 mixed-precision for base_model.
            # We would need to cast all params to BF16 in forward and backward pass, while keeping
            # the params in FP32 for optim step.
            # torch.autocast() will only do this for F.linear() layer (and its backward).
            # To keep it simple, we just use a larger tolerance here.
            bf16_mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

            extra_args = [
                (int8_weight_only_quantized_training(), bf16_mp_policy, 1e-2),
                (int8_mixed_precision_training(), bf16_mp_policy, 1e-2),
                (bitnet_training(), bf16_mp_policy, 1e-2),
            ]
            test_args.extend(extra_args)

        self.run_subtests({"args": test_args}, self._run_subtest)

    def _run_subtest(self, args):
        quantize_fn, mp_policy, tolerance = args

        batch_size = 3
        vocab_size = 32
        seq_len = 64

        # NOTE: if weight_tying=True and we also quantize LM head, INT8 mixed-precision will fail.
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
        fsdp_model = copy.deepcopy(base_model)

        quantize_(base_model.layers, quantize_fn)
        quantize_(fsdp_model.layers, quantize_fn)

        for layer in fsdp_model.layers:
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(fsdp_model, mp_policy=mp_policy)

        # start testing
        base_optim = torch.optim.Adam(
            base_model.parameters(), lr=1e-2, foreach=False, fused=False
        )
        fsdp_optim = torch.optim.Adam(
            fsdp_model.parameters(), lr=1e-2, foreach=False, fused=False
        )

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            inp = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            fsdp_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = fsdp_model(inp).sum()
            fsdp_loss.backward()
            for param in fsdp_model.parameters():
                assert param.grad is not None
            fsdp_optim.step()

            base_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            base_loss = base_model(inp).sum()
            base_loss.backward()
            for param in base_model.parameters():
                assert param.grad is not None
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            base_optim.step()

            rel_error = (fsdp_loss - base_loss).abs() / base_loss.abs()
            assert rel_error < tolerance, (
                quantize_fn.__name__,
                mp_policy,
                iter_idx,
                rel_error,
            )

    @skip_if_lt_x_gpu(_FSDP_WORLD_SIZE)
    def test_precompute_bitnet_scale(self):
        from torchao.prototype.quantized_training.bitnet import (
            get_bitnet_scale,
            precompute_bitnet_scale_for_fsdp,
        )

        model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)).cuda()
        model_fsdp = copy.deepcopy(model)
        quantize_(model_fsdp, bitnet_training())
        fully_shard(model_fsdp)

        precompute_bitnet_scale_for_fsdp(model_fsdp)

        torch.testing.assert_close(
            get_bitnet_scale(model[0].weight),
            model_fsdp[0].weight._local_tensor._precomputed_scale,
        )
        torch.testing.assert_close(
            get_bitnet_scale(model[2].weight),
            model_fsdp[2].weight._local_tensor._precomputed_scale,
        )


instantiate_parametrized_tests(TestQuantizedTraining)


if __name__ == "__main__":
    run_tests()
