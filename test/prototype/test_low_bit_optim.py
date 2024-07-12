import copy

import pytest
import torch
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointWrapper,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torchao.prototype import low_bit_optim
from torchao.prototype.low_bit_optim.quant_utils import quantize_8bit_with_qmap, quantize_4bit_with_qmap
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    import lpmm
except ImportError:
    lpmm = None

# for FSDP2 test
if TORCH_VERSION_AFTER_2_4:
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestQuantize(TestCase):
    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_correctness(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.rand(256, device=device).sort().values

        actual_codes, actual_scale = quantize_8bit_with_qmap(x, qmap, 256, implementation=1)
        expected_codes, expected_scale = quantize_8bit_with_qmap(x, qmap, 256, implementation=0)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_compile(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.rand(256, device=device).sort().values

        compiled_f = torch.compile(quantize_8bit_with_qmap, fullgraph=True)
        actual_codes, actual_scale = compiled_f(x, qmap, 256)
        expected_codes, expected_scale = quantize_8bit_with_qmap(x, qmap, 256)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_correctness(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.rand(16, device=device).sort().values

        actual_codes, actual_scale = quantize_4bit_with_qmap(x, qmap, 256, implementation=1)
        expected_codes, expected_scale = quantize_4bit_with_qmap(x, qmap, 256, implementation=0)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_compile(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.rand(16, device=device).sort().values

        compiled_f = torch.compile(quantize_4bit_with_qmap, fullgraph=True)
        actual_codes, actual_scale = compiled_f(x, qmap, 256)
        expected_codes, expected_scale = quantize_4bit_with_qmap(x, qmap, 256)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)


class TestOptim(TestCase):
    @pytest.mark.skipif(bnb is None, reason="bitsandbytes is not availablle")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 8-bit Adam only works for CUDA")
    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["Adam8bit", "AdamW8bit"])
    def test_optim_8bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        model2 = copy.deepcopy(model1)

        optim1 = getattr(bnb.optim, optim_name)(model1.parameters())
        optim2 = getattr(low_bit_optim, optim_name)(model2.parameters())

        for _ in range(2):
            x = torch.randn(4, 32, device=device)

            loss1 = model1(x).sum()
            loss1.backward()
            optim1.step()
            optim1.zero_grad()

            loss2 = model2(x).sum()
            loss2.backward()
            optim2.step()
            optim2.zero_grad()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(lpmm is None, reason="lpmm is not availablle")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="lpmm 4-bit Adam only works for CUDA")
    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["Adam4bit", "AdamW4bit"])
    def test_optim_4bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        model2 = copy.deepcopy(model1)

        # lpmm doesn't have Adam. use AdamW with no weight decay instead.
        if optim_name == "Adam4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters(), weight_decay=0)
        elif optim_name == "AdamW4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters())
        else:
            raise ValueError(f"Unsupported {optim_name} optimizer for lpmm")
        optim2 = getattr(low_bit_optim, optim_name)(model2.parameters())

        for _ in range(2):
            x = torch.randn(4, 32, device=device)

            loss1 = model1(x).sum()
            loss1.backward()
            optim1.step()
            optim1.zero_grad()

            loss2 = model2(x).sum()
            loss2.backward()
            optim2.step()
            optim2.zero_grad()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1, rtol=1e-5, atol=1e-5)

    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["AdamFp8", "AdamWFp8"])
    @parametrize("device", _DEVICES)
    def test_optim_fp8_smoke(self, optim_name, device):
        if device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("FP8 requires compute capability >= 8.9")

        model = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        optim = getattr(low_bit_optim, optim_name)(model.parameters())

        x = torch.randn(4, 32, device=device)
        loss = model(x).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @pytest.mark.skipif(not TORCH_VERSION_AFTER_2_4, reason="torch >= 2.4 required")
    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self):
        self.run_subtests(
            {
                "enable_activation_checkpointing": [False, True],
                "offload_policy": [
                    OffloadPolicy(),
                    # CPUOffloadPolicy(pin_memory=True),  # compile take too long -> test will timeout
                    # CPUOffloadPolicy(pin_memory=False),
                ],
            },
            self._test_fsdp2,
        )

    def _test_fsdp2(self, enable_activation_checkpointing, offload_policy):
        from torch.testing._internal.distributed._tensor.common_dtensor import (
            ModelArgs,
            Transformer,
            TransformerBlock,
        )

        batch_size = 3
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
        if enable_activation_checkpointing:
            apply_activation_checkpointing(base_model, auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}))
        base_optim = low_bit_optim.Adam8bit(base_model.parameters(), lr=1e-2)

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
        fsdp_optim = low_bit_optim.Adam8bit(fsdp_model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            inp = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            fsdp_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = fsdp_model(inp).mean()
            fsdp_loss.backward()
            fsdp_optim.step()

            base_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            base_loss = base_model(inp).mean()
            base_loss.backward()
            for param in base_model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
            base_optim.step()
            self.assertEqual(fsdp_loss, base_loss, atol=1e-5, rtol=1e-5)


instantiate_parametrized_tests(TestQuantize)
instantiate_parametrized_tests(TestOptim)


if __name__ == "__main__":
    run_tests()
