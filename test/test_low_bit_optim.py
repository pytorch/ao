# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from packaging.version import Version
from torchao import optim
from torchao.optim.quant_utils import (
    _fp32_to_bf16_sr,
    quantize_4bit_with_qmap,
    quantize_8bit_with_qmap,
)
from torchao.optim.subclass_4bit import OptimState4bit
from torchao.optim.subclass_8bit import OptimState8bit
from torchao.optim.subclass_fp8 import OptimStateFp8
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    get_available_devices,
)

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    import lpmm
except ImportError:
    lpmm = None

if torch.version.hip is not None:
    pytest.skip("Skipping the test in ROCm", allow_module_level=True)

_DEVICES = get_available_devices()


class TestQuantize(TestCase):
    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_correctness(self, device):
        x = torch.rand(32, 1024, device=device)
        qmap = torch.rand(256, device=device).sort().values

        actual = (x.unsqueeze(-1) - qmap).abs().argmin(-1).to(torch.uint8)
        expected = quantize_8bit_with_qmap(x, qmap)

        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_compile(self, device):
        x = torch.rand(32, 1024, device=device)
        qmap = torch.rand(256, device=device).sort().values

        compiled_f = torch.compile(quantize_8bit_with_qmap, fullgraph=True)
        actual = compiled_f(x, qmap)
        expected = quantize_8bit_with_qmap(x, qmap)

        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_correctness(self, device):
        x = torch.rand(32, 1024, device=device)
        qmap = torch.rand(16, device=device).sort().values

        actual = (x.unsqueeze(-1) - qmap).abs().argmin(-1).to(torch.uint8)
        expected = quantize_4bit_with_qmap(x, qmap)

        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_compile(self, device):
        x = torch.rand(32, 1024, device=device)
        qmap = torch.rand(16, device=device).sort().values

        compiled_f = torch.compile(quantize_4bit_with_qmap, fullgraph=True)
        actual = compiled_f(x, qmap)
        expected = quantize_4bit_with_qmap(x, qmap)

        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    @parametrize("compile", [False, True])
    def test_bf16_stochastic_round(self, device, compile):
        x = torch.rand(32, device=device) * 100
        x_rep = x.view(-1, 1).repeat(1, 100_000)

        func = torch.compile(
            _fp32_to_bf16_sr, fullgraph=True, dynamic=False, disable=not compile
        )
        x_rep_bf16 = func(x_rep)
        assert x_rep_bf16.dtype is torch.bfloat16

        # must cast BF16 tensor back to FP32 so that .mean() is accurate
        torch.testing.assert_close(x_rep_bf16.float().mean(1), x, atol=3e-5, rtol=3e-5)


class TestOptim(TestCase):
    @parametrize(
        "optim_name",
        ["Adam8bit", "AdamW8bit", "Adam4bit", "AdamW4bit", "AdamFp8", "AdamWFp8"],
    )
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("device", _DEVICES)
    @skip_if_rocm("ROCm enablement in progress")
    def test_optim_smoke(self, optim_name, dtype, device):
        if optim_name.endswith("Fp8") and device == "cuda":
            if not TORCH_VERSION_AT_LEAST_2_4:
                pytest.skip("FP8 CUDA requires PyTorch >= 2.4")
            if torch.cuda.get_device_capability() < (8, 9):
                pytest.skip("FP8 CUDA requires compute capability >= 8.9")

        model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 32))
        model.to(device=device, dtype=dtype)
        optimizer = getattr(optim, optim_name)(model.parameters())

        x = torch.randn(4, 32, device=device, dtype=dtype)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # test serialization. also test the case CUDA optim loads CPU state dict
        with tempfile.NamedTemporaryFile() as f:
            torch.save(optimizer.state_dict(), f.name)
            state_dict = torch.load(f.name, map_location="cpu")

        model2 = copy.deepcopy(model)
        optim2 = getattr(optim, optim_name)(model2.parameters())
        optim2.load_state_dict(state_dict)

        for _ in range(2):
            x = torch.randn(4, 32, device=device, dtype=dtype)

            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            model2(x).sum().backward()
            optim2.step()
            optim2.zero_grad()

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1)

    # aten.slice is required for dcp.load() when world size changes i.e. re-sharding
    # however, it's cumbersome to test it directly, since we would need to run distributed
    # test 2 times with different world size, and persist checkpoint across the 2 runs.
    # thus, we only test for the required op. note that future implementations of dcp.load()
    # may use other ops.
    @parametrize("subclass", [OptimState4bit, OptimState8bit, OptimStateFp8])
    @parametrize("shape", [(4096,), (256, 256)])
    @parametrize("device", _DEVICES)
    def test_subclass_slice(self, subclass, shape, device):
        if subclass == OptimStateFp8:
            if device == "cpu" and len(shape) > 1 and not TORCH_VERSION_AT_LEAST_2_5:
                pytest.skip("fill_cpu not implemented for Float8_e4m3fn for torch<2.5")
            if device == "cuda" and not TORCH_VERSION_AT_LEAST_2_4:
                pytest.skip("FP8 CUDA requires PyTorch >= 2.4")
            if device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
                pytest.skip("FP8 CUDA requires compute capability >= 8.9")

        tensor = subclass.zeros(shape, device=device)
        offset = shape[0] // 2

        torch.testing.assert_close(
            tensor.dequantize()[:offset], tensor[:offset].dequantize()
        )
        torch.testing.assert_close(
            tensor.dequantize()[offset : offset * 2],
            tensor[offset : offset * 2].dequantize(),
        )

    @pytest.mark.skipif(bnb is None, reason="bitsandbytes is not available")
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="bitsandbytes 8-bit Adam only works for CUDA",
    )
    @skip_if_rocm("ROCm enablement in progress")
    @parametrize("optim_name", ["Adam8bit", "AdamW8bit"])
    def test_optim_8bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128))
        model1.to(device)
        model2 = copy.deepcopy(model1)

        # https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/v0.44.0
        block_size = 256 if Version(bnb.__version__) >= Version("0.44.0") else 2048

        optim1 = getattr(bnb.optim, optim_name)(model1.parameters())
        optim2 = getattr(optim, optim_name)(model2.parameters(), block_size=block_size)

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

    # this will not run in CI because we can't install lpmm
    @pytest.mark.skipif(lpmm is None, reason="lpmm is not available")
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="lpmm 4-bit Adam only works for CUDA"
    )
    @parametrize("optim_name", ["Adam4bit", "AdamW4bit"])
    def test_optim_4bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128))
        model1.to(device)
        model2 = copy.deepcopy(model1)

        # lpmm doesn't have Adam. use AdamW with no weight decay instead.
        if optim_name == "Adam4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters(), weight_decay=0)
        elif optim_name == "AdamW4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters())
        else:
            raise ValueError(f"Unsupported {optim_name} optimizer for lpmm")
        optim2 = getattr(optim, optim_name)(model2.parameters())

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

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        reason="optim CPU offload requires CUDA or XPU",
    )
    @parametrize("offload_grad,grad_accum", [(False, 1), (False, 2), (True, 1)])
    def test_optim_cpu_offload_correctness(self, offload_grad, grad_accum):
        device = _DEVICES[-1]
        # The first two layers are chosen so that they have a terrible arithmetic density.
        # this means long transfers and comparatively quick computation, increasing the chances
        # that missing synchronization will lead to test failures.
        # The third layer is very small, here to validate non-trainable parameters,
        # but shouldn't influence the timings
        model1 = nn.Sequential(
            nn.Linear(32, 131072),
            nn.ReLU(),
            nn.Linear(131072, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
        )
        model1.to(device)

        # make sure it can work in the presence of non-trainable params
        model1[2].requires_grad_(False)
        model2 = copy.deepcopy(model1)

        optim1 = torch.optim.AdamW(model1.parameters())
        optim2 = optim.CPUOffloadOptimizer(
            model2.parameters(),
            torch.optim.AdamW,
            offload_gradients=offload_grad,
        )

        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, 100)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, 100)

        rng = torch.Generator(device=device)
        rng.manual_seed(42)

        # make sure to run both models separately; otherwise, model1 gives additional
        # time for operations in model2 to complete, marking potential race conditions.
        for _ in range(2):
            for _ in range(grad_accum):
                x = torch.randn(4, 32, device=device, generator=rng)
                model1(x).sum().backward()

            optim1.step()
            optim1.zero_grad()
            scheduler1.step()

        # reset the rng
        rng.manual_seed(42)
        for _ in range(2):
            for _ in range(grad_accum):
                x = torch.randn(4, 32, device=device, generator=rng)
                model2(x).sum().backward()

            optim2.step()
            optim2.zero_grad()
            scheduler2.step()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1)

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        reason="optim CPU offload requires CUDA or XPU",
    )
    def test_optim_cpu_offload_save_load(self):
        device = _DEVICES[-1]
        # enable bias parameters so we have some small tensors that
        # are not offloaded.
        model1 = nn.Sequential(
            nn.Linear(32, 1024, bias=True), nn.ReLU(), nn.Linear(1024, 128, bias=True)
        )
        model1.to(device)
        optim1 = optim.CPUOffloadOptimizer(model1.parameters(), torch.optim.AdamW)

        for _ in range(2):
            x = torch.randn(4, 32, device=device)
            model1(x).sum().backward()
            optim1.step()
            optim1.zero_grad()

        # save checkpoint. make sure it can be serialized by torch.save()
        with tempfile.NamedTemporaryFile() as file:
            torch.save(optim1.state_dict(), file.name)
            state_dict = torch.load(file.name, map_location="cpu")

        # resume training
        model2 = copy.deepcopy(model1)
        optim2 = optim.CPUOffloadOptimizer(model2.parameters(), torch.optim.AdamW)
        optim2.load_state_dict(state_dict)

        for _ in range(2):
            x = torch.randn(4, 32, device=device)

            model1(x).sum().backward()
            optim1.step()
            optim1.zero_grad()

            model2(x).sum().backward()
            optim2.step()
            optim2.zero_grad()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1)

    def test_optim_bf16_stochastic_round_correctness(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(2024)
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128))
        model1.to(device)
        model2 = copy.deepcopy(model1).bfloat16()

        # small LR so that weight update is small
        # when bf16_stochastic_round=False, the test will fail after 1 iteration
        optim1 = torch.optim.AdamW(model1.parameters(), lr=1e-5)
        optim2 = optim._AdamW(
            model2.parameters(),
            lr=1e-5,
            bf16_stochastic_round=True,
        )

        # overfit on this sample
        x = torch.randn(4, 32, device=device)

        for idx in range(5):
            # mixed-precision training
            with torch.autocast(device, dtype=torch.bfloat16):
                loss1 = model1(x)
            loss1 = loss1.sum()  # under autocast context, bf16.sum() will return fp32
            loss1.backward()
            optim1.step()
            optim1.zero_grad()

            # full BF16 training with stochastic round weight update
            loss2 = model2(x.bfloat16()).sum()
            loss2.backward()
            optim2.step()
            optim2.zero_grad()

            torch.testing.assert_close(
                loss1, loss2, msg=lambda msg: f"Iteration {idx}. {msg}"
            )


_FSDP_WORLD_SIZE = 2


class TestFSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return _FSDP_WORLD_SIZE

    @pytest.mark.skipif(
        not TORCH_VERSION_AT_LEAST_2_5, reason="PyTorch>=2.5 is required."
    )
    @skip_if_lt_x_gpu(_FSDP_WORLD_SIZE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_fsdp2(self):
        optim_classes = [optim.AdamW8bit, optim.AdamW4bit]
        if torch.cuda.get_device_capability() >= (8, 9):
            optim_classes.append(optim.AdamWFp8)

        self.run_subtests(
            {"optim_cls": optim_classes},
            self._test_fsdp2,
        )

    def _test_fsdp2(self, optim_cls):
        import torch.distributed as dist
        import torch.distributed.checkpoint as dcp
        import torch.utils._pytree as pytree
        from torch.distributed.tensor import DTensor
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
            dim=512,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dropout_p=0,
        )
        torch.manual_seed(42)
        with torch.device("cuda"):
            base_model = Transformer(model_args)
        base_optim = optim_cls(base_model.parameters(), lr=1e-2)

        fsdp_model = copy.deepcopy(base_model)
        for m in fsdp_model.modules():
            if isinstance(m, TransformerBlock):
                fully_shard(m)
        fully_shard(fsdp_model)
        fsdp_optim = optim_cls(fsdp_model.parameters(), lr=1e-2)

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
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            base_optim.step()
            self.assertEqual(fsdp_loss, base_loss)

        base_param = base_optim.param_groups[0]["params"][0]
        base_exp_avg = base_optim.state[base_param]["exp_avg"]

        fsdp_param = fsdp_optim.param_groups[0]["params"][0]
        fsdp_exp_avg = fsdp_optim.state[fsdp_param]["exp_avg"]
        full_fsdp_exp_avg = fsdp_exp_avg.full_tensor()

        self.assertEqual(base_exp_avg.dequantize(), full_fsdp_exp_avg.dequantize())

        # test for compatibility with dcp.save() and .load()
        checkpoint_id = f"_fsdp_low_bit_optim_{optim_cls.__name__}"
        if Path(checkpoint_id).exists():
            shutil.rmtree(checkpoint_id)
        dcp.save(fsdp_optim.state_dict(), checkpoint_id=checkpoint_id)

        # normally we would want to use dcp.state_dict.get_optimizer_state_dict() to initialize optim states.
        # however, currently it does not respect tensor-ness of LR pytorch/pytorch#139575.
        # therefore, we have to manually initialize optim state here.
        resumed_fsdp_optim = optim_cls(fsdp_model.parameters(), lr=1e-2)
        for p in fsdp_model.parameters():
            p.grad = torch.zeros_like(p)

        # this will change model weights due to weight decay, but since we don't use the model anymore, it's fine.
        resumed_fsdp_optim.step()

        dcp.load(resumed_fsdp_optim.state_dict(), checkpoint_id=checkpoint_id)
        if dist.get_rank() == 0:
            shutil.rmtree(checkpoint_id)

        subclasses = (OptimState4bit, OptimState8bit, OptimStateFp8)

        for v1, v2 in zip(
            pytree.tree_iter(resumed_fsdp_optim.state_dict()),
            pytree.tree_iter(fsdp_optim.state_dict()),
        ):
            assert v1.__class__ == v2.__class__, (v1.__class__, v2.__class__)
            if isinstance(v1, DTensor):
                v1 = v1.to_local()
                v2 = v2.to_local()
                assert v1.__class__ == v2.__class__, (v1.__class__, v2.__class__)
            if isinstance(v1, subclasses):
                v1 = v1.dequantize()
                v2 = v2.dequantize()
            self.assertEqual(v1, v2)

    @pytest.mark.skipif(
        not TORCH_VERSION_AT_LEAST_2_5, reason="PyTorch>=2.5 is required."
    )
    @skip_if_lt_x_gpu(_FSDP_WORLD_SIZE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_uneven_shard(self):
        in_dim = 512
        out_dim = _FSDP_WORLD_SIZE * 16 + 1

        # 1st dim of linear weight will not be divisible by WORLD_SIZE
        model = nn.Linear(in_dim, out_dim, device="cuda")
        assert model.weight.shape[0] % _FSDP_WORLD_SIZE != 0
        fully_shard(model)

        # currently all of our low-bit Adam/AdamW share the same implementation.
        # thus, we only need to test for 1 optimizer class.
        optimizer = optim.AdamW8bit(model.parameters())

        for _ in range(2):
            inputs = torch.randn(2, in_dim, device="cuda")
            model(inputs).sum().backward()
            optimizer.step()
            optimizer.zero_grad()


instantiate_parametrized_tests(TestQuantize)
instantiate_parametrized_tests(TestOptim)


if __name__ == "__main__":
    run_tests()
