import copy
from typing import Optional

import pytest

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, is_sm_at_least_89

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    swap_linear_layers,
)
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import GemmInputRole
from torchao.testing.float8.fsdp2_utils import check_parity_fp8_comm_only

if not is_sm_at_least_89():
    pytest.skip("Unsupported CUDA device capability version", allow_module_level=True)


class Float8CommTestLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fp8_param = hp_tensor_to_float8_dynamic(
            self.weight,
            torch.float8_e4m3fn,
            None,  # mm_linear_config,
            reduce_amax=False,
            gemm_input_role=GemmInputRole.WEIGHT,
        )
        weight_orig = fp8_param.to_original_precision()
        output = torch.matmul(input, weight_orig.t())
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(
        cls,
        mod,
    ):
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def convert_to_float8_comm_test_layers(
    module: nn.Module,
) -> nn.Module:
    from_float = lambda m: Float8CommTestLinear.from_float(
        m,
    )
    return swap_linear_layers(
        module,
        from_float,
    )


class TestFloat8Common:
    def broadcast_module(self, module: nn.Module) -> None:
        # Broadcast for multi-threaded process group tests since seed is per
        # process, not per thread
        for param in module.parameters():
            dist.broadcast(param, src=0)

    def init_transformer(
        self, weight_tying: bool, dtype: Optional[torch.dtype] = None
    ) -> nn.Module:
        torch.manual_seed(42)
        args = ModelArgs(
            n_layers=3,
            dim=768,
            n_heads=12,
            dropout_p=0.0,
            weight_tying=weight_tying,
            vocab_size=32,
        )
        module = Transformer(args).cuda()
        if dtype is not None:
            module = module.to(dtype=dtype)
        self.broadcast_module(module)
        return module


class TestFloat8MultiProcess(FSDPTest, TestFloat8Common):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parity(self):
        self.run_subtests(
            {
                "compile_transformer_block": [False, True],
                "precompute": [False, True],
                "scaling_type_weight": [ScalingType.DYNAMIC],
                "dtype": [torch.float32, torch.bfloat16],
            },
            self._test_transformer_parity,
        )

    def _test_transformer_parity(
        self,
        precompute: bool,
        scaling_type_weight: ScalingType,
        compile_transformer_block: bool,
        dtype: Optional[torch.dtype] = None,
    ):
        if scaling_type_weight is ScalingType.DELAYED and precompute:
            return

        module = self.init_transformer(weight_tying=False, dtype=dtype)

        local_inp = torch.randint(
            0, module.tok_embeddings.weight.size(0), (16, 16), device="cuda"
        )

        # reference modules
        ref_module = copy.deepcopy(module)
        convert_to_float8_comm_test_layers(
            ref_module,
        )

        # fp8 comm-only modules
        float8_linear_config2 = Float8LinearConfig(
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            enable_fsdp_float8_all_gather=True,
            use_fp8_all_gather_only=True,
        )
        convert_to_float8_training(
            module,
            config=float8_linear_config2,
        )

        for layer_id, transformer_block in module.layers.named_children():
            if compile_transformer_block:
                transformer_block = torch.compile(transformer_block, dynamic=False)
            fully_shard(transformer_block)
            module.layers.register_module(layer_id, transformer_block)
        fully_shard(module)

        ref_optim = torch.optim.Adam(ref_module.parameters(), lr=1e-2)
        optim = torch.optim.Adam(module.parameters(), lr=1e-2, foreach=True)

        check_parity_fp8_comm_only(
            self,
            ref_module,
            ref_optim,
            module,
            optim,
            local_inp,
            config=float8_linear_config2,
            precompute=precompute,
            compile=compile_transformer_block,
        )


if __name__ == "__main__":
    run_tests()
