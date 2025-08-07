# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.prototype.moe_quant.utils import MoEQuantConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    _is_fbgemm_genai_gpu_available,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 128


class Experts(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.num_local_experts = num_local_experts
        self.dim = dim

        self.w1: nn.Parameter = nn.Parameter(
            torch.randn(
                num_local_experts,
                dim,
                hidden_dim,
                dtype=dtype,
                device=device,
            )
        )

        self.w2: nn.Parameter = nn.Parameter(
            torch.randn(
                num_local_experts,
                hidden_dim,
                dim,
                dtype=dtype,
                device=device,
            )
        )

        self.w3: nn.Parameter = nn.Parameter(
            torch.randn(
                num_local_experts,
                dim,
                hidden_dim,
                dtype=dtype,
                device=device,
            )
        )

    def forward(
        self,
        routed_in_egD: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        e = self.num_local_experts
        D = self.dim

        x_egD = routed_in_egD.view(e, -1, D)

        middle_out_egF = F.silu(torch.bmm(x_egD, self.w1)) * torch.bmm(x_egD, self.w3)
        out_egD = torch.bmm(middle_out_egF, self.w2)
        out_egD = out_egD.view(-1, D)

        return out_egD


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# TODO: move tests in test_affine_quantized_float.py here after we migrated all implementations
@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
class TestFloat8Tensor(TestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "kernel_preference",
        [KernelPreference.AUTO, KernelPreference.TORCH, KernelPreference.FBGEMM],
    )
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_fp8_linear_variants(
        self,
        dtype: torch.dtype,
        mode: str,
        compile: bool,
        granularity,
        kernel_preference: KernelPreference,
        sizes: Tuple,
    ):
        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"

        if mode == "weight-only" and kernel_preference != KernelPreference.AUTO:
            return unittest.skip(
                "weight only quant only uses AUTO kernel preference right now"
            )

        if kernel_preference == KernelPreference.FBGEMM and (
            (not _is_fbgemm_genai_gpu_available()) or (not is_sm_at_least_90())
        ):
            return unittest.skip(
                "Requires fbgemm_gpu_genai to run fbgemm kernel preference test"
            )

        error_context = (
            self.assertRaisesRegex(AssertionError, error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)

            if mode == "dynamic":
                config = Float8DynamicActivationFloat8WeightConfig(
                    granularity=granularity,
                    kernel_preference=kernel_preference,
                )
            else:
                assert mode == "weight-only", f"Unsupported mode: {mode}"
                config = Float8WeightOnlyConfig()

            quantize_(quantized_model, config)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @unittest.skipIf(
        not is_sm_at_least_90(),
        "Failing in SM89 right now: "
        "AssertionError: tensor(False, device='cuda:0') is not true : sqnr: -2.90625, will fix a bit later",
    )
    def test_slice(self, granularity):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        dtype = torch.bfloat16
        device = "cuda"
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)
        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )
        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, config)
        weight1 = dummy.weight.clone().narrow(0, 0, 64)
        weight2 = dummy.weight.clone().narrow(1, 0, 128)
        self.assertEqual(
            weight1.qdata,
            dummy.weight.qdata.narrow(0, 0, 64),
        )
        self.assertEqual(
            weight2.qdata,
            dummy.weight.qdata.narrow(1, 0, 128),
        )
        if isinstance(granularity, PerRow):
            self.assertEqual(
                weight1.scale,
                dummy.weight.scale.narrow(0, 0, 64),
            )
            self.assertEqual(
                weight2.scale,
                dummy.weight.scale,
            )
        else:
            self.assertEqual(
                weight1.scale,
                dummy.weight.scale,
            )
            self.assertEqual(
                weight2.scale,
                dummy.weight.scale,
            )

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 25, f"sqnr: {sqnr}")

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 15, f"sqnr: {sqnr}")

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_kernel_preference_numerical_equivalence(self, granularity, sizes):
        """Test different kernel preferences have the same numerics for float8 dynamic activation
        and float8 weight config
        """
        M, N, K = sizes
        dtype = torch.bfloat16
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
        # Create a linear layer with bfloat16 dtype
        model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

        # reference kernel preference and results
        # we are using KerenelPreference.TORCH as the reference
        kp_ref = KernelPreference.TORCH
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity, kernel_preference=kp_ref
        )
        quantized_model = copy.deepcopy(model)
        quantize_(quantized_model, config)
        res_ref = quantized_model(input_tensor)

        other_kernel_preferences = [
            KernelPreference.AUTO,
        ]
        if _is_fbgemm_genai_gpu_available() and is_sm_at_least_90():
            other_kernel_preferences.append(KernelPreference.FBGEMM)

        quantized_outputs = {}
        for kp in other_kernel_preferences:
            config = Float8DynamicActivationFloat8WeightConfig(
                granularity=granularity, kernel_preference=kp
            )
            quantized_model = copy.deepcopy(model)
            quantize_(quantized_model, config)
            quantized_outputs[kp] = quantized_model(input_tensor)

        from torchao.quantization.utils import compute_error

        # comparing numerics between different kernel preferences, using TORCH as the standard
        kp_and_res = list(quantized_outputs.items())
        for i in range(len(kp_and_res)):
            kp, res = kp_and_res[i]
            self.assertTrue(
                compute_error(res, res_ref) > 28,
                f"mismatch between {kp=} and {kp_ref}, {sizes=}, {granularity=}",
            )

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_slice_preserves_aliasing(self, granularity):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        # Making sure the aliasing is preserved in sliced quantized Tensor
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_slice_and_copy_similar_to_vllm(self, granularity):
        # making sure https://github.com/vllm-project/vllm/blob/90bd2ab6e3eb7e83d3f40d99fc23e6e43834743a/vllm/model_executor/layers/linear.py#L483-L495 works properly
        # the test is similar to the linked code, but with some hardcoded arguments
        # and does not use tensor parallelism

        dtype = torch.bfloat16
        device = "cuda"
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        l = torch.nn.Linear(1024, 1024, device="cuda", dtype=dtype)
        quantize_(l, config)

        # high level, we do a narrow for both param.data and the loaded_weights
        # and do inplace copy_ to copy from the loaded_weights into param.data

        # simulate loaded_weight
        dummy_l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        # making the weight different
        dummy_l.weight = torch.nn.Parameter(
            dummy_l.weight + 2 * torch.randn(1024, 1024, device=device, dtype=dtype),
            requires_grad=False,
        )
        quantize_(dummy_l, config)

        output_dim = 0
        shard_size = 512
        for tp_rank in [0, 1]:
            start_idx = tp_rank * shard_size
            param = l.weight
            param_data = param.data
            param_data = param_data.narrow(output_dim, start_idx, shard_size)
            orig_value = param_data.qdata[0][0].item()
            loaded_weight = dummy_l.weight
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            # making sure param.data.qdata[0][0] is not the same as loaded_weight.qdata[0][0]
            assert orig_value != loaded_weight.qdata[0][0]
            param_data.copy_(loaded_weight)
            # making sure param.data is updated to loaded_weight
            assert param_data.qdata[0][0] == loaded_weight.qdata[0][0]
            assert param_data.scale[0] == loaded_weight.scale[0]

    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_bmm(self):
        # only support per row quantization
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())

        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.bmm(x, self.weight)

        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(10, 32, 128, dtype=dtype, device=device)
        weight = torch.randn(10, 128, 256, dtype=dtype, device=device)
        m = M(weight).eval()
        original = m(input)
        # we need to transpose the weight first for bmm
        m.weight = torch.nn.Parameter(m.weight.transpose(1, 2).contiguous())
        quantize_(m, config, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_to_device(self, granularity, sizes):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        M, N, K = sizes
        dtype = torch.bfloat16
        for device in self.GPU_DEVICES:
            input_tensor = torch.randn(*M, K, dtype=dtype, device=device)
            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device)
            linear(input_tensor)

            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device=device)
            linear(input_tensor)

            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device)
            linear(input_tensor)

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_cat(self, granularity, sizes):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        dtype = torch.bfloat16
        device = "cuda"
        M, N, K = sizes
        linear1 = torch.nn.Linear(K, N, dtype=dtype, device=device)
        linear2 = torch.nn.Linear(K, N, dtype=dtype, device=device)
        input_cat1 = torch.randn(*M, K, dtype=dtype, device=device)

        cat_weight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        dummy_linear1 = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)

        dummy_linear1.weight = torch.nn.Parameter(cat_weight1)
        quantize_(dummy_linear1, config)

        quantize_(linear1, config)
        quantize_(linear2, config)

        cat_qweight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        self.assertTrue(cat_qweight1.shape, (2 * N, K))
        self.assertEqual(
            dummy_linear1.weight.qdata,
            cat_qweight1.qdata,
        )
        self.assertEqual(
            dummy_linear1.weight.scale,
            cat_qweight1.scale,
        )

        # making sure cat_qweight1 can be used for inference
        dummy_linear1.weight = torch.nn.Parameter(cat_qweight1, requires_grad=False)
        dummy_linear1(input_cat1)

        # align the scale before concatenation
        linear2.weight.scale = linear1.weight.scale
        cat_qweight2 = torch.cat([linear1.weight, linear2.weight], dim=1)
        self.assertTrue(cat_qweight2.shape, (N, 2 * K))
        ref_data = torch.cat(
            [
                linear1.weight.qdata,
                linear2.weight.qdata,
            ],
            dim=1,
        )
        ref_scale = linear1.weight.scale
        self.assertEqual(cat_qweight2.qdata, ref_data)
        self.assertEqual(cat_qweight2.scale, ref_scale)

    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_moe_weight_reshape_ops(self):
        """This is testing the op call sequence in saving and loading quantization
        checkpoints in llama-models for llama4
        (https://github.com/meta-llama/llama-models/tree/main/models/llama4)
        """
        # only per row quantization is supported for bmm
        granularity = PerRow()
        dtype = torch.bfloat16
        device = "cuda"

        bmm_config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        moe_config = MoEQuantConfig(bmm_config)

        batch_size = 4
        num_experts = 2
        input_dim = 64
        dim = 128
        hidden_dim = 256

        moe1 = Experts(num_experts, dim, hidden_dim, dtype, device)
        moe2 = Experts(num_experts, dim, hidden_dim, dtype, device)
        moe_combined = Experts(num_experts, dim, 2 * hidden_dim, dtype, device)
        input = torch.randn(batch_size, input_dim, dim, dtype=dtype, device=device)

        moes = [moe1, moe2]

        for moe in moes:
            moe(input)

            def filter_fn(module, fqn):
                return isinstance(module, Experts)

            # need to transpose before quantizing
            moe.w1 = torch.nn.Parameter(
                moe.w1.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w2 = torch.nn.Parameter(
                moe.w2.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w3 = torch.nn.Parameter(
                moe.w3.transpose(1, 2).contiguous(), requires_grad=False
            )

            quantize_(moe, moe_config, filter_fn=filter_fn)

            # make sure it runs
            before = moe(input)

            # transposing for resharding support since only 2D resharding is supported
            new_last_dim = moe.w1.shape[-2]
            moe.w1 = torch.nn.Parameter(
                moe.w1.transpose(1, 2).reshape(-1, new_last_dim), requires_grad=False
            )
            new_last_dim = moe.w2.shape[-2]
            moe.w2 = torch.nn.Parameter(
                moe.w2.transpose(1, 2).reshape(-1, new_last_dim), requires_grad=False
            )
            new_last_dim = moe.w3.shape[-2]
            moe.w3 = torch.nn.Parameter(
                moe.w3.transpose(1, 2).reshape(-1, new_last_dim), requires_grad=False
            )

            moe.w1 = torch.nn.Parameter(
                moe.w1.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )
            moe.w2 = torch.nn.Parameter(
                moe.w2.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )
            moe.w3 = torch.nn.Parameter(
                moe.w3.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )

            # transpose again to recover the original weights
            moe.w1 = torch.nn.Parameter(moe.w1.transpose(1, 2), requires_grad=False)
            moe.w2 = torch.nn.Parameter(moe.w2.transpose(1, 2), requires_grad=False)
            moe.w3 = torch.nn.Parameter(moe.w3.transpose(1, 2), requires_grad=False)

            # make sure it runs
            after = moe(input)

            self.assertEqual(before, after)

        state_dicts = [moe1.state_dict(), moe2.state_dict()]
        # align the scale parameter so they can be concatenated
        for key in ["w1", "w2", "w3"]:
            weights = [st[key] for st in state_dicts]
            for i in range(1, len(weights)):
                weights[i].scale = weights[0].scale

        def process_key(key: str) -> torch.Tensor:
            tensors = [s[key] for s in state_dicts]
            # Note: we have a hacky implementation for cat in user codebase
            # since it is not implemented correctly before
            if key == "w2":
                return torch.cat(tensors, dim=-1)
            else:
                return torch.cat(tensors, dim=-2)

        new_state_dict = {}
        for key in ["w1", "w2", "w3"]:
            new_state_dict[key] = process_key(key)

        moe_combined.w1 = torch.nn.Parameter(
            moe_combined.w1.transpose(1, 2), requires_grad=False
        )
        moe_combined.w2 = torch.nn.Parameter(
            moe_combined.w2.transpose(1, 2), requires_grad=False
        )
        moe_combined.w3 = torch.nn.Parameter(
            moe_combined.w3.transpose(1, 2), requires_grad=False
        )
        moe_combined.load_state_dict(new_state_dict, assign=True)
        # make sure it runs
        moe_combined(input)


common_utils.instantiate_parametrized_tests(TestFloat8Tensor)

if __name__ == "__main__":
    run_tests()
