import unittest

import pytest
import torch
from parameterized import parameterized

from torchao.dtypes.floatx.float8_layout import Float8AQTTensorImpl
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl
from torchao.prototype.moe_quant.quantizable_moe_modules import (
    MOEFeedForwardAOQuantizable,
)
from torchao.prototype.moe_quant.utils import (
    FakeExtraDimTensor,
    MoEQuantConfig,
    UseFakeExtraDimTensor,
    cond_ffn_filter,
)
from torchao.quantization.quant_api import (
    AffineQuantizedTensor,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    LinearActivationQuantizedTensor,
    quantize_,
    PerRow,
    PerTensor,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    is_sm_at_least_90,
)
from torchao.quantization.utils import compute_error

if torch.version.hip is not None:
    pytest.skip(
        "ROCm support for MoE quantization is under development",
        allow_module_level=True,
    )
from torchao.prototype.moe_quant.kernels import fp8_dq_moe_op
from torchao.quantization.utils import _fbgemm_available

torch.manual_seed(0)

class TestMoEQuantCompile(unittest.TestCase):
    DEFAULT_PARAMS = (512, 256, 8, 2)  # hidden_dim, expert_dim, num_experts, top_k

    @torch.no_grad()
    def _test_impl_moe_quant(
        self,
        config,
        num_tokens=1,
        model_params=None,
        base_class=AffineQuantizedTensor,
        tensor_impl_class=None,
        dtype=torch.bfloat16,
        device="cuda",
        fullgraph=False,
    ):
        """
        Tests moe quant for techniques using fake extra dim
        """
        if model_params is None:
            model_params = self.DEFAULT_PARAMS

        input_shape = (num_tokens, model_params[0])
        model = (
            MOEFeedForwardAOQuantizable(*model_params, empty_init=False)
            .to(dtype)
            .to(device)
        )
        input = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
        out = model(input)

        quantize_(model, config, cond_ffn_filter)

        if (
            isinstance(config, MoEQuantConfig)
            and config.use_fake_extra_dim_tensor == UseFakeExtraDimTensor.TRUE
        ):
            self.assertIsInstance(model.experts.w1, FakeExtraDimTensor)
            if base_class is not None:
                self.assertIsInstance(model.experts.w1.head_tensor, base_class)
            if tensor_impl_class is not None:
                self.assertIsInstance(
                    model.experts.w1.head_tensor.tensor_impl, tensor_impl_class
                )
        else:
            if base_class is not None:
                self.assertIsInstance(model.experts.w1, base_class)
            if tensor_impl_class is not None:
                self.assertIsInstance(model.experts.w1.tensor_impl, tensor_impl_class)

        out_q = model(input)

        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        model_c = torch.compile(model, mode="reduce-overhead", fullgraph=fullgraph)

        model_c(input)
        model_c(input)
        out_qc = model_c(input).clone()

        for i in range(10):
            input = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
            model_c(input)

        self.assertGreaterEqual(compute_error(out_q, out), 10)
        self.assertGreaterEqual(compute_error(out_qc, out), 10)

    @parameterized.expand(
        [
            ("single_token", 1, False),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_int4wo_fake_dim(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            Int4WeightOnlyConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE
        )
        tensor_impl_class = TensorCoreTiledAQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, True),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_int4wo_base(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(Int4WeightOnlyConfig())
        tensor_impl_class = TensorCoreTiledAQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, False),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_int8wo_fake_dim(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            Int8WeightOnlyConfig(), use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE
        )
        tensor_impl_class = PlainAQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, True),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_int8wo_base(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_6:
            self.skipTest("Test only enabled for 2.6+")

        config = MoEQuantConfig(Int8WeightOnlyConfig())
        tensor_impl_class = PlainAQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, True),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_int8wo_base_cpu(self, name, num_tokens, fullgraph):
        if not TORCH_VERSION_AT_LEAST_2_6:
            self.skipTest("Test only enabled for 2.6+")

        config = MoEQuantConfig(Int8WeightOnlyConfig())
        tensor_impl_class = PlainAQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
            device="cpu",
        )

    @parameterized.expand(
        [
            ("multiple_tokens", 32, False),
        ]
    )
    def test_int8dq_fake_dim(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            Int8DynamicActivationInt8WeightConfig(),
            use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
        )
        base_class = LinearActivationQuantizedTensor

        self._test_impl_moe_quant(
            model_params=(512, 256, 2, 2),
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("multiple_tokens", 32, False),
        ]
    )
    def test_int8dq_base(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(Int8DynamicActivationInt8WeightConfig())
        base_class = LinearActivationQuantizedTensor

        self._test_impl_moe_quant(
            model_params=(512, 256, 2, 2),
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, False),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_fp8wo_fake_dim(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        config = MoEQuantConfig(
            Float8WeightOnlyConfig(),
            use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
        )
        tensor_impl_class = Float8AQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, True),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_fp8wo_base(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        config = MoEQuantConfig(Float8WeightOnlyConfig())
        tensor_impl_class = Float8AQTTensorImpl

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, False),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_fp8dq_fake_dim(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        config = MoEQuantConfig(
            Float8DynamicActivationFloat8WeightConfig(),
            use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE,
        )
        base_class = LinearActivationQuantizedTensor

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
        )

    @parameterized.expand(
        [
            ("single_token", 1, True),
            ("multiple_tokens", 8, False),
        ]
    )
    def test_fp8dq_base(self, name, num_tokens, fullgraph):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        config = MoEQuantConfig(Float8DynamicActivationFloat8WeightConfig())
        base_class = LinearActivationQuantizedTensor

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
        )

class TestFusedMoEQuant(unittest.TestCase):
    DEFAULT_PARAMS = (512, 256, 8, 2)  # hidden_dim, expert_dim, num_experts, top_k

    @parameterized.expand(
        [
            ("multiple_tokens", 8),
        ]
    )
    def test_pytorch_scaled_grouped_gemm(self, name, num_tokens):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        device = "cuda"
        dtype = torch.bfloat16

        config = MoEQuantConfig(Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

        model_params = self.DEFAULT_PARAMS

        input_shape = (num_tokens, model_params[0])
        input = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

        model = (
            MOEFeedForwardAOQuantizable(*model_params, empty_init=False)
        )
        model = model.to(dtype).to(device)

        out_orig = model(input)

        quantize_(model, config, cond_ffn_filter)
        
        w1 = model.experts.w1
        w2 = model.experts.w2
        w3 = model.experts.w3

        router = model.router
        top_k = model.top_k

        # preprocess
        scores = router(input)  # [T, E]
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores, expert_indices = torch.topk(
            scores, top_k, dim=-1
        )  # [T, A], [T, A]
        scores /= scores.sum(dim=-1, keepdim=True).to(input.dtype)  # [T, A]

        out = fp8_dq_moe_op(input, w1, w2, w3, expert_indices, scores)
        out2 = model(input)

        self.assertTrue(compute_error(out_orig, out) > 20)
        self.assertTrue(compute_error(out_orig, out2) > 20)


    @parameterized.expand(
        [
            ("multiple_tokens", 8),
        ]
    )
    def test_fbgemm_scaled_grouped_gemm(self, name, num_tokens):
        if not _fbgemm_available:
            self.skipTest("Need FBGEMM available")
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        device = "cuda"
        dtype = torch.bfloat16

        config = MoEQuantConfig(Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

        model_params = self.DEFAULT_PARAMS

        input_shape = (num_tokens, model_params[0])
        input = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

        model = (
            MOEFeedForwardAOQuantizable(*model_params, empty_init=False, use_fbgemm_kernel=True)
        )
        model = model.to(dtype).to(device)

        out_orig = model(input)

        quantize_(model, config, cond_ffn_filter)
        
        w1 = model.experts.w1
        w2 = model.experts.w2
        w3 = model.experts.w3

        router = model.router
        top_k = model.top_k

        # preprocess
        scores = router(input)  # [T, E]
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores, expert_indices = torch.topk(
            scores, top_k, dim=-1
        )  # [T, A], [T, A]
        scores /= scores.sum(dim=-1, keepdim=True).to(input.dtype)  # [T, A]

        out = fp8_dq_moe_op(input, w1, w2, w3, expert_indices, scores, use_fbgemm_kernel=True)
        out2 = model(input)

        self.assertTrue(compute_error(out_orig, out) > 20)
        self.assertTrue(compute_error(out_orig, out2) > 20)

if __name__ == "__main__":
    unittest.main()
