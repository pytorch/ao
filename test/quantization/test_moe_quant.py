import unittest

import pytest
import torch
from parameterized import parameterized

from torchao.dtypes.floatx.float8_layout import Float8AQTTensorImpl
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl
from torchao.prototype.moe_quant.quantizable_moe_modules import (
    MoEFeedForwardAOQuantizable,
)
from torchao.prototype.moe_quant.utils import (
    FakeExtraDimTensor,
    MoEQuantConfig,
    UseFakeExtraDimTensor,
)
from torchao.quantization.quant_api import (
    AffineQuantizedTensor,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    LinearActivationQuantizedTensor,
    PerRow,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)

if torch.version.hip is not None:
    pytest.skip(
        "ROCm support for MoE quantization is under development",
        allow_module_level=True,
    )


def _moe_filter(mod, fqn):
    return isinstance(mod, MoEFeedForwardAOQuantizable)


class TestMoEQuantCompile(unittest.TestCase):
    DEFAULT_PARAMS = (8, 512, 256, 2)  # num_experts, hidden_dim, expert_dim, top_k

    @torch.no_grad()
    def _test_impl_moe_quant(
        self,
        config,
        num_tokens=1,
        model_params=None,
        base_class=None,
        tensor_impl_class=None,
        dtype=torch.bfloat16,
        device="cuda",
        fullgraph=False,
        decompose_grouped_mm=True,
    ):
        """
        Tests moe quant for techniques using fake extra dim
        """
        if model_params is None:
            model_params = self.DEFAULT_PARAMS

        input_shape = (num_tokens, model_params[1])
        model = (
            MoEFeedForwardAOQuantizable(
                *model_params,
                empty_init=False,
                decompose_grouped_mm=decompose_grouped_mm,
            )
            .to(dtype)
            .to(device)
        )
        input = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

        out = model(input)

        if config is not None:
            quantize_(model, config, _moe_filter)

        if (
            isinstance(config, MoEQuantConfig)
            and config.use_fake_extra_dim_tensor == UseFakeExtraDimTensor.TRUE
        ):
            self.assertIsInstance(model.experts.up_proj, FakeExtraDimTensor)
            if base_class is not None:
                self.assertIsInstance(model.experts.up_proj.head_tensor, base_class)
            if tensor_impl_class is not None:
                self.assertIsInstance(
                    model.experts.up_proj.head_tensor.tensor_impl, tensor_impl_class
                )
        else:
            if base_class is not None:
                self.assertIsInstance(model.experts.up_proj, base_class)
            if tensor_impl_class is not None:
                self.assertIsInstance(
                    model.experts.up_proj.tensor_impl, tensor_impl_class
                )

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
            (
                "single_token_grouped_mm_base",
                1,
                True,
                UseFakeExtraDimTensor.FALSE,
                False,
            ),
            (
                "multiple_token_grouped_mm_base",
                8,
                False,
                UseFakeExtraDimTensor.FALSE,
                False,
            ),
        ]
    )
    def test_noquant(
        self,
        name,
        num_tokens,
        fullgraph,
        use_fake_extra_dim_tensor,
        decompose_grouped_mm,
    ):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")
        if not (decompose_grouped_mm or TORCH_VERSION_AT_LEAST_2_8):
            self.skipTest("Test only enabled for 2.8+ for grouped mm")

        config = None

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )

    @parameterized.expand(
        [
            ("single_token_base", 1, True, UseFakeExtraDimTensor.FALSE),
            ("multiple_token_base", 8, False, UseFakeExtraDimTensor.FALSE),
            ("single_token_fake", 1, False, UseFakeExtraDimTensor.TRUE),
            ("multiple_token_fake", 8, False, UseFakeExtraDimTensor.TRUE),
        ]
    )
    def test_int4wo(self, name, num_tokens, fullgraph, use_fake_extra_dim_tensor):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            base_config=Int4WeightOnlyConfig(),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        base_class = AffineQuantizedTensor
        tensor_impl_class = TensorCoreTiledAQTTensorImpl
        decompose_grouped_mm = True

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )

    @parameterized.expand(
        [
            ("single_token_base", 1, True, UseFakeExtraDimTensor.FALSE),
            ("multiple_token_base", 8, False, UseFakeExtraDimTensor.FALSE),
            ("single_token_fake", 1, False, UseFakeExtraDimTensor.TRUE),
            ("multiple_token_fake", 8, False, UseFakeExtraDimTensor.TRUE),
        ]
    )
    def test_int8wo(self, name, num_tokens, fullgraph, use_fake_extra_dim_tensor):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            base_config=Int8WeightOnlyConfig(),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        tensor_impl_class = PlainAQTTensorImpl
        base_class = AffineQuantizedTensor
        decompose_grouped_mm = True

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )

    @parameterized.expand(
        [
            ("single_token_base", 1, True, UseFakeExtraDimTensor.FALSE),
            ("multiple_token_base", 8, False, UseFakeExtraDimTensor.FALSE),
            ("single_token_fake", 1, False, UseFakeExtraDimTensor.TRUE),
            ("multiple_token_fake", 8, False, UseFakeExtraDimTensor.TRUE),
        ]
    )
    def test_int8wo_cpu(self, name, num_tokens, fullgraph, use_fake_extra_dim_tensor):
        if not TORCH_VERSION_AT_LEAST_2_6:
            self.skipTest("Test only enabled for 2.6+")

        config = MoEQuantConfig(
            base_config=Int8WeightOnlyConfig(),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        tensor_impl_class = PlainAQTTensorImpl
        base_class = AffineQuantizedTensor
        decompose_grouped_mm = True

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
            device="cpu",
        )

    @parameterized.expand(
        [
            ("multiple_tokens_base", 32, False, UseFakeExtraDimTensor.FALSE),
            ("multiple_tokens_fake", 32, False, UseFakeExtraDimTensor.TRUE),
        ]
    )
    def test_int8dq(self, name, num_tokens, fullgraph, use_fake_extra_dim_tensor):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not TORCH_VERSION_AT_LEAST_2_5:
            self.skipTest("Test only enabled for 2.5+")

        config = MoEQuantConfig(
            base_config=Int8DynamicActivationInt8WeightConfig(),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        base_class = LinearActivationQuantizedTensor
        decompose_grouped_mm = True

        self._test_impl_moe_quant(
            model_params=(2, 512, 256, 2),
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )

    @parameterized.expand(
        [
            ("single_token_base", 1, True, UseFakeExtraDimTensor.FALSE),
            ("multiple_token_base", 8, False, UseFakeExtraDimTensor.FALSE),
            ("single_token_fake", 1, False, UseFakeExtraDimTensor.TRUE),
            ("multiple_token_fake", 8, False, UseFakeExtraDimTensor.TRUE),
        ]
    )
    def test_fp8wo(self, name, num_tokens, fullgraph, use_fake_extra_dim_tensor):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")

        config = MoEQuantConfig(
            base_config=Float8WeightOnlyConfig(),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        tensor_impl_class = Float8AQTTensorImpl
        base_class = AffineQuantizedTensor
        decompose_grouped_mm = True

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            tensor_impl_class=tensor_impl_class,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )

    @parameterized.expand(
        [
            ("single_token_base", 1, True, UseFakeExtraDimTensor.FALSE, True),
            ("multiple_token_base", 8, False, UseFakeExtraDimTensor.FALSE, True),
            ("single_token_fake", 1, False, UseFakeExtraDimTensor.TRUE, True),
            ("multiple_token_fake", 8, False, UseFakeExtraDimTensor.TRUE, True),
            (
                "single_token_grouped_mm_base",
                1,
                True,
                UseFakeExtraDimTensor.FALSE,
                False,
            ),
            (
                "multiple_token_grouped_mm_base",
                8,
                True,
                UseFakeExtraDimTensor.FALSE,
                False,
            ),
        ]
    )
    def test_fp8dq(
        self,
        name,
        num_tokens,
        fullgraph,
        use_fake_extra_dim_tensor,
        decompose_grouped_mm,
    ):
        if not torch.cuda.is_available():
            self.skipTest("Need CUDA available")
        if not is_sm_at_least_90():
            self.skipTest("Requires CUDA capability >= 9.0")
        if not (decompose_grouped_mm or TORCH_VERSION_AT_LEAST_2_8):
            self.skipTest("Test only enabled for 2.8+ for grouped mm")

        config = MoEQuantConfig(
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            use_fake_extra_dim_tensor=use_fake_extra_dim_tensor,
        )
        base_class = LinearActivationQuantizedTensor

        self._test_impl_moe_quant(
            config=config,
            num_tokens=num_tokens,
            base_class=base_class,
            fullgraph=fullgraph,
            decompose_grouped_mm=decompose_grouped_mm,
        )


if __name__ == "__main__":
    unittest.main()
