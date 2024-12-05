import pytest

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import copy
import io
import math
import random
import unittest
from contextlib import nullcontext
from functools import partial
from typing import Tuple

import pytest
import torch
from einops import rearrange, repeat
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal import common_utils

from torchao.float8.float8_utils import compute_error
from torchao.quantization import (
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_api import (
    float8_static_activation_float8_weight,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
)
from torchao.utils import (
    is_sm_at_least_89,
    is_sm_at_least_90,
)

random.seed(0)
torch.manual_seed(0)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestAffineQuantizedFloat8Compile(InductorTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "granularity", [PerTensor(), PerRow()] if is_sm_at_least_90() else [PerTensor()]
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
        self, dtype: torch.dtype, mode: str, compile: bool, sizes: Tuple, granularity
    ):
        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"
            elif mode == "static":
                error_message = (
                    "Static quantization only supports PerTensor granularity"
                )

        error_context = (
            pytest.raises(AssertionError, match=error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
            # Get a "reasonable" scale for the input tensor even though
            # we use the same scale for multiple activations
            scale, _ = choose_qparams_affine(
                input_tensor,
                MappingType.SYMMETRIC,
                input_tensor.shape,
                torch.float8_e4m3fn,
                scale_dtype=torch.float32,
            )
            mode_map = {
                "dynamic": partial(
                    float8_dynamic_activation_float8_weight, granularity=granularity
                ),
                "weight-only": float8_weight_only,
                "static": partial(
                    float8_static_activation_float8_weight,
                    scale=scale,
                    granularity=granularity,
                ),
            }

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)
            factory = mode_map[mode]()
            quantize_(quantized_model, factory)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert (
                compute_error(output_original, output_quantized) > 20
            ), f"Quantization error is too high got a SQNR of {error}"

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            float8_dynamic_activation_float8_weight(granularity="invalid")

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_mismatched_granularity(self):
        with pytest.raises(
            ValueError,
            match="Different granularities for activation and weight are not supported",
        ):
            float8_dynamic_activation_float8_weight(granularity=(PerTensor(), PerRow()))

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_unsupported_granularity(self):
        class UnsupportedGranularity:
            pass

        with pytest.raises(ValueError, match="Invalid granularity types"):
            float8_dynamic_activation_float8_weight(
                granularity=(UnsupportedGranularity(), UnsupportedGranularity())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_per_row_with_float32(self):
        with pytest.raises(
            AssertionError,
            match="PerRow quantization only works for bfloat16 precision",
        ):
            model = ToyLinearModel(64, 64).eval().to(torch.float32).to("cuda")
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerRow())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    def test_serialization(self, mode: str):
        # Create and quantize the model
        model = ToyLinearModel(16, 32).to(device="cuda")

        mode_map = {
            "dynamic": partial(
                float8_dynamic_activation_float8_weight, granularity=PerTensor()
            ),
            "weight-only": float8_weight_only,
            "static": partial(
                float8_static_activation_float8_weight,
                scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
                granularity=PerTensor(),
            ),
        }
        factory = mode_map[mode]()
        quantize_(model, factory)

        # Save the state dict to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)

        # Reset the buffer position
        buffer.seek(0)

        # Load the state dict from the buffer
        weights_only_load = True
        loaded_state_dict = torch.load(buffer, weights_only=weights_only_load)

        # Create a new model and load the state dict
        with torch.device("meta"):
            new_model = ToyLinearModel(16, 32)
            if mode == "static":
                quantize_(new_model, factory)
            new_model.load_state_dict(loaded_state_dict, assign=True)

        # Compare the original and loaded models
        for layer_name in ["linear1", "linear2"]:
            original_layer = getattr(model, layer_name)
            new_layer = getattr(new_model, layer_name)

            # Compare weights
            if mode == "weight-only":
                original_weight = original_layer.weight.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = new_layer.weight.tensor_impl.float8_data.to(torch.float32)
            else:
                original_weight = original_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = (
                    new_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                        torch.float32
                    )
                )

            assert torch.allclose(
                original_weight, new_weight
            ), f"Weights do not match for {layer_name}"

            # Compare scales
            if hasattr(original_layer.weight, "scale"):
                assert torch.allclose(
                    original_layer.weight.scale, new_layer.weight.scale
                ), f"Scales do not match for {layer_name}"

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_fp8_weight_dimension_warning(self):
        # Create model with incompatible dimensions (not multiples of 16)
        model = ToyLinearModel(10, 25).cuda()  # 10x25 and 25x10 weights

        # Set up logging capture
        with self.assertLogs(
            "torchao.quantization.quant_api", level="INFO"
        ) as log_context:
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerTensor())
            )
            print(model)

        # Verify warning messages for both layers
        expected_messages = [
            "Skipping float8 quantization: weight shape torch.Size([25, 10])",
            "Skipping float8 quantization: weight shape torch.Size([10, 25])",
        ]
        # Check that we got warnings for both incompatible layers
        warning_count = sum(
            1 for msg in log_context.output if "Skipping float8 quantization" in msg
        )
        self.assertEqual(warning_count, 2, "Expected warnings for both linear layers")

        # Check warning message content
        for expected in expected_messages:
            self.assertTrue(
                any(expected in msg for msg in log_context.output),
                f"Expected warning message containing: {expected}",
            )


# copied from https://github.com/Dao-AILab/flash-attention/blob/1feb711f46563960fc10a8e659c93c300619504b/tests/test_util.py#L185


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores /= softcap
        scores = scores.tanh()
        scores *= softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    if key_padding_mask is not None:
        output.masked_fill_(
            rearrange(
                torch.logical_not(torch.any(key_padding_mask, 1)), "b -> b 1 1 1"
            ),
            0.0,
        )
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


class TestAffineQuantizedFloat8Attention(common_utils.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_float8_attention(self):
        import torch.nn.functional as F

        from torchao.quantization.quant_api import _float8_symmetric_per_tensor_quant

        class MyModel(torch.nn.Module):
            def forward(self, q, k, v, float8_quantize=False):
                if float8_quantize:
                    q = _float8_symmetric_per_tensor_quant(q)
                    k = _float8_symmetric_per_tensor_quant(k)
                    v = _float8_symmetric_per_tensor_quant(v)
                return F.scaled_dot_product_attention(q, k, v)

        # note: last headdim must be 64, 128, 256
        q = torch.randn([64, 8, 8, 64], dtype=torch.bfloat16, device="cuda")
        k = torch.randn([64, 8, 8, 64], dtype=torch.bfloat16, device="cuda")
        v = torch.randn([64, 8, 8, 64], dtype=torch.bfloat16, device="cuda")

        m = MyModel().eval()
        # it differs a lot from the non-quantized implementation
        # sqnr = -2.5
        # ref = m(q, k, v)

        # but matches the custom attention implementation in flash attention repo
        ref = attention_ref(q, k, v)[0]
        quantized = m(q, k, v, True)
        assert compute_error(ref, quantized) > 25.0


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])
    common_utils.run_tests()
