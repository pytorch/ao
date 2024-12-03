import torchao

import contextlib
import functools
import itertools
import math

import torch
import torch.utils.checkpoint
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torchao.quantization.sfdp_int8_fx_pass import _sfdp_init_int8

class SelfAttnLikeModule(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        has_mask,
        num_attention_heads=None,
        attention_head_size=None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.q_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        assert num_attention_heads is not None
        assert attention_head_size is not None
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = torch.nn.Linear(self.all_head_size, self.all_head_size)
        self.dropout = torch.nn.Dropout(0)
        self.has_mask = has_mask

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute([0, 2, 1, 3])

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim**0.5)
        if self.has_mask:
            scores = scores + mask
        attention = self.softmax(scores)
        # attention = self.dropout(attention)
        context_layer = torch.matmul(attention, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.all_head_size,)
        )
        return self.dense(context_layer)

def _generate_qdq_quantized_model(mod, inputs, quantizer):
    with torch.no_grad():
        export_model = capture_pre_autograd_graph(mod, inputs)
        prepare_model = prepare_pt2e(export_model, quantizer)
        prepare_model(*inputs)
        convert_model = convert_pt2e(prepare_model)
        torch.ao.quantization.move_exported_model_to_eval(convert_model)
        return convert_model

class TestSDPAPatternRewriterTemplate(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
        check_train=True,
        override_check_equal=False,
        dtype=torch.float,
        rtol=1.3e-6,
    ):
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        for training in [False, True] if check_train else [False]:
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            counters.clear()
            torch.manual_seed(1234)
            result2, source_code = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True),
                *(args2 + dropout_arg),
            )
            source_code = "\n".join(source_code)
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention_int8"], 1)
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "torchao.scaled_dot_product_int8",
                    source_code,
                )

            # some tests configured with very low dropout where we still want to check equality
            if not has_dropout or override_check_equal:
                self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and (not has_dropout or override_check_equal)
                    ):
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=rtol)

    @skipIfRocm
    @config.patch({"freezing": True})
    def _test_sdpa_rewriter_int8_1_to_4(self):
        # pattern is different for bs=1
        for dtype, has_mask, bs in itertools.product(
            [torch.float32], [True, False], [56, 1]
        ):
            mod = SelfAttnLikeModule(
                input_dim=64 * 16,
                has_mask=has_mask,
                num_attention_heads=16,
                attention_head_size=64,
            ).eval()
            maybe_autocast = (
                torch.cpu.amp.autocast()
                if dtype == torch.bfloat16
                else contextlib.nullcontext()
            )
            inputs = [
                torch.randn((bs, 384, 64 * 16), device=self.device, dtype=dtype),
                torch.randn((bs, 1, 1, 384), device=self.device) if has_mask else None,
            ]
            with torch.no_grad(), maybe_autocast:
                _sfdp_init_int8()
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                quantizer.set_function_type_qconfig(
                    torch.matmul, quantizer.get_global_quantization_config()
                )
                convert_model = _generate_qdq_quantized_model(mod, inputs, quantizer)
                self._check_common(
                    convert_model, args1=inputs, check_train=False, atol=1.0
                )

if HAS_CPU:
    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_sdpa_rewriter_int8_1_to_4_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_int8_1_to_4

if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
