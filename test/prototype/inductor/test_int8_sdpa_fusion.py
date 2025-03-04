import itertools

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.utils.checkpoint
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import run_and_get_code
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torch.export import export_for_training
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU

from torchao.prototype.inductor.fx_passes.int8_sdpa_fusion import _int8_sdpa_init


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
        if self.has_mask and mask.dtype != scores.dtype:
            scores = scores + mask
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        context_layer = torch.matmul(attention, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.all_head_size,)
        )
        return self.dense(context_layer)

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
            compiled_model = torch.compile(dot_prod_attention, fullgraph=True)
            result2, source_code = run_and_get_code(
                compiled_model,
                *(args2 + dropout_arg),
            )
            source_code = "\n".join(source_code)
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["int8_fuse_attention"], 1)
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
    def _test_sdpa_int8_rewriter(self):
        # pattern is different for bs=1
        for dtype, has_mask, bs in itertools.product(
            [torch.float32, torch.bfloat16], [True, False], [56, 1]
        ):
            seqlen, numhead, headsize = 197, 16, 64
            mod = SelfAttnLikeModule(
                input_dim=headsize * numhead,
                has_mask=has_mask,
                num_attention_heads=numhead,
                attention_head_size=headsize,
            ).eval()
            inputs = (
                torch.randn(
                    (bs, seqlen, headsize * numhead), device=self.device, dtype=dtype
                ) * 10,
                torch.randn((bs, 1, 1, seqlen), device=self.device) * 10
                if has_mask
                else None,
            )
            enable_autocast = (dtype == torch.bfloat16)
            with torch.no_grad(), torch.amp.autocast("cpu", enabled=enable_autocast, dtype=torch.bfloat16):
                _int8_sdpa_init()
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                quantizer.set_function_type_qconfig(
                    torch.matmul, quantizer.get_global_quantization_config()
                )
                export_model = export_for_training(
                    mod,
                    inputs,
                ).module()
                prepare_model = prepare_pt2e(export_model, quantizer)
                prepare_model(*inputs)
                convert_model = convert_pt2e(prepare_model)
                torch.ao.quantization.move_exported_model_to_eval(convert_model)
                self._check_common(
                    convert_model, args1=inputs, check_train=False, atol=1.0
                )

if HAS_CPU:
    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_sdpa_int8_rewriter_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_int8_rewriter

if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
