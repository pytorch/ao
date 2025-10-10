import itertools
import unittest

import torch
import torch.utils.checkpoint
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU

import torchao
from torchao.prototype.inductor.fx_passes.qsdpa_fusion import (
    _qsdpa_init,
    custom_pass,
)
from torchao.utils import torch_version_at_least


def qdq(input, scale):
    dtype = input.dtype
    q_input = torch.ops.torchao.quantize_affine_float8_non_decomposed.default(
        input,
        torch.tensor([scale]),
        torch.float8_e4m3fn,
    )
    dq_input = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
        q_input,
        torch.tensor([scale]),
        dtype,
    )
    return dq_input


def fp8_convert_(model):
    def generate_model_info(model):
        from collections import namedtuple

        mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
        parent_child_mod_dict = {}

        def create_mod_info_recursion(parent):
            for name, mod in parent.named_children():
                parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
                create_mod_info_recursion(mod)

        create_mod_info_recursion(model)
        return parent_child_mod_dict

    parent_child_mod_dict = generate_model_info(model)
    for name, mod in model.named_modules():
        mod_type_str = mod.__class__.__name__
        if mod_type_str not in [
            "Linear",
            "SDPA",
        ]:
            continue
        if mod_type_str == "Linear":
            param = mod.weight
            xmax = torch.max(param)
            weight_scale = xmax / torch.finfo(torch.float8_e4m3fn).max
            mod.weight_scale = weight_scale
            q_param = torch.clamp(
                (param / weight_scale),
                torch.finfo(torch.float8_e4m3fn).min,
                torch.finfo(torch.float8_e4m3fn).max,
            ).to(torch.float8_e4m3fn)
            mod.weight.data = q_param
            patched_mod = FP8QDQLinear(mod.in_features, mod.out_features, False)
            patched_mod.bias = mod.bias
            patched_mod.weight_scale = weight_scale.item()
            patched_mod.weight.data = q_param
        else:
            patched_mod = FP8QDQSDPA()
            patched_mod.__dict__.update(mod.__dict__)
            patched_mod.transpose_for_scores = mod.transpose_for_scores

            patched_mod.q_out_scale = (
                patched_mod.q_out_scale / torch.finfo(torch.float8_e4m3fn).max
            )
            patched_mod.k_out_scale = (
                patched_mod.k_out_scale / torch.finfo(torch.float8_e4m3fn).max
            )
            patched_mod.attn_weights_scale = (
                patched_mod.attn_weights_scale / torch.finfo(torch.float8_e4m3fn).max
            )
            patched_mod.v_out_scale = (
                patched_mod.v_out_scale / torch.finfo(torch.float8_e4m3fn).max
            )
            patched_mod.qk_out_scale = (
                patched_mod.qk_out_scale / torch.finfo(torch.float8_e4m3fn).max
            )
            patched_mod.attn_out_scale = (
                patched_mod.attn_out_scale / torch.finfo(torch.float8_e4m3fn).max
            )

        parent = parent_child_mod_dict[mod].parent
        name = parent_child_mod_dict[mod].name
        setattr(parent, name, patched_mod)
    model.eval()
    return model


class FP8QDQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias):
        super().__init__()
        self.qtype = torch.float8_e4m3fn
        self.weight = torch.randn((out_features, in_features)).to(self.qtype)
        self.weight_scale = 2.0
        self.scale = 2.0
        self.bias = None
        if has_bias:
            self.bias = torch.randn((out_features,))

    def forward(self, input):
        weight = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=self.weight.data,
            scale=torch.tensor([self.weight_scale]),
            output_dtype=torch.float,
        )

        q_input = torch.ops.torchao.quantize_affine_float8_non_decomposed.default(
            tensor=input,
            scale=torch.tensor([self.scale]),
            float8_dtype=self.qtype,
        )
        dq_input = torch.ops.torchao.dequantize_affine_float8_non_decomposed.default(
            tensor=q_input,
            scale=torch.tensor([self.scale]),
            output_dtype=torch.float,
        )

        out = torch.nn.functional.linear(dq_input, weight, self.bias)
        return out


class FP8QDQSDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_out_scale = 1.5
        self.k_out_scale = 1.5
        self.attn_weights_scale = 1.5
        self.v_out_scale = 1.5
        self.attn_out_scale = 1.5
        self.qk_out_scale = 1.5

    def forward(self, q, k, v, mask):
        key = self.transpose_for_scores(q)
        value = self.transpose_for_scores(k)
        query = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_qdq = qdq(query, self.q_out_scale)
        key_qdq = qdq(key.transpose(-1, -2), self.k_out_scale)
        attn_weights = torch.matmul(query_qdq, key_qdq) / (self.input_dim**0.5)

        # Normalize the attention scores to probabilities.
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout = 0.0 if not self.training else self.dropout_prob
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=self.training
        )

        # Mask heads if we want to
        if mask is not None:
            attn_weights = attn_weights + mask

        value_qdq = qdq(value, self.v_out_scale)
        attn_weights_qdq = qdq(attn_weights, self.attn_weights_scale)
        attn_output = torch.matmul(attn_weights_qdq, value_qdq)
        attn_output = attn_output.transpose(1, 2).contiguous()

        new_context_layer_shape = attn_output.size()[:-2] + (self.all_head_size,)
        attn_output = attn_output.reshape(new_context_layer_shape)

        return attn_output


class SDPA(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        has_mask,
        num_attention_heads,
        attention_head_size,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = torch.nn.Dropout(0)
        self.has_mask = has_mask

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute([0, 2, 1, 3])

    def forward(self, q, k, v, mask):
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
        return context_layer.reshape(context_layer.size()[:-2] + (self.all_head_size,))


class MHAModule(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        has_mask,
        num_attention_heads,
        attention_head_size,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.q_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = torch.nn.Linear(self.all_head_size, self.all_head_size)
        self.attn_mod = SDPA(
            input_dim,
            has_mask,
            num_attention_heads,
            attention_head_size,
        )

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        context_layer = self.attn_mod(q, k, v, mask)
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
                self.assertGreaterEqual(counters["inductor"]["qsdpa_fuse_attention"], 1)
            if contains:
                self.assertTrue(
                    any(
                        op_name in source_code
                        for op_name in [
                            "qscaled_dot_product",
                            "cpp_fused_quantize_per_tensor",
                            "cpp_fused__unsafe_view_quantize_per_tensor",
                        ]
                    )
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
    @unittest.skipIf(
        not torch_version_at_least("2.7.0"),
        reason="qsdpa requires torch 2.7 or later",
    )
    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::qscaled_dot_product"),
        reason="cpp kernels not built",
    )
    @config.patch({"freezing": True})
    def _test_int8_sdpa_rewriter(self):
        import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
        )

        # pattern is different for bs=1
        torch.manual_seed(1234)
        for dtype, has_mask, bs in itertools.product(
            [torch.float32, torch.bfloat16], [True, False], [56, 1]
        ):
            seqlen, numhead, headsize = 197, 16, 64
            mod = MHAModule(
                input_dim=headsize * numhead,
                has_mask=has_mask,
                num_attention_heads=numhead,
                attention_head_size=headsize,
            ).eval()
            inputs = (
                torch.randn(
                    (bs, seqlen, headsize * numhead), device=self.device, dtype=dtype
                ),
                torch.randn((bs, 1, 1, seqlen), device=self.device)
                if has_mask
                else None,
            )
            enable_autocast = dtype == torch.bfloat16
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    self.device, enabled=enable_autocast, dtype=torch.bfloat16
                ),
                config.patch(post_grad_custom_pre_pass=custom_pass),
            ):
                _qsdpa_init()
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                quantizer.set_function_type_qconfig(
                    torch.matmul, quantizer.get_global_quantization_config()
                )
                export_model = torch.export.export(mod, inputs, strict=True).module()
                prepare_model = prepare_pt2e(export_model, quantizer)
                prepare_model(*inputs)
                convert_model = convert_pt2e(prepare_model)
                torchao.quantization.pt2e.move_exported_model_to_eval(convert_model)

                self._check_common(
                    convert_model, args1=inputs, check_train=False, atol=1.0
                )

    @skipIfRocm
    @unittest.skipIf(
        not torch_version_at_least("2.7.0"),
        reason="qsdpa requires torch 2.7 or later",
    )
    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::qscaled_dot_product"),
        reason="cpp kernels not built",
    )
    @config.patch({"freezing": True})
    def _test_fp8_sdpa_rewriter(self):
        import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq

        # pattern is different for bs=1
        torch.manual_seed(1234)
        for dtype, bs in itertools.product([torch.float32, torch.bfloat16], [56, 1]):
            seqlen, numhead, headsize = 197, 16, 64
            mod = MHAModule(
                input_dim=headsize * numhead,
                has_mask=False,
                num_attention_heads=numhead,
                attention_head_size=headsize,
            ).eval()
            inputs = (
                torch.randn(
                    (bs, seqlen, headsize * numhead), device=self.device, dtype=dtype
                ),
                None,
            )
            enable_autocast = dtype == torch.bfloat16
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    self.device, enabled=enable_autocast, dtype=torch.bfloat16
                ),
                config.patch(post_grad_custom_pre_pass=custom_pass),
            ):
                _qsdpa_init()
                convert_model = fp8_convert_(mod)

                self._check_common(
                    convert_model, args1=inputs, check_train=False, atol=1.0
                )


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_int8_sdpa_rewriter_cpu = (
            TestSDPAPatternRewriterTemplate._test_int8_sdpa_rewriter
        )
        test_fp8_sdpa_rewriter_cpu = (
            TestSDPAPatternRewriterTemplate._test_fp8_sdpa_rewriter
        )


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
