"""
Prototype of QAT with exact (instead of emulated) forward pass using
integer matrix multiply.

Quant spec:
* int4 symmetric weights w/ group size 32 or 256,
* int8 asymmetric per-token dynamic activations

"""

import copy

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.float8.float8_utils import compute_error
from torchao.quantization import quantize_
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.qat.fake_quantizer import FakeQuantizer
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    _do_fake_quantize_affine,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)

torch.manual_seed(0)


class Int8PerTokenActivationInt4PerGroupWeightLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # manually create fake quantizer configs
        activation_config = FakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False
        )
        weight_config = FakeQuantizeConfig(torch.int4, group_size=32)

        # manually create fake quantizers
        # reference: `FakeQuantizedLinear` (https://github.com/pytorch/ao/blob/c2a6568a04075acc371a338206216bb65536fb27/torchao/quantization/qat/linear.py)
        self.activation_fq = FakeQuantizer(activation_config)
        self.weight_fq = FakeQuantizer(weight_config)

    def forward(self, input):
        # fake quantize, scales and zero points will be stored on the fq objects
        input_fq = self.activation_fq(input)
        weight_fq = self.weight_fq(self.weight)

        # print('input.shape', input.shape)
        # print('input_scale.shape', self.activation_fq.scale.shape)
        # print('input_zero_point.shape', self.activation_fq.zero_point.shape)
        # print('input_zero_point', self.activation_fq.zero_point)

        # fake quantize the input again, this time also getting the quantize
        # values
        # Note: this is inefficient, but easiest to get 100% matching numerics
        # with existing QAT code without any code changes. In the real version,
        # we should just quantize once.
        #
        # modified version of torchao.quantization.qat.utils::_fake_quantize_per_token,
        # with `_fake_quantize_affine` replaced by `_do_fake_quantize_affine`
        #
        act_block_size = _get_per_token_block_size(input)
        act_qmin, act_qmax = _DTYPE_TO_QVALUE_BOUNDS[self.activation_fq.config.dtype]
        act_q, act_fq = _do_fake_quantize_affine(
            input,
            act_block_size,
            self.activation_fq.scale,
            self.activation_fq.zero_point,
            quant_dtype=torch.int32,
            quant_min=act_qmin,
            quant_max=act_qmax,
        )
        # print(act_q)
        act_fq = act_fq.reshape_as(input).to(input.dtype)
        torch.testing.assert_close(input_fq, act_fq, rtol=0, atol=0)

        # verify we can dequantize manually:
        #   lp = round(hp / scale + zp)
        #   (lp - zp) * scale = hp
        act_q_dq = (
            act_q - self.activation_fq.zero_point.unsqueeze(-1)
        ) * self.activation_fq.scale.unsqueeze(-1)
        torch.testing.assert_close(input_fq, act_q_dq, rtol=0, atol=0)

        # print('weight.shape', self.weight.shape)
        # print('weight_scale.shape', self.weight_fq.scale.shape)

        # fake quantize the weight again, this time also getting the quantize
        # values
        #
        # modified version of torchao.quantization.qat.utils::_fake_quantize_per_channel_group,
        # with `_fake_quantize_affine` replaced by `_do_fake_quantize_affine`
        #
        w_qmin, w_qmax = _DTYPE_TO_QVALUE_BOUNDS[self.weight_fq.config.dtype]
        w_granularity = self.weight_fq.config.granularity
        w_group_size = w_granularity.group_size
        w_block_size = (1, w_group_size)
        w_q, w_fq = _do_fake_quantize_affine(
            self.weight,
            w_block_size,
            self.weight_fq.scale,
            self.weight_fq.zero_point,
            quant_dtype=torch.int32,
            quant_min=w_qmin,
            quant_max=w_qmax,
            zero_point_domain=self.weight_fq.config.zero_point_domain,
        )
        # print('w_q', w_q)
        torch.testing.assert_close(weight_fq, w_fq, rtol=0, atol=0)

        # verify we can dequantize manually:
        #   lp = round(hp / scale + zp)
        #   (lp - zp) * scale = hp
        w_q_r = w_q.reshape(w_q.shape[0], w_q.shape[1] // w_group_size, w_group_size)
        w_q_dq = w_q_r * self.weight_fq.scale.unsqueeze(-1)
        w_q_dq = w_q_dq.reshape(w_fq.shape)
        torch.testing.assert_close(weight_fq, w_q_dq, rtol=0, atol=0)

        # cast the quantized tensors to int32 (right now they are floats)
        act_q = act_q.to(torch.int32)
        w_q = w_q.to(torch.int32)

        #
        # now we have the scales/zero_points/quant values for both gemm operands
        # below is a manual slow gemm with integer operands and float rescaling,
        # implemented using eager PyTorch ops. This should be slow but closely
        # (but not exactly) matching a real int8,int8->int32 gemm with
        # rescaling, with the only difference being that the sum inside of the
        # dot product is done in float32 right now.
        #
        q_output = torch.zeros(
            input.shape[0],
            self.weight.shape[0],
            dtype=torch.float32,
            device=input.device,
        )
        for m_idx in range(act_q.shape[0]):
            for n_idx in range(w_q.shape[0]):
                for g_idx in range(w_q.shape[1] // w_group_size):
                    k_start = g_idx * w_group_size
                    k_end = k_start + w_group_size
                    act_chunk = act_q[m_idx][k_start:k_end]
                    w_chunk = w_q[n_idx][k_start:k_end]
                    act_zp = self.activation_fq.zero_point[m_idx]
                    # print('k', k_start, k_end)
                    # print('act', act_chunk, act_chunk.dtype)
                    # print('w', w_chunk, w_chunk.dtype)

                    # print('elem', (act_chunk - act_zp) * w_chunk)
                    # (act_q - act_zp) * w_q
                    # result still in int32
                    elem_int32 = (act_chunk - act_zp) * w_chunk

                    # sum((act_q - act_zp) * w_q)
                    # this is in float32, so likely a small deviation from the real
                    # kernel, where the entire dot product would be in int32
                    sum_float32 = torch.sum(elem_int32)

                    # scale
                    act_scale = self.activation_fq.scale[m_idx]
                    w_scale = self.weight_fq.scale[n_idx][g_idx]
                    sum_scaled = sum_float32 * act_scale * w_scale

                    # accumulate
                    q_output[m_idx][n_idx] += sum_scaled

        fq_output = F.linear(input_fq, weight_fq, self.bias)
        sqnr_q_fq = compute_error(q_output, fq_output)
        assert sqnr_q_fq >= 100.0  # very close!
        return q_output

    @classmethod
    def from_float(cls, mod: torch.nn.Linear):
        new_mod = cls(mod.in_features, mod.out_features)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def run():
    M, K, N = 32, 64, 128

    # TODO(before land): also implement bias=True
    m_hp = nn.Sequential(nn.Linear(K, N, bias=False)).cuda()
    mq_ref = copy.deepcopy(m_hp)
    mq = copy.deepcopy(m_hp)

    # create a baseline: QAT with fake quants. Our exact QAT's output should
    # be close to this
    activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
    weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
    quantize_(
        mq_ref,
        IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
    )
    # print(mq_ref)

    # create the experiment: forward pass with an integer gemm
    mq[0] = Int8PerTokenActivationInt4PerGroupWeightLinear.from_float(mq[0])
    # print(mq)

    x_hp = torch.randn(M, K, device="cuda")
    xq_ref = copy.deepcopy(x_hp)
    xq = copy.deepcopy(x_hp)

    with torch.no_grad():
        y_hp = m_hp(x_hp)
        yq_ref = mq_ref(xq_ref)
        yq = mq(xq)
    # print(y_hp)
    # print(yq_ref)
    # print(yq)

    sqnr_hp_qref = compute_error(y_hp, yq_ref)
    sqnr_hp_q = compute_error(y_hp, yq)
    sqnr_qref_q = compute_error(yq_ref, yq)
    print("sqnr_hp_qref", sqnr_hp_qref)
    print("sqnr_hp_q", sqnr_hp_q)
    print("sqnr_qref_q", sqnr_qref_q)


if __name__ == "__main__":
    fire.Fire(run)
