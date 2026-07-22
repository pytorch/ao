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

from torchao.float8.float8_utils import compute_error
from torchao.prototype.qat_exact.reference_gemm import (
    cpu_x_token_assym_fp8_w_group_sym_int4_gemm,
    naive_x_token_assym_fp8_w_group_sym_int4_gemm,
)
from torchao.prototype.qat_exact.triton_gemm import int8_matmul_triton
from torchao.quantization import quantize_
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.qat.fake_quantizer import FakeQuantizer
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)

torch.manual_seed(0)


def quantize_x(x_fp32):
    # Dynamic quantization of activation
    x_mapping_type = MappingType.ASYMMETRIC
    per_token_block_size = _get_per_token_block_size(x_fp32)
    x_quant_min, x_quant_max = _DTYPE_TO_QVALUE_BOUNDS[torch.int8]
    x_eps = torch.finfo(torch.float32).eps
    x_scales_type = torch.float32
    x_zero_points_type = torch.int32
    x_scale, x_zero_point = torch.ops.torchao.choose_qparams_affine(
        x_fp32,
        x_mapping_type.name,
        per_token_block_size,
        torch.int8,
        x_quant_min,
        x_quant_max,
        x_eps,
        x_scales_type,
        x_zero_points_type,
    )
    x_i8 = torch.ops.torchao.quantize_affine(
        x_fp32,
        per_token_block_size,
        x_scale,
        x_zero_point,
        torch.int8,
        x_quant_min,
        x_quant_max,
    )
    return x_i8, x_scale, x_zero_point


class Int8PerTokenActivationInt4PerGroupWeightLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        gemm_mode = kwargs.pop("gemm_mode")
        assert gemm_mode in (
            "int8_naive_reference",
            "int8_cpu_reference",
            "int8_triton",
        )
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
        self.gemm_mode = gemm_mode

    def forward(self, input):
        # quantize x
        input_i8, input_scale, input_zp = quantize_x(input)

        # quantize w
        _ = self.weight_fq(self.weight)
        w_qmin, w_qmax = _DTYPE_TO_QVALUE_BOUNDS[torch.int4]
        w_granularity = self.weight_fq.config.granularity
        w_group_size = w_granularity.group_size
        w_block_size = (1, w_group_size)
        weight_int4 = torch.ops.torchao.quantize_affine(
            self.weight,
            w_block_size,
            self.weight_fq.scale,
            self.weight_fq.zero_point,
            torch.int8,
            w_qmin,
            w_qmax,
        )

        if self.gemm_mode == "int8_naive_reference":
            # original reference
            q_output = naive_x_token_assym_fp8_w_group_sym_int4_gemm(
                input_i8.to(torch.int32),
                input_scale,
                input_zp,
                weight_int4.to(torch.int32),
                self.weight_fq.scale,
                w_group_size,
            )
        elif self.gemm_mode == "int8_cpu_reference":
            # now also check Kimish's implementation
            q_output = cpu_x_token_assym_fp8_w_group_sym_int4_gemm(
                input_i8.cpu(),
                input_scale.cpu(),
                input_zp.cpu(),
                weight_int4.cpu(),
                self.weight_fq.scale.cpu(),
                self.weight_fq.zero_point.cpu(),
                self.bias,
                self.weight_fq.config.granularity.group_size,
            ).cuda()
        elif self.gemm_mode == "int8_triton":
            # finally, check vs triton gemm
            q_output = int8_matmul_triton(
                input_i8,
                weight_int4.t(),
                input_scale,
                input_zp,
                self.weight_fq.scale.t(),
                w_group_size,
            )

        return q_output

    @classmethod
    def from_float(cls, mod: torch.nn.Linear, gemm_mode: str):
        new_mod = cls(mod.in_features, mod.out_features, gemm_mode=gemm_mode)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def run():
    M, K, N = 32, 64, 128
    M, K, N = 64, 256, 1024

    # TODO(before land): also implement bias=True
    # m_hp = nn.Sequential(nn.Linear(K, N, bias=False)).cuda()
    # x_hp = torch.randn(M, K, device="cuda")

    # load from disk
    fname = '/home/vasiliy/local/tmp/20250711_input_list.pt'
    data = torch.load(fname)
    for data_idx in range(len(data)):
        print("data_idx", data_idx)
        x_from_data = data[data_idx]['x'].cuda()
        w_from_data = data[data_idx]['w_dq'].cuda()
        M, K = x_from_data.reshape(-1, x_from_data.shape[-1]).shape
        N, K2 = w_from_data.shape
        assert K2 == K
        print("MKN", M, K, N)
        m_hp = nn.Sequential(nn.Linear(K, N, bias=False)).cuda()
        m_hp[0].weight = nn.Parameter(w_from_data)
        x_hp = x_from_data.reshape(-1, x_from_data.shape[-1])

        mq_ref = copy.deepcopy(m_hp)
        mq_naive = copy.deepcopy(m_hp)
        mq_cpu = copy.deepcopy(m_hp)
        mq_triton = copy.deepcopy(m_hp)

        xq_ref = copy.deepcopy(x_hp)
        xq = copy.deepcopy(x_hp)


        # create a baseline: QAT with fake quants. Our exact QAT's output should
        # be close to this
        activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
        quantize_(
            mq_ref,
            IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
        )

        # create the experiment: forward pass with an integer gemm
        mq_naive[0] = Int8PerTokenActivationInt4PerGroupWeightLinear.from_float(
            mq_naive[0], "int8_naive_reference"
        )
        mq_cpu[0] = Int8PerTokenActivationInt4PerGroupWeightLinear.from_float(
            mq_cpu[0], "int8_cpu_reference"
        )
        mq_triton[0] = Int8PerTokenActivationInt4PerGroupWeightLinear.from_float(
            mq_triton[0], "int8_triton"
        )

        with torch.no_grad():
            # y_hp = m_hp(x_hp)
            yq_ref = mq_ref(xq_ref)
            # yq_naive = mq_naive(xq)
            yq_cpu = mq_cpu(xq)
            yq_triton = mq_triton(xq)

        # sqnr_hp_qref = compute_error(y_hp, yq_ref)
        # sqnr_hp_qnaive = compute_error(y_hp, yq_naive)
        # sqnr_qref_qnaive = compute_error(yq_ref, yq_naive)
        # sqnr_qcpu_qnaive = compute_error(yq_cpu, yq_naive)
        sqnr_yq_ref_qcpu = compute_error(yq_ref, yq_cpu)
        sqnr_qcpu_qtriton = compute_error(yq_cpu, yq_triton)
        # sqnr_qnaive_qtriton = compute_error(yq_naive, yq_triton)
        # print("sqnr_hp_qref", sqnr_hp_qref)
        # print("sqnr_hp_qnaive", sqnr_hp_qnaive)
        # print("sqnr_qref_qnaive", sqnr_qref_qnaive)
        # print("sqnr_qcpu_qnaive", sqnr_qcpu_qnaive)
        print("sqnr_yq_ref_qcpu", sqnr_yq_ref_qcpu)
        print("sqnr_qcpu_triton", sqnr_qcpu_qtriton)
        # print("sqnr_qnaive_qtriton", sqnr_qnaive_qtriton)
        print()


if __name__ == "__main__":
    fire.Fire(run)
