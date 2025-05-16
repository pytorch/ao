# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_cpuhrsch/o3/co3lwkqijt4ivkkwok725np6dimhuinsxblrq2aj6xprhiiazzp4.py
# Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# input_1 => add
# x_4 => convert_element_type, var_mean
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1182
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 6)
    x0 = xindex % 6
    x3 = xindex
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r2 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 197, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((196*r2) + (25088*x0) + (((-1) + x1) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr2 + (r2 + (128*x0)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp8, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp7, tmp15)
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight, roffset == 0
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
""",
    device_str="cuda",
)

from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.triton_heuristics import (
    grid,
)

# kernel path: /tmp/torchinductor_cpuhrsch/sc/csc6lxi5jpunalyhf7fubptfpzyvfgsm34m3zi4phomvke76t5gj.py
# Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# input_1 => add
# x_4 => convert_element_type, var_mean
triton_per_fused_add_cat_native_layer_norm_1 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 197
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ra/crabjmxg3icgamvajnimui7mqen7hcmpijpks4savxyqhidwm7io.py
# Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# input_1 => add
# x_4 => add_1, add_2, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_add_cat_native_layer_norm_2 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp17 = tl.load(in_ptr3 + (x2), xmask).to(tl.float32)
    tmp20 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x0) + (((-1) + x1) % 196)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp21 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp35, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/v2/cv2ukvhu7724i5ukaatadevmt3mmhuguajnahv6asb76bj7ywrj2.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
# l__self___encoder_layers_encoder_layer_0_self_attention => _scaled_dot_product_flash_attention
triton_poi_fused__scaled_dot_product_flash_attention_3 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2304*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/om/commpoqdtg4z75e2l3w2o2i2dsfi4uf2b4qf3vgzl67bthjcknet.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
# l__self___encoder_layers_encoder_layer_0_self_attention => _scaled_dot_product_flash_attention
triton_poi_fused__scaled_dot_product_flash_attention_4 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (2304*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (768 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/is/cisjxdclqxuup27b7pws7u6qjadwdkn5bu4mvd2fcfs7kicysoso.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
# l__self___encoder_layers_encoder_layer_0_self_attention => _scaled_dot_product_flash_attention
triton_poi_fused__scaled_dot_product_flash_attention_5 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (2304*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1536 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ga/cgajhmp5k2qgfqlv7izjnuj5xlliagddsoq5dayzuc7ney2ogmoh.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
# l__self___encoder_layers_encoder_layer_0_self_attention => abs_1, amax, clamp_max, clamp_min, clamp_min_1, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, div, div_1, round_1
triton_per_fused__to_copy_abs_amax_clamp_div_round_6 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i8', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_amax_clamp_div_round_6', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 127.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp0 / tmp11
    tmp13 = libdevice.nearbyint(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = -127.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = triton_helpers.minimum(tmp16, tmp10)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.int8)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/vn/cvncq77km77vud5hii36q6sqmlvhszw54xzurhgncixrcpkjlmog.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
# l__self___encoder_layers_encoder_layer_0_self_attention => clamp_max, clamp_min, clamp_min_1, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, div, div_1, fused_int_mm_mul_36, round_1, view_11
triton_tem_fused__to_copy_clamp_div_mul_round_view_7 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {0: '*i8', 1: '*i8', 2: '*bf16', 3: '*bf16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_clamp_div_mul_round_view_7', 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.int32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 197
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (768*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_m, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)
""",
    device_str="cuda",
)
import torch._inductor.kernel.mm_common

meta0 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.int32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
}


# kernel path: /tmp/torchinductor_cpuhrsch/q4/cq4zxhger5lhmqxvocvrj6a2rmv6dcydvchysqtp3emi3w2lwwcq.py
# Source Nodes: [cat_1, input_1, l__self___encoder_layers_encoder_layer_0_mlp_0, x_6, x_7, y], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.cat, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
# cat_1 => cat
# input_1 => add
# l__self___encoder_layers_encoder_layer_0_mlp_0 => abs_2, amax_1, clamp_max_1, clamp_min_2, clamp_min_3, convert_element_type_12, convert_element_type_13, convert_element_type_14, convert_element_type_15, convert_element_type_16, div_2, div_3, round_2
# x_6 => clone_2
# x_7 => add_4
# y => add_5, add_6, convert_element_type_10, convert_element_type_11, mul_4, mul_5, rsqrt_1, sub_1, var_mean_1
triton_per_fused__to_copy_abs_add_amax_cat_clamp_clone_div_native_layer_norm_round_8 = (
    async_compile.triton(
        "triton_",
        """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*i8', 10: '*bf16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_add_amax_cat_clamp_clone_div_native_layer_norm_round_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp5 >= tmp8
    tmp14 = tl.full([1], 197, tl.int64)
    tmp15 = tmp5 < tmp14
    tmp16 = tl.load(in_ptr3 + ((196*r1) + (((-1) + x0) % 196)), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp13, tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp12, tmp20)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp4 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 768, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 768.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-06
    tmp46 = tmp44 + tmp45
    tmp47 = libdevice.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp48 * tmp50
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp51 + tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tl_math.abs(tmp55)
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = tl.where(rmask & xmask, tmp57, float("-inf"))
    tmp60 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp59, 0))
    tmp61 = tmp60.to(tl.float32)
    tmp62 = 1e-05
    tmp63 = triton_helpers.maximum(tmp61, tmp62)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = 127.0
    tmp66 = tmp64 / tmp65
    tmp67 = tmp55 / tmp66
    tmp68 = libdevice.nearbyint(tmp67)
    tmp69 = tmp68.to(tl.float32)
    tmp70 = -127.0
    tmp71 = triton_helpers.maximum(tmp69, tmp70)
    tmp72 = triton_helpers.minimum(tmp71, tmp65)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73.to(tl.int8)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp24, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp74, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp66, xmask)
""",
        device_str="cuda",
    )
)


# kernel path: /tmp/torchinductor_cpuhrsch/of/coffxvswhxpdgy7a5dgqiwc4iqfjp2g7j5mmqio2n5amz6yt4hcb.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_0], Original ATen: [aten.mul]
# l__self___encoder_layers_encoder_layer_0_mlp_0 => fused_int_mm_mul_35
triton_tem_fused_mul_9 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.template(
    num_stages=3,
    num_warps=8,
    triton_meta={'signature': {0: '*i8', 1: '*i8', 2: '*bf16', 3: '*bf16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_mul_9', 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.int32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = 197
    N = 3072
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_m, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)
""",
    device_str="cuda",
)
meta1 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.int32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
}


# kernel path: /tmp/torchinductor_cpuhrsch/p5/cp5cr2vgwzpljz6jq7tcwqrm5tsrkitobsx2s7qug75km2wwrj5i.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_1, l__self___encoder_layers_encoder_layer_0_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
# l__self___encoder_layers_encoder_layer_0_mlp_1 => add_8, convert_element_type_17, convert_element_type_18, erf, mul_10, mul_8, mul_9
# l__self___encoder_layers_encoder_layer_0_mlp_3 => abs_3, amax_2, clamp_max_2, clamp_min_4, clamp_min_5, convert_element_type_19, convert_element_type_20, convert_element_type_21, convert_element_type_22, convert_element_type_23, div_4, div_5, round_3
triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*i8', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 197
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 0.5
        tmp7 = tmp5 * tmp6
        tmp8 = 0.7071067811865476
        tmp9 = tmp5 * tmp8
        tmp10 = libdevice.erf(tmp9)
        tmp11 = 1.0
        tmp12 = tmp10 + tmp11
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl_math.abs(tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = triton_helpers.maximum(_tmp17, tmp16)
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = triton_helpers.max2(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 0.5
        tmp26 = tmp24 * tmp25
        tmp27 = 0.7071067811865476
        tmp28 = tmp24 * tmp27
        tmp29 = libdevice.erf(tmp28)
        tmp30 = 1.0
        tmp31 = tmp29 + tmp30
        tmp32 = tmp26 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp17.to(tl.float32)
        tmp35 = 1e-05
        tmp36 = triton_helpers.maximum(tmp34, tmp35)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 127.0
        tmp39 = tmp37 / tmp38
        tmp40 = tmp33 / tmp39
        tmp41 = libdevice.nearbyint(tmp40)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = -127.0
        tmp44 = triton_helpers.maximum(tmp42, tmp43)
        tmp45 = triton_helpers.minimum(tmp44, tmp38)
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp46.to(tl.int8)
        tl.store(out_ptr1 + (r1 + (3072*x0)), tmp47, rmask & xmask)
    tmp48 = tmp17.to(tl.float32)
    tmp49 = 1e-05
    tmp50 = triton_helpers.maximum(tmp48, tmp49)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = 127.0
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr2 + (x0), tmp53, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ks/cks5rp2uy7gomyqyseduaqb56j3c2hyvr4netpbwhj6cu6y46plq.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_3], Original ATen: [aten.mul]
# l__self___encoder_layers_encoder_layer_0_mlp_3 => fused_int_mm_mul_34
triton_tem_fused_mul_11 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {0: '*i8', 1: '*i8', 2: '*bf16', 3: '*bf16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_mul_11', 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.int32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 197
    N = 768
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (768*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_m, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)
""",
    device_str="cuda",
)
meta2 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.int32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 32,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
}


# kernel path: /tmp/torchinductor_cpuhrsch/2f/c2fgqdaxjpk4kijg5wfswujk6damnjgzmmsket3spttgcg6fm42a.py
# Source Nodes: [add_2, x_8], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_10
# x_8 => add_11, add_12, convert_element_type_24, convert_element_type_25, mul_13, mul_14, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_native_layer_norm_12 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 768, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tmp7 - tmp17
    tmp25 = 768.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/6m/c6mnkwk3zrdagm6zf6feet7isza3vibvauvur5pgnwf3rrdcrfm5.py
# Source Nodes: [add_2, l__self___encoder_layers_encoder_layer_1_mlp_0, x_10, x_11, y_2], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
# add_2 => add_10
# l__self___encoder_layers_encoder_layer_1_mlp_0 => abs_5, amax_4, clamp_max_4, clamp_min_8, clamp_min_9, convert_element_type_36, convert_element_type_37, convert_element_type_38, convert_element_type_39, convert_element_type_40, div_8, div_9, round_5
# x_10 => clone_6
# x_11 => add_14
# y_2 => add_15, add_16, convert_element_type_34, convert_element_type_35, mul_17, mul_18, rsqrt_3, sub_3, var_mean_3
triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13 = (
    async_compile.triton(
        "triton_",
        """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*i8', 10: '*bf16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tmp4 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 768, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tmp13 - tmp23
    tmp31 = 768.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 * tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl_math.abs(tmp43)
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, float("-inf"))
    tmp48 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp47, 0))
    tmp49 = tmp48.to(tl.float32)
    tmp50 = 1e-05
    tmp51 = triton_helpers.maximum(tmp49, tmp50)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = 127.0
    tmp54 = tmp52 / tmp53
    tmp55 = tmp43 / tmp54
    tmp56 = libdevice.nearbyint(tmp55)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = -127.0
    tmp59 = triton_helpers.maximum(tmp57, tmp58)
    tmp60 = triton_helpers.minimum(tmp59, tmp53)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61.to(tl.int8)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp62, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp54, xmask)
""",
        device_str="cuda",
    )
)


# kernel path: /tmp/torchinductor_cpuhrsch/iz/cizsyx5toabafav5gknz4763c32d2m5w2kza2uucdfeovusxryby.py
# Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
# x_54 => abs_37, amax_36, clamp_max_36, clamp_min_72, clamp_min_73, convert_element_type_290, convert_element_type_291, convert_element_type_292, convert_element_type_293, convert_element_type_294, div_72, div_73, round_37
triton_per_fused__to_copy_abs_amax_clamp_div_round_14 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i8', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_amax_clamp_div_round_14', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 127.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp0 / tmp11
    tmp13 = libdevice.nearbyint(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = -127.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = triton_helpers.minimum(tmp16, tmp10)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.int8)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp19, rmask)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp11, None)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/oq/coqyzjtevk7ui5sq7ogzsmktecfzfzh65jcppygrnmzaf4dzp3t6.py
# Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
# x_54 => clamp_max_36, clamp_min_72, clamp_min_73, convert_element_type_290, convert_element_type_291, convert_element_type_292, convert_element_type_293, convert_element_type_294, div_72, div_73, fused_int_mm_mul, round_37, view_337
triton_poi_fused__to_copy_clamp_div_mul_round_view_15 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1024],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_div_mul_round_view_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tl.store(out_ptr0 + (x0), tmp1, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ej/cejk5r3nr6hqhc7ax46tvfts67vp7o4xgjeo3xpxc3726y2wabnr.py
# Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
# x_54 => add_123, clamp_max_36, clamp_min_72, clamp_min_73, convert_element_type_290, convert_element_type_291, convert_element_type_292, convert_element_type_293, convert_element_type_294, div_72, div_73, fused_int_mm_mul, mul_159, round_37, view_337
triton_tem_fused__to_copy_add_clamp_div_mul_round_view_16 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.template(
    num_stages=3,
    num_warps=8,
    triton_meta={'signature': {0: '*i8', 1: '*i8', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_clamp_div_mul_round_view_16', 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, out_ptr1):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.int32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 1000
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (1000*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr3 + (tl.broadcast_to(xindex, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr4 + (tl.broadcast_to(xindex, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = acc * tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, mask.shape)), tmp4, mask)
""",
    device_str="cuda",
)
meta3 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.int32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 16,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
}


async_compile.wait(globals())
del async_compile


def call(args):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        arg30_1,
        arg31_1,
        arg32_1,
        arg33_1,
        arg34_1,
        arg35_1,
        arg36_1,
        arg37_1,
        arg38_1,
        arg39_1,
        arg40_1,
        arg41_1,
        arg42_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg46_1,
        arg47_1,
        arg48_1,
        arg49_1,
        arg50_1,
        arg51_1,
        arg52_1,
        arg53_1,
        arg54_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg59_1,
        arg60_1,
        arg61_1,
        arg62_1,
        arg63_1,
        arg64_1,
        arg65_1,
        arg66_1,
        arg67_1,
        arg68_1,
        arg69_1,
        arg70_1,
        arg71_1,
        arg72_1,
        arg73_1,
        arg74_1,
        arg75_1,
        arg76_1,
        arg77_1,
        arg78_1,
        arg79_1,
        arg80_1,
        arg81_1,
        arg82_1,
        arg83_1,
        arg84_1,
        arg85_1,
        arg86_1,
        arg87_1,
        arg88_1,
        arg89_1,
        arg90_1,
        arg91_1,
        arg92_1,
        arg93_1,
        arg94_1,
        arg95_1,
        arg96_1,
        arg97_1,
        arg98_1,
        arg99_1,
        arg100_1,
        arg101_1,
        arg102_1,
        arg103_1,
        arg104_1,
        arg105_1,
        arg106_1,
        arg107_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
        arg112_1,
        arg113_1,
        arg114_1,
        arg115_1,
        arg116_1,
        arg117_1,
        arg118_1,
        arg119_1,
        arg120_1,
        arg121_1,
        arg122_1,
        arg123_1,
        arg124_1,
        arg125_1,
        arg126_1,
        arg127_1,
        arg128_1,
        arg129_1,
        arg130_1,
        arg131_1,
        arg132_1,
        arg133_1,
        arg134_1,
        arg135_1,
        arg136_1,
        arg137_1,
        arg138_1,
        arg139_1,
        arg140_1,
        arg141_1,
        arg142_1,
        arg143_1,
        arg144_1,
        arg145_1,
        arg146_1,
        arg147_1,
        arg148_1,
        arg149_1,
        arg150_1,
        arg151_1,
        arg152_1,
        arg153_1,
        arg154_1,
        arg155_1,
        arg156_1,
        arg157_1,
        arg158_1,
        arg159_1,
        arg160_1,
        arg161_1,
        arg162_1,
        arg163_1,
        arg164_1,
        arg165_1,
        arg166_1,
        arg167_1,
        arg168_1,
        arg169_1,
        arg170_1,
        arg171_1,
        arg172_1,
        arg173_1,
        arg174_1,
        arg175_1,
        arg176_1,
        arg177_1,
        arg178_1,
        arg179_1,
        arg180_1,
        arg181_1,
        arg182_1,
        arg183_1,
        arg184_1,
        arg185_1,
        arg186_1,
        arg187_1,
        arg188_1,
        arg189_1,
    ) = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg1_1, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(arg2_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg3_1, (768,), (1,))
    assert_size_stride(arg4_1, (768,), (1,))
    assert_size_stride(arg5_1, (768,), (1,))
    assert_size_stride(arg6_1, (2304, 768), (768, 1))
    assert_size_stride(arg7_1, (2304,), (1,))
    assert_size_stride(arg8_1, (768, 768), (1, 768))
    assert_size_stride(arg9_1, (768,), (1,))
    assert_size_stride(arg10_1, (768,), (1,))
    assert_size_stride(arg11_1, (768,), (1,))
    assert_size_stride(arg12_1, (768,), (1,))
    assert_size_stride(arg13_1, (768, 3072), (1, 768))
    assert_size_stride(arg14_1, (3072,), (1,))
    assert_size_stride(arg15_1, (3072,), (1,))
    assert_size_stride(arg16_1, (3072, 768), (1, 3072))
    assert_size_stride(arg17_1, (768,), (1,))
    assert_size_stride(arg18_1, (768,), (1,))
    assert_size_stride(arg19_1, (768,), (1,))
    assert_size_stride(arg20_1, (768,), (1,))
    assert_size_stride(arg21_1, (2304, 768), (768, 1))
    assert_size_stride(arg22_1, (2304,), (1,))
    assert_size_stride(arg23_1, (768, 768), (1, 768))
    assert_size_stride(arg24_1, (768,), (1,))
    assert_size_stride(arg25_1, (768,), (1,))
    assert_size_stride(arg26_1, (768,), (1,))
    assert_size_stride(arg27_1, (768,), (1,))
    assert_size_stride(arg28_1, (768, 3072), (1, 768))
    assert_size_stride(arg29_1, (3072,), (1,))
    assert_size_stride(arg30_1, (3072,), (1,))
    assert_size_stride(arg31_1, (3072, 768), (1, 3072))
    assert_size_stride(arg32_1, (768,), (1,))
    assert_size_stride(arg33_1, (768,), (1,))
    assert_size_stride(arg34_1, (768,), (1,))
    assert_size_stride(arg35_1, (768,), (1,))
    assert_size_stride(arg36_1, (2304, 768), (768, 1))
    assert_size_stride(arg37_1, (2304,), (1,))
    assert_size_stride(arg38_1, (768, 768), (1, 768))
    assert_size_stride(arg39_1, (768,), (1,))
    assert_size_stride(arg40_1, (768,), (1,))
    assert_size_stride(arg41_1, (768,), (1,))
    assert_size_stride(arg42_1, (768,), (1,))
    assert_size_stride(arg43_1, (768, 3072), (1, 768))
    assert_size_stride(arg44_1, (3072,), (1,))
    assert_size_stride(arg45_1, (3072,), (1,))
    assert_size_stride(arg46_1, (3072, 768), (1, 3072))
    assert_size_stride(arg47_1, (768,), (1,))
    assert_size_stride(arg48_1, (768,), (1,))
    assert_size_stride(arg49_1, (768,), (1,))
    assert_size_stride(arg50_1, (768,), (1,))
    assert_size_stride(arg51_1, (2304, 768), (768, 1))
    assert_size_stride(arg52_1, (2304,), (1,))
    assert_size_stride(arg53_1, (768, 768), (1, 768))
    assert_size_stride(arg54_1, (768,), (1,))
    assert_size_stride(arg55_1, (768,), (1,))
    assert_size_stride(arg56_1, (768,), (1,))
    assert_size_stride(arg57_1, (768,), (1,))
    assert_size_stride(arg58_1, (768, 3072), (1, 768))
    assert_size_stride(arg59_1, (3072,), (1,))
    assert_size_stride(arg60_1, (3072,), (1,))
    assert_size_stride(arg61_1, (3072, 768), (1, 3072))
    assert_size_stride(arg62_1, (768,), (1,))
    assert_size_stride(arg63_1, (768,), (1,))
    assert_size_stride(arg64_1, (768,), (1,))
    assert_size_stride(arg65_1, (768,), (1,))
    assert_size_stride(arg66_1, (2304, 768), (768, 1))
    assert_size_stride(arg67_1, (2304,), (1,))
    assert_size_stride(arg68_1, (768, 768), (1, 768))
    assert_size_stride(arg69_1, (768,), (1,))
    assert_size_stride(arg70_1, (768,), (1,))
    assert_size_stride(arg71_1, (768,), (1,))
    assert_size_stride(arg72_1, (768,), (1,))
    assert_size_stride(arg73_1, (768, 3072), (1, 768))
    assert_size_stride(arg74_1, (3072,), (1,))
    assert_size_stride(arg75_1, (3072,), (1,))
    assert_size_stride(arg76_1, (3072, 768), (1, 3072))
    assert_size_stride(arg77_1, (768,), (1,))
    assert_size_stride(arg78_1, (768,), (1,))
    assert_size_stride(arg79_1, (768,), (1,))
    assert_size_stride(arg80_1, (768,), (1,))
    assert_size_stride(arg81_1, (2304, 768), (768, 1))
    assert_size_stride(arg82_1, (2304,), (1,))
    assert_size_stride(arg83_1, (768, 768), (1, 768))
    assert_size_stride(arg84_1, (768,), (1,))
    assert_size_stride(arg85_1, (768,), (1,))
    assert_size_stride(arg86_1, (768,), (1,))
    assert_size_stride(arg87_1, (768,), (1,))
    assert_size_stride(arg88_1, (768, 3072), (1, 768))
    assert_size_stride(arg89_1, (3072,), (1,))
    assert_size_stride(arg90_1, (3072,), (1,))
    assert_size_stride(arg91_1, (3072, 768), (1, 3072))
    assert_size_stride(arg92_1, (768,), (1,))
    assert_size_stride(arg93_1, (768,), (1,))
    assert_size_stride(arg94_1, (768,), (1,))
    assert_size_stride(arg95_1, (768,), (1,))
    assert_size_stride(arg96_1, (2304, 768), (768, 1))
    assert_size_stride(arg97_1, (2304,), (1,))
    assert_size_stride(arg98_1, (768, 768), (1, 768))
    assert_size_stride(arg99_1, (768,), (1,))
    assert_size_stride(arg100_1, (768,), (1,))
    assert_size_stride(arg101_1, (768,), (1,))
    assert_size_stride(arg102_1, (768,), (1,))
    assert_size_stride(arg103_1, (768, 3072), (1, 768))
    assert_size_stride(arg104_1, (3072,), (1,))
    assert_size_stride(arg105_1, (3072,), (1,))
    assert_size_stride(arg106_1, (3072, 768), (1, 3072))
    assert_size_stride(arg107_1, (768,), (1,))
    assert_size_stride(arg108_1, (768,), (1,))
    assert_size_stride(arg109_1, (768,), (1,))
    assert_size_stride(arg110_1, (768,), (1,))
    assert_size_stride(arg111_1, (2304, 768), (768, 1))
    assert_size_stride(arg112_1, (2304,), (1,))
    assert_size_stride(arg113_1, (768, 768), (1, 768))
    assert_size_stride(arg114_1, (768,), (1,))
    assert_size_stride(arg115_1, (768,), (1,))
    assert_size_stride(arg116_1, (768,), (1,))
    assert_size_stride(arg117_1, (768,), (1,))
    assert_size_stride(arg118_1, (768, 3072), (1, 768))
    assert_size_stride(arg119_1, (3072,), (1,))
    assert_size_stride(arg120_1, (3072,), (1,))
    assert_size_stride(arg121_1, (3072, 768), (1, 3072))
    assert_size_stride(arg122_1, (768,), (1,))
    assert_size_stride(arg123_1, (768,), (1,))
    assert_size_stride(arg124_1, (768,), (1,))
    assert_size_stride(arg125_1, (768,), (1,))
    assert_size_stride(arg126_1, (2304, 768), (768, 1))
    assert_size_stride(arg127_1, (2304,), (1,))
    assert_size_stride(arg128_1, (768, 768), (1, 768))
    assert_size_stride(arg129_1, (768,), (1,))
    assert_size_stride(arg130_1, (768,), (1,))
    assert_size_stride(arg131_1, (768,), (1,))
    assert_size_stride(arg132_1, (768,), (1,))
    assert_size_stride(arg133_1, (768, 3072), (1, 768))
    assert_size_stride(arg134_1, (3072,), (1,))
    assert_size_stride(arg135_1, (3072,), (1,))
    assert_size_stride(arg136_1, (3072, 768), (1, 3072))
    assert_size_stride(arg137_1, (768,), (1,))
    assert_size_stride(arg138_1, (768,), (1,))
    assert_size_stride(arg139_1, (768,), (1,))
    assert_size_stride(arg140_1, (768,), (1,))
    assert_size_stride(arg141_1, (2304, 768), (768, 1))
    assert_size_stride(arg142_1, (2304,), (1,))
    assert_size_stride(arg143_1, (768, 768), (1, 768))
    assert_size_stride(arg144_1, (768,), (1,))
    assert_size_stride(arg145_1, (768,), (1,))
    assert_size_stride(arg146_1, (768,), (1,))
    assert_size_stride(arg147_1, (768,), (1,))
    assert_size_stride(arg148_1, (768, 3072), (1, 768))
    assert_size_stride(arg149_1, (3072,), (1,))
    assert_size_stride(arg150_1, (3072,), (1,))
    assert_size_stride(arg151_1, (3072, 768), (1, 3072))
    assert_size_stride(arg152_1, (768,), (1,))
    assert_size_stride(arg153_1, (768,), (1,))
    assert_size_stride(arg154_1, (768,), (1,))
    assert_size_stride(arg155_1, (768,), (1,))
    assert_size_stride(arg156_1, (2304, 768), (768, 1))
    assert_size_stride(arg157_1, (2304,), (1,))
    assert_size_stride(arg158_1, (768, 768), (1, 768))
    assert_size_stride(arg159_1, (768,), (1,))
    assert_size_stride(arg160_1, (768,), (1,))
    assert_size_stride(arg161_1, (768,), (1,))
    assert_size_stride(arg162_1, (768,), (1,))
    assert_size_stride(arg163_1, (768, 3072), (1, 768))
    assert_size_stride(arg164_1, (3072,), (1,))
    assert_size_stride(arg165_1, (3072,), (1,))
    assert_size_stride(arg166_1, (3072, 768), (1, 3072))
    assert_size_stride(arg167_1, (768,), (1,))
    assert_size_stride(arg168_1, (768,), (1,))
    assert_size_stride(arg169_1, (768,), (1,))
    assert_size_stride(arg170_1, (768,), (1,))
    assert_size_stride(arg171_1, (2304, 768), (768, 1))
    assert_size_stride(arg172_1, (2304,), (1,))
    assert_size_stride(arg173_1, (768, 768), (1, 768))
    assert_size_stride(arg174_1, (768,), (1,))
    assert_size_stride(arg175_1, (768,), (1,))
    assert_size_stride(arg176_1, (768,), (1,))
    assert_size_stride(arg177_1, (768,), (1,))
    assert_size_stride(arg178_1, (768, 3072), (1, 768))
    assert_size_stride(arg179_1, (3072,), (1,))
    assert_size_stride(arg180_1, (3072,), (1,))
    assert_size_stride(arg181_1, (3072, 768), (1, 3072))
    assert_size_stride(arg182_1, (768,), (1,))
    assert_size_stride(arg183_1, (768,), (1,))
    assert_size_stride(arg184_1, (768,), (1,))
    assert_size_stride(arg185_1, (768,), (1,))
    assert_size_stride(arg186_1, (768, 1000), (1, 768))
    assert_size_stride(arg187_1, (1000,), (1,))
    assert_size_stride(arg188_1, (1000,), (1,))
    assert_size_stride(arg189_1, (1, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(
            arg189_1,
            arg2_1,
            stride=(16, 16),
            padding=(0, 0),
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
            bias=None,
        )
        assert_size_stride(buf0, (1, 768, 14, 14), (150528, 196, 14, 1))
        del arg189_1
        del arg2_1
        buf1 = empty_strided_cuda((1, 197, 1, 6), (1182, 6, 1182, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 197, 1, 6), (1182, 6, 1182, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 197, 1, 6), (1182, 6, 1182, 1), torch.float32)
        # Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(
            arg0_1,
            buf0,
            arg3_1,
            arg1_1,
            buf1,
            buf2,
            buf3,
            1182,
            128,
            grid=grid(1182),
            stream=stream0,
        )
        buf4 = empty_strided_cuda((1, 197, 1), (197, 1, 197), torch.float32)
        buf5 = empty_strided_cuda((1, 197, 1), (197, 1, 197), torch.float32)
        # Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_1.run(
            buf1, buf2, buf3, buf4, buf5, 197, 6, grid=grid(197), stream=stream0
        )
        del buf1
        del buf2
        del buf3
        buf8 = empty_strided_cuda((1, 197, 768), (151296, 768, 1), torch.bfloat16)
        # Source Nodes: [cat_1, input_1, x_4], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_2.run(
            arg0_1,
            buf0,
            arg3_1,
            arg1_1,
            buf4,
            buf5,
            arg4_1,
            arg5_1,
            buf8,
            151296,
            grid=grid(151296),
            stream=stream0,
        )
        del arg4_1
        del arg5_1
        del buf4
        del buf5
        buf9 = empty_strided_cuda((197, 2304), (2304, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf8, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg6_1, (768, 2304), (1, 768), 0),
            out=buf9,
        )
        del arg6_1
        buf10 = reinterpret_tensor(buf8, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf8  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf9, arg7_1, buf10, 151296, grid=grid(151296), stream=stream0
        )
        buf11 = empty_strided_cuda(
            (1, 12, 197, 64), (151296, 64, 768, 1), torch.bfloat16
        )
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf9, arg7_1, buf11, 151296, grid=grid(151296), stream=stream0
        )
        buf12 = empty_strided_cuda(
            (1, 12, 197, 64), (151296, 64, 768, 1), torch.bfloat16
        )
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf9, arg7_1, buf12, 151296, grid=grid(151296), stream=stream0
        )
        del arg7_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf13 = aten._scaled_dot_product_flash_attention.default(
            buf10, buf11, buf12, scale=0.125
        )
        buf14 = buf13[0]
        del buf13
        buf20 = empty_strided_cuda((197, 768), (768, 1), torch.int8)
        buf21 = empty_strided_cuda((197, 1), (1, 1), torch.bfloat16)
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf14, buf20, buf21, 197, 768, grid=grid(197), stream=stream0
        )
        buf22 = reinterpret_tensor(buf14, (197, 768), (768, 1), 0)
        del buf14  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf20,
            arg8_1,
            buf21,
            buf22,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg8_1
        buf23 = reinterpret_tensor(buf22, (1, 197, 768), (151296, 768, 1), 0)
        del buf22  # reuse
        buf29 = reinterpret_tensor(buf20, (1, 197, 768), (151296, 768, 1), 0)
        del buf20  # reuse
        buf30 = reinterpret_tensor(buf21, (1, 197, 1), (197, 1, 1), 0)
        del buf21  # reuse
        # Source Nodes: [cat_1, input_1, l__self___encoder_layers_encoder_layer_0_mlp_0, x_6, x_7, y], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.cat, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_cat_clamp_clone_div_native_layer_norm_round_8.run(
            buf23,
            arg9_1,
            arg10_1,
            arg0_1,
            buf0,
            arg3_1,
            arg1_1,
            arg11_1,
            arg12_1,
            buf29,
            buf30,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg1_1
        del arg3_1
        del arg9_1
        del buf0
        buf31 = empty_strided_cuda((197, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf29,
            arg13_1,
            buf30,
            buf31,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg13_1
        buf33 = empty_strided_cuda((1, 197, 3072), (605184, 3072, 1), torch.int8)
        buf34 = buf30
        del buf30  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_1, l__self___encoder_layers_encoder_layer_0_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf31,
            arg14_1,
            arg15_1,
            buf33,
            buf34,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg14_1
        del arg15_1
        buf35 = reinterpret_tensor(buf12, (197, 768), (768, 1), 0)
        del buf12  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf33,
            arg16_1,
            buf34,
            buf35,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg16_1
        buf39 = reinterpret_tensor(buf11, (1, 197, 768), (151296, 768, 1), 0)
        del buf11  # reuse
        # Source Nodes: [add_2, x_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf23,
            buf35,
            arg17_1,
            arg18_1,
            arg19_1,
            arg20_1,
            buf39,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg19_1
        del arg20_1
        buf40 = buf9
        del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf39, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg21_1, (768, 2304), (1, 768), 0),
            out=buf40,
        )
        del arg21_1
        buf41 = reinterpret_tensor(buf39, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf39  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf40, arg22_1, buf41, 151296, grid=grid(151296), stream=stream0
        )
        buf42 = buf10
        del buf10  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf40, arg22_1, buf42, 151296, grid=grid(151296), stream=stream0
        )
        buf43 = empty_strided_cuda(
            (1, 12, 197, 64), (151296, 64, 768, 1), torch.bfloat16
        )
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf40, arg22_1, buf43, 151296, grid=grid(151296), stream=stream0
        )
        del arg22_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf44 = aten._scaled_dot_product_flash_attention.default(
            buf41, buf42, buf43, scale=0.125
        )
        del buf41
        buf45 = buf44[0]
        del buf44
        buf51 = reinterpret_tensor(buf29, (197, 768), (768, 1), 0)
        del buf29  # reuse
        buf52 = reinterpret_tensor(buf34, (197, 1), (1, 1), 0)
        del buf34  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf45, buf51, buf52, 197, 768, grid=grid(197), stream=stream0
        )
        buf53 = reinterpret_tensor(buf45, (197, 768), (768, 1), 0)
        del buf45  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf51,
            arg23_1,
            buf52,
            buf53,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg23_1
        buf54 = reinterpret_tensor(buf53, (1, 197, 768), (151296, 768, 1), 0)
        del buf53  # reuse
        buf60 = reinterpret_tensor(buf51, (1, 197, 768), (151296, 768, 1), 0)
        del buf51  # reuse
        buf61 = reinterpret_tensor(buf52, (1, 197, 1), (197, 1, 1), 0)
        del buf52  # reuse
        # Source Nodes: [add_2, l__self___encoder_layers_encoder_layer_1_mlp_0, x_10, x_11, y_2], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf54,
            arg24_1,
            arg25_1,
            buf23,
            buf35,
            arg17_1,
            arg18_1,
            arg26_1,
            arg27_1,
            buf60,
            buf61,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg17_1
        del arg18_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        buf62 = buf31
        del buf31  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf60,
            arg28_1,
            buf61,
            buf62,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg28_1
        buf64 = buf33
        del buf33  # reuse
        buf65 = buf61
        del buf61  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_mlp_1, l__self___encoder_layers_encoder_layer_1_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf62,
            arg29_1,
            arg30_1,
            buf64,
            buf65,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg29_1
        del arg30_1
        buf66 = buf35
        del buf35  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf64,
            arg31_1,
            buf65,
            buf66,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg31_1
        buf70 = buf23
        del buf23  # reuse
        # Source Nodes: [add_4, x_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf54,
            buf66,
            arg32_1,
            arg33_1,
            arg34_1,
            arg35_1,
            buf70,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg34_1
        del arg35_1
        buf71 = buf40
        del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf70, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg36_1, (768, 2304), (1, 768), 0),
            out=buf71,
        )
        del arg36_1
        buf72 = reinterpret_tensor(buf70, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf70  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf71, arg37_1, buf72, 151296, grid=grid(151296), stream=stream0
        )
        buf73 = buf43
        del buf43  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf71, arg37_1, buf73, 151296, grid=grid(151296), stream=stream0
        )
        buf74 = buf42
        del buf42  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf71, arg37_1, buf74, 151296, grid=grid(151296), stream=stream0
        )
        del arg37_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf75 = aten._scaled_dot_product_flash_attention.default(
            buf72, buf73, buf74, scale=0.125
        )
        del buf72
        buf76 = buf75[0]
        del buf75
        buf82 = reinterpret_tensor(buf60, (197, 768), (768, 1), 0)
        del buf60  # reuse
        buf83 = reinterpret_tensor(buf65, (197, 1), (1, 1), 0)
        del buf65  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf76, buf82, buf83, 197, 768, grid=grid(197), stream=stream0
        )
        buf84 = reinterpret_tensor(buf76, (197, 768), (768, 1), 0)
        del buf76  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf82,
            arg38_1,
            buf83,
            buf84,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg38_1
        buf85 = reinterpret_tensor(buf84, (1, 197, 768), (151296, 768, 1), 0)
        del buf84  # reuse
        buf91 = reinterpret_tensor(buf82, (1, 197, 768), (151296, 768, 1), 0)
        del buf82  # reuse
        buf92 = reinterpret_tensor(buf83, (1, 197, 1), (197, 1, 1), 0)
        del buf83  # reuse
        # Source Nodes: [add_4, l__self___encoder_layers_encoder_layer_2_mlp_0, x_14, x_15, y_4], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf85,
            arg39_1,
            arg40_1,
            buf54,
            buf66,
            arg32_1,
            arg33_1,
            arg41_1,
            arg42_1,
            buf91,
            buf92,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg32_1
        del arg33_1
        del arg39_1
        del arg40_1
        del arg41_1
        del arg42_1
        buf93 = buf62
        del buf62  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf91,
            arg43_1,
            buf92,
            buf93,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg43_1
        buf95 = buf64
        del buf64  # reuse
        buf96 = buf92
        del buf92  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_mlp_1, l__self___encoder_layers_encoder_layer_2_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf93,
            arg44_1,
            arg45_1,
            buf95,
            buf96,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg44_1
        del arg45_1
        buf97 = buf66
        del buf66  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf95,
            arg46_1,
            buf96,
            buf97,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg46_1
        buf101 = buf54
        del buf54  # reuse
        # Source Nodes: [add_6, x_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf85,
            buf97,
            arg47_1,
            arg48_1,
            arg49_1,
            arg50_1,
            buf101,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg49_1
        del arg50_1
        buf102 = buf71
        del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf101, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg51_1, (768, 2304), (1, 768), 0),
            out=buf102,
        )
        del arg51_1
        buf103 = reinterpret_tensor(buf101, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf101  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf102, arg52_1, buf103, 151296, grid=grid(151296), stream=stream0
        )
        buf104 = buf74
        del buf74  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf102, arg52_1, buf104, 151296, grid=grid(151296), stream=stream0
        )
        buf105 = buf73
        del buf73  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf102, arg52_1, buf105, 151296, grid=grid(151296), stream=stream0
        )
        del arg52_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf106 = aten._scaled_dot_product_flash_attention.default(
            buf103, buf104, buf105, scale=0.125
        )
        del buf103
        buf107 = buf106[0]
        del buf106
        buf113 = reinterpret_tensor(buf91, (197, 768), (768, 1), 0)
        del buf91  # reuse
        buf114 = reinterpret_tensor(buf96, (197, 1), (1, 1), 0)
        del buf96  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf107, buf113, buf114, 197, 768, grid=grid(197), stream=stream0
        )
        buf115 = reinterpret_tensor(buf107, (197, 768), (768, 1), 0)
        del buf107  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf113,
            arg53_1,
            buf114,
            buf115,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg53_1
        buf116 = reinterpret_tensor(buf115, (1, 197, 768), (151296, 768, 1), 0)
        del buf115  # reuse
        buf122 = reinterpret_tensor(buf113, (1, 197, 768), (151296, 768, 1), 0)
        del buf113  # reuse
        buf123 = reinterpret_tensor(buf114, (1, 197, 1), (197, 1, 1), 0)
        del buf114  # reuse
        # Source Nodes: [add_6, l__self___encoder_layers_encoder_layer_3_mlp_0, x_18, x_19, y_6], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf116,
            arg54_1,
            arg55_1,
            buf85,
            buf97,
            arg47_1,
            arg48_1,
            arg56_1,
            arg57_1,
            buf122,
            buf123,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg47_1
        del arg48_1
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        buf124 = buf93
        del buf93  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf122,
            arg58_1,
            buf123,
            buf124,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg58_1
        buf126 = buf95
        del buf95  # reuse
        buf127 = buf123
        del buf123  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_mlp_1, l__self___encoder_layers_encoder_layer_3_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf124,
            arg59_1,
            arg60_1,
            buf126,
            buf127,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg59_1
        del arg60_1
        buf128 = buf97
        del buf97  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf126,
            arg61_1,
            buf127,
            buf128,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg61_1
        buf132 = buf85
        del buf85  # reuse
        # Source Nodes: [add_8, x_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf116,
            buf128,
            arg62_1,
            arg63_1,
            arg64_1,
            arg65_1,
            buf132,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg64_1
        del arg65_1
        buf133 = buf102
        del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf132, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg66_1, (768, 2304), (1, 768), 0),
            out=buf133,
        )
        del arg66_1
        buf134 = reinterpret_tensor(buf132, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf132  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf133, arg67_1, buf134, 151296, grid=grid(151296), stream=stream0
        )
        buf135 = buf105
        del buf105  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf133, arg67_1, buf135, 151296, grid=grid(151296), stream=stream0
        )
        buf136 = buf104
        del buf104  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf133, arg67_1, buf136, 151296, grid=grid(151296), stream=stream0
        )
        del arg67_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf137 = aten._scaled_dot_product_flash_attention.default(
            buf134, buf135, buf136, scale=0.125
        )
        del buf134
        buf138 = buf137[0]
        del buf137
        buf144 = reinterpret_tensor(buf122, (197, 768), (768, 1), 0)
        del buf122  # reuse
        buf145 = reinterpret_tensor(buf127, (197, 1), (1, 1), 0)
        del buf127  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf138, buf144, buf145, 197, 768, grid=grid(197), stream=stream0
        )
        buf146 = reinterpret_tensor(buf138, (197, 768), (768, 1), 0)
        del buf138  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf144,
            arg68_1,
            buf145,
            buf146,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg68_1
        buf147 = reinterpret_tensor(buf146, (1, 197, 768), (151296, 768, 1), 0)
        del buf146  # reuse
        buf153 = reinterpret_tensor(buf144, (1, 197, 768), (151296, 768, 1), 0)
        del buf144  # reuse
        buf154 = reinterpret_tensor(buf145, (1, 197, 1), (197, 1, 1), 0)
        del buf145  # reuse
        # Source Nodes: [add_8, l__self___encoder_layers_encoder_layer_4_mlp_0, x_22, x_23, y_8], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf147,
            arg69_1,
            arg70_1,
            buf116,
            buf128,
            arg62_1,
            arg63_1,
            arg71_1,
            arg72_1,
            buf153,
            buf154,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg62_1
        del arg63_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        buf155 = buf124
        del buf124  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf153,
            arg73_1,
            buf154,
            buf155,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg73_1
        buf157 = buf126
        del buf126  # reuse
        buf158 = buf154
        del buf154  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_mlp_1, l__self___encoder_layers_encoder_layer_4_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf155,
            arg74_1,
            arg75_1,
            buf157,
            buf158,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg74_1
        del arg75_1
        buf159 = buf128
        del buf128  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf157,
            arg76_1,
            buf158,
            buf159,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg76_1
        buf163 = buf116
        del buf116  # reuse
        # Source Nodes: [add_10, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf147,
            buf159,
            arg77_1,
            arg78_1,
            arg79_1,
            arg80_1,
            buf163,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg79_1
        del arg80_1
        buf164 = buf133
        del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf163, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg81_1, (768, 2304), (1, 768), 0),
            out=buf164,
        )
        del arg81_1
        buf165 = reinterpret_tensor(buf163, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf163  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf164, arg82_1, buf165, 151296, grid=grid(151296), stream=stream0
        )
        buf166 = buf136
        del buf136  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf164, arg82_1, buf166, 151296, grid=grid(151296), stream=stream0
        )
        buf167 = buf135
        del buf135  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf164, arg82_1, buf167, 151296, grid=grid(151296), stream=stream0
        )
        del arg82_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf168 = aten._scaled_dot_product_flash_attention.default(
            buf165, buf166, buf167, scale=0.125
        )
        del buf165
        buf169 = buf168[0]
        del buf168
        buf175 = reinterpret_tensor(buf153, (197, 768), (768, 1), 0)
        del buf153  # reuse
        buf176 = reinterpret_tensor(buf158, (197, 1), (1, 1), 0)
        del buf158  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf169, buf175, buf176, 197, 768, grid=grid(197), stream=stream0
        )
        buf177 = reinterpret_tensor(buf169, (197, 768), (768, 1), 0)
        del buf169  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf175,
            arg83_1,
            buf176,
            buf177,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg83_1
        buf178 = reinterpret_tensor(buf177, (1, 197, 768), (151296, 768, 1), 0)
        del buf177  # reuse
        buf184 = reinterpret_tensor(buf175, (1, 197, 768), (151296, 768, 1), 0)
        del buf175  # reuse
        buf185 = reinterpret_tensor(buf176, (1, 197, 1), (197, 1, 1), 0)
        del buf176  # reuse
        # Source Nodes: [add_10, l__self___encoder_layers_encoder_layer_5_mlp_0, x_26, x_27, y_10], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf178,
            arg84_1,
            arg85_1,
            buf147,
            buf159,
            arg77_1,
            arg78_1,
            arg86_1,
            arg87_1,
            buf184,
            buf185,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg77_1
        del arg78_1
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        buf186 = buf155
        del buf155  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf184,
            arg88_1,
            buf185,
            buf186,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg88_1
        buf188 = buf157
        del buf157  # reuse
        buf189 = buf185
        del buf185  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_mlp_1, l__self___encoder_layers_encoder_layer_5_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf186,
            arg89_1,
            arg90_1,
            buf188,
            buf189,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg89_1
        del arg90_1
        buf190 = buf159
        del buf159  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf188,
            arg91_1,
            buf189,
            buf190,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg91_1
        buf194 = buf147
        del buf147  # reuse
        # Source Nodes: [add_12, x_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf178,
            buf190,
            arg92_1,
            arg93_1,
            arg94_1,
            arg95_1,
            buf194,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg94_1
        del arg95_1
        buf195 = buf164
        del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf194, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg96_1, (768, 2304), (1, 768), 0),
            out=buf195,
        )
        del arg96_1
        buf196 = reinterpret_tensor(buf194, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf194  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf195, arg97_1, buf196, 151296, grid=grid(151296), stream=stream0
        )
        buf197 = buf167
        del buf167  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf195, arg97_1, buf197, 151296, grid=grid(151296), stream=stream0
        )
        buf198 = buf166
        del buf166  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf195, arg97_1, buf198, 151296, grid=grid(151296), stream=stream0
        )
        del arg97_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf199 = aten._scaled_dot_product_flash_attention.default(
            buf196, buf197, buf198, scale=0.125
        )
        del buf196
        buf200 = buf199[0]
        del buf199
        buf206 = reinterpret_tensor(buf184, (197, 768), (768, 1), 0)
        del buf184  # reuse
        buf207 = reinterpret_tensor(buf189, (197, 1), (1, 1), 0)
        del buf189  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf200, buf206, buf207, 197, 768, grid=grid(197), stream=stream0
        )
        buf208 = reinterpret_tensor(buf200, (197, 768), (768, 1), 0)
        del buf200  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf206,
            arg98_1,
            buf207,
            buf208,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg98_1
        buf209 = reinterpret_tensor(buf208, (1, 197, 768), (151296, 768, 1), 0)
        del buf208  # reuse
        buf215 = reinterpret_tensor(buf206, (1, 197, 768), (151296, 768, 1), 0)
        del buf206  # reuse
        buf216 = reinterpret_tensor(buf207, (1, 197, 1), (197, 1, 1), 0)
        del buf207  # reuse
        # Source Nodes: [add_12, l__self___encoder_layers_encoder_layer_6_mlp_0, x_30, x_31, y_12], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf209,
            arg99_1,
            arg100_1,
            buf178,
            buf190,
            arg92_1,
            arg93_1,
            arg101_1,
            arg102_1,
            buf215,
            buf216,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg100_1
        del arg101_1
        del arg102_1
        del arg92_1
        del arg93_1
        del arg99_1
        buf217 = buf186
        del buf186  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf215,
            arg103_1,
            buf216,
            buf217,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg103_1
        buf219 = buf188
        del buf188  # reuse
        buf220 = buf216
        del buf216  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_mlp_1, l__self___encoder_layers_encoder_layer_6_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf217,
            arg104_1,
            arg105_1,
            buf219,
            buf220,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg104_1
        del arg105_1
        buf221 = buf190
        del buf190  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf219,
            arg106_1,
            buf220,
            buf221,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg106_1
        buf225 = buf178
        del buf178  # reuse
        # Source Nodes: [add_14, x_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf209,
            buf221,
            arg107_1,
            arg108_1,
            arg109_1,
            arg110_1,
            buf225,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg109_1
        del arg110_1
        buf226 = buf195
        del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf225, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg111_1, (768, 2304), (1, 768), 0),
            out=buf226,
        )
        del arg111_1
        buf227 = reinterpret_tensor(buf225, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf225  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf226, arg112_1, buf227, 151296, grid=grid(151296), stream=stream0
        )
        buf228 = buf198
        del buf198  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf226, arg112_1, buf228, 151296, grid=grid(151296), stream=stream0
        )
        buf229 = buf197
        del buf197  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf226, arg112_1, buf229, 151296, grid=grid(151296), stream=stream0
        )
        del arg112_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf230 = aten._scaled_dot_product_flash_attention.default(
            buf227, buf228, buf229, scale=0.125
        )
        del buf227
        buf231 = buf230[0]
        del buf230
        buf237 = reinterpret_tensor(buf215, (197, 768), (768, 1), 0)
        del buf215  # reuse
        buf238 = reinterpret_tensor(buf220, (197, 1), (1, 1), 0)
        del buf220  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf231, buf237, buf238, 197, 768, grid=grid(197), stream=stream0
        )
        buf239 = reinterpret_tensor(buf231, (197, 768), (768, 1), 0)
        del buf231  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf237,
            arg113_1,
            buf238,
            buf239,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg113_1
        buf240 = reinterpret_tensor(buf239, (1, 197, 768), (151296, 768, 1), 0)
        del buf239  # reuse
        buf246 = reinterpret_tensor(buf237, (1, 197, 768), (151296, 768, 1), 0)
        del buf237  # reuse
        buf247 = reinterpret_tensor(buf238, (1, 197, 1), (197, 1, 1), 0)
        del buf238  # reuse
        # Source Nodes: [add_14, l__self___encoder_layers_encoder_layer_7_mlp_0, x_34, x_35, y_14], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf240,
            arg114_1,
            arg115_1,
            buf209,
            buf221,
            arg107_1,
            arg108_1,
            arg116_1,
            arg117_1,
            buf246,
            buf247,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg107_1
        del arg108_1
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf248 = buf217
        del buf217  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf246,
            arg118_1,
            buf247,
            buf248,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg118_1
        buf250 = buf219
        del buf219  # reuse
        buf251 = buf247
        del buf247  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_mlp_1, l__self___encoder_layers_encoder_layer_7_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf248,
            arg119_1,
            arg120_1,
            buf250,
            buf251,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg119_1
        del arg120_1
        buf252 = buf221
        del buf221  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf250,
            arg121_1,
            buf251,
            buf252,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg121_1
        buf256 = buf209
        del buf209  # reuse
        # Source Nodes: [add_16, x_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf240,
            buf252,
            arg122_1,
            arg123_1,
            arg124_1,
            arg125_1,
            buf256,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg124_1
        del arg125_1
        buf257 = buf226
        del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf256, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg126_1, (768, 2304), (1, 768), 0),
            out=buf257,
        )
        del arg126_1
        buf258 = reinterpret_tensor(buf256, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf256  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf257, arg127_1, buf258, 151296, grid=grid(151296), stream=stream0
        )
        buf259 = buf229
        del buf229  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf257, arg127_1, buf259, 151296, grid=grid(151296), stream=stream0
        )
        buf260 = buf228
        del buf228  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf257, arg127_1, buf260, 151296, grid=grid(151296), stream=stream0
        )
        del arg127_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf261 = aten._scaled_dot_product_flash_attention.default(
            buf258, buf259, buf260, scale=0.125
        )
        del buf258
        buf262 = buf261[0]
        del buf261
        buf268 = reinterpret_tensor(buf246, (197, 768), (768, 1), 0)
        del buf246  # reuse
        buf269 = reinterpret_tensor(buf251, (197, 1), (1, 1), 0)
        del buf251  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf262, buf268, buf269, 197, 768, grid=grid(197), stream=stream0
        )
        buf270 = reinterpret_tensor(buf262, (197, 768), (768, 1), 0)
        del buf262  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf268,
            arg128_1,
            buf269,
            buf270,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg128_1
        buf271 = reinterpret_tensor(buf270, (1, 197, 768), (151296, 768, 1), 0)
        del buf270  # reuse
        buf277 = reinterpret_tensor(buf268, (1, 197, 768), (151296, 768, 1), 0)
        del buf268  # reuse
        buf278 = reinterpret_tensor(buf269, (1, 197, 1), (197, 1, 1), 0)
        del buf269  # reuse
        # Source Nodes: [add_16, l__self___encoder_layers_encoder_layer_8_mlp_0, x_38, x_39, y_16], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf271,
            arg129_1,
            arg130_1,
            buf240,
            buf252,
            arg122_1,
            arg123_1,
            arg131_1,
            arg132_1,
            buf277,
            buf278,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg122_1
        del arg123_1
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        buf279 = buf248
        del buf248  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf277,
            arg133_1,
            buf278,
            buf279,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg133_1
        buf281 = buf250
        del buf250  # reuse
        buf282 = buf278
        del buf278  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_mlp_1, l__self___encoder_layers_encoder_layer_8_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf279,
            arg134_1,
            arg135_1,
            buf281,
            buf282,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg134_1
        del arg135_1
        buf283 = buf252
        del buf252  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf281,
            arg136_1,
            buf282,
            buf283,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg136_1
        buf287 = buf240
        del buf240  # reuse
        # Source Nodes: [add_18, x_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf271,
            buf283,
            arg137_1,
            arg138_1,
            arg139_1,
            arg140_1,
            buf287,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg139_1
        del arg140_1
        buf288 = buf257
        del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf287, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg141_1, (768, 2304), (1, 768), 0),
            out=buf288,
        )
        del arg141_1
        buf289 = reinterpret_tensor(buf287, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf287  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf288, arg142_1, buf289, 151296, grid=grid(151296), stream=stream0
        )
        buf290 = buf260
        del buf260  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf288, arg142_1, buf290, 151296, grid=grid(151296), stream=stream0
        )
        buf291 = buf259
        del buf259  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf288, arg142_1, buf291, 151296, grid=grid(151296), stream=stream0
        )
        del arg142_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf292 = aten._scaled_dot_product_flash_attention.default(
            buf289, buf290, buf291, scale=0.125
        )
        del buf289
        buf293 = buf292[0]
        del buf292
        buf299 = reinterpret_tensor(buf277, (197, 768), (768, 1), 0)
        del buf277  # reuse
        buf300 = reinterpret_tensor(buf282, (197, 1), (1, 1), 0)
        del buf282  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf293, buf299, buf300, 197, 768, grid=grid(197), stream=stream0
        )
        buf301 = reinterpret_tensor(buf293, (197, 768), (768, 1), 0)
        del buf293  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf299,
            arg143_1,
            buf300,
            buf301,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg143_1
        buf302 = reinterpret_tensor(buf301, (1, 197, 768), (151296, 768, 1), 0)
        del buf301  # reuse
        buf308 = reinterpret_tensor(buf299, (1, 197, 768), (151296, 768, 1), 0)
        del buf299  # reuse
        buf309 = reinterpret_tensor(buf300, (1, 197, 1), (197, 1, 1), 0)
        del buf300  # reuse
        # Source Nodes: [add_18, l__self___encoder_layers_encoder_layer_9_mlp_0, x_42, x_43, y_18], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf302,
            arg144_1,
            arg145_1,
            buf271,
            buf283,
            arg137_1,
            arg138_1,
            arg146_1,
            arg147_1,
            buf308,
            buf309,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg137_1
        del arg138_1
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        buf310 = buf279
        del buf279  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf308,
            arg148_1,
            buf309,
            buf310,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg148_1
        buf312 = buf281
        del buf281  # reuse
        buf313 = buf309
        del buf309  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_mlp_1, l__self___encoder_layers_encoder_layer_9_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf310,
            arg149_1,
            arg150_1,
            buf312,
            buf313,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg149_1
        del arg150_1
        buf314 = buf283
        del buf283  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf312,
            arg151_1,
            buf313,
            buf314,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg151_1
        buf318 = buf271
        del buf271  # reuse
        # Source Nodes: [add_20, x_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf302,
            buf314,
            arg152_1,
            arg153_1,
            arg154_1,
            arg155_1,
            buf318,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg154_1
        del arg155_1
        buf319 = buf288
        del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf318, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg156_1, (768, 2304), (1, 768), 0),
            out=buf319,
        )
        del arg156_1
        buf320 = reinterpret_tensor(buf318, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf318  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf319, arg157_1, buf320, 151296, grid=grid(151296), stream=stream0
        )
        buf321 = buf291
        del buf291  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf319, arg157_1, buf321, 151296, grid=grid(151296), stream=stream0
        )
        buf322 = buf290
        del buf290  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf319, arg157_1, buf322, 151296, grid=grid(151296), stream=stream0
        )
        del arg157_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf323 = aten._scaled_dot_product_flash_attention.default(
            buf320, buf321, buf322, scale=0.125
        )
        del buf320
        buf324 = buf323[0]
        del buf323
        buf330 = reinterpret_tensor(buf308, (197, 768), (768, 1), 0)
        del buf308  # reuse
        buf331 = reinterpret_tensor(buf313, (197, 1), (1, 1), 0)
        del buf313  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf324, buf330, buf331, 197, 768, grid=grid(197), stream=stream0
        )
        buf332 = reinterpret_tensor(buf324, (197, 768), (768, 1), 0)
        del buf324  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf330,
            arg158_1,
            buf331,
            buf332,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg158_1
        buf333 = reinterpret_tensor(buf332, (1, 197, 768), (151296, 768, 1), 0)
        del buf332  # reuse
        buf339 = reinterpret_tensor(buf330, (1, 197, 768), (151296, 768, 1), 0)
        del buf330  # reuse
        buf340 = reinterpret_tensor(buf331, (1, 197, 1), (197, 1, 1), 0)
        del buf331  # reuse
        # Source Nodes: [add_20, l__self___encoder_layers_encoder_layer_10_mlp_0, x_46, x_47, y_20], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf333,
            arg159_1,
            arg160_1,
            buf302,
            buf314,
            arg152_1,
            arg153_1,
            arg161_1,
            arg162_1,
            buf339,
            buf340,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg152_1
        del arg153_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        buf341 = buf310
        del buf310  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf339,
            arg163_1,
            buf340,
            buf341,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg163_1
        buf343 = buf312
        del buf312  # reuse
        buf344 = buf340
        del buf340  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_mlp_1, l__self___encoder_layers_encoder_layer_10_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf341,
            arg164_1,
            arg165_1,
            buf343,
            buf344,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg164_1
        del arg165_1
        buf345 = buf314
        del buf314  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf343,
            arg166_1,
            buf344,
            buf345,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg166_1
        buf349 = buf302
        del buf302  # reuse
        # Source Nodes: [add_22, x_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf333,
            buf345,
            arg167_1,
            arg168_1,
            arg169_1,
            arg170_1,
            buf349,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg169_1
        del arg170_1
        buf350 = buf319
        del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf349, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg171_1, (768, 2304), (1, 768), 0),
            out=buf350,
        )
        del arg171_1
        buf351 = reinterpret_tensor(buf349, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf349  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf350, arg172_1, buf351, 151296, grid=grid(151296), stream=stream0
        )
        buf352 = buf322
        del buf322  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf350, arg172_1, buf352, 151296, grid=grid(151296), stream=stream0
        )
        buf353 = buf321
        del buf321  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf350, arg172_1, buf353, 151296, grid=grid(151296), stream=stream0
        )
        del arg172_1
        del buf350
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf354 = aten._scaled_dot_product_flash_attention.default(
            buf351, buf352, buf353, scale=0.125
        )
        del buf351
        del buf352
        del buf353
        buf355 = buf354[0]
        del buf354
        buf361 = reinterpret_tensor(buf339, (197, 768), (768, 1), 0)
        del buf339  # reuse
        buf362 = reinterpret_tensor(buf344, (197, 1), (1, 1), 0)
        del buf344  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_6.run(
            buf355, buf361, buf362, 197, 768, grid=grid(197), stream=stream0
        )
        buf363 = reinterpret_tensor(buf355, (197, 768), (768, 1), 0)
        del buf355  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_clamp_div_mul_round_view_7.run(
            buf361,
            arg173_1,
            buf362,
            buf363,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta0),
            stream=stream0,
        )
        del arg173_1
        buf364 = reinterpret_tensor(buf363, (1, 197, 768), (151296, 768, 1), 0)
        del buf363  # reuse
        buf370 = reinterpret_tensor(buf361, (1, 197, 768), (151296, 768, 1), 0)
        del buf361  # reuse
        buf371 = reinterpret_tensor(buf362, (1, 197, 1), (197, 1, 1), 0)
        del buf362  # reuse
        # Source Nodes: [add_22, l__self___encoder_layers_encoder_layer_11_mlp_0, x_50, x_51, y_22], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.amax, aten.clamp, aten.clone, aten.div, aten.native_layer_norm, aten.round]
        triton_per_fused__to_copy_abs_add_amax_clamp_clone_div_native_layer_norm_round_13.run(
            buf364,
            arg174_1,
            arg175_1,
            buf333,
            buf345,
            arg167_1,
            arg168_1,
            arg176_1,
            arg177_1,
            buf370,
            buf371,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg167_1
        del arg168_1
        del arg174_1
        del arg175_1
        del arg176_1
        del arg177_1
        buf372 = buf341
        del buf341  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_mlp_0], Original ATen: [aten.mul]
        triton_tem_fused_mul_9.run(
            buf370,
            arg178_1,
            buf371,
            buf372,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 3072, meta1),
            stream=stream0,
        )
        del arg178_1
        del buf370
        buf374 = buf343
        del buf343  # reuse
        buf375 = buf371
        del buf371  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_mlp_1, l__self___encoder_layers_encoder_layer_11_mlp_3], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.gelu, aten.round]
        triton_red_fused__to_copy_abs_amax_clamp_div_gelu_round_10.run(
            buf372,
            arg179_1,
            arg180_1,
            buf374,
            buf375,
            197,
            3072,
            grid=grid(197),
            stream=stream0,
        )
        del arg179_1
        del arg180_1
        del buf372
        buf376 = buf345
        del buf345  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_mlp_3], Original ATen: [aten.mul]
        triton_tem_fused_mul_11.run(
            buf374,
            arg181_1,
            buf375,
            buf376,
            grid=torch._inductor.kernel.mm_common.mm_grid(197, 768, meta2),
            stream=stream0,
        )
        del arg181_1
        del buf374
        del buf375
        buf380 = buf333
        del buf333  # reuse
        # Source Nodes: [add_24, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(
            buf364,
            buf376,
            arg182_1,
            arg183_1,
            arg184_1,
            arg185_1,
            buf380,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf364
        del buf376
        buf382 = empty_strided_cuda((1, 768), (768, 1), torch.int8)
        buf383 = empty_strided_cuda((1, 1), (1, 1), torch.bfloat16)
        # Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.abs, aten.amax, aten.clamp, aten.div, aten.round]
        triton_per_fused__to_copy_abs_amax_clamp_div_round_14.run(
            buf380, buf382, buf383, 1, 768, grid=grid(1), stream=stream0
        )
        del buf380
        buf384 = empty_strided_cuda((1, 1000), (1000, 1), torch.bfloat16)
        # Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_poi_fused__to_copy_clamp_div_mul_round_view_15.run(
            buf383, buf384, 1000, grid=grid(1000), stream=stream0
        )
        del buf383
        buf386 = empty_strided_cuda((1, 1000), (1000, 1), torch.bfloat16)
        # Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.div, aten.mul, aten.round, aten.view]
        triton_tem_fused__to_copy_add_clamp_div_mul_round_view_16.run(
            buf382,
            arg186_1,
            buf384,
            arg187_1,
            arg188_1,
            buf386,
            grid=torch._inductor.kernel.mm_common.mm_grid(1, 1000, meta3),
            stream=stream0,
        )
        del arg186_1
        del arg187_1
        del arg188_1
        del buf382
        del buf384
    return (buf386,)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = rand_strided(
        (1, 1, 768), (768, 768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg1_1 = rand_strided(
        (1, 197, 768), (151296, 768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg2_1 = rand_strided(
        (768, 3, 16, 16), (768, 256, 16, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg3_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg4_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg5_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg6_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg7_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg8_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg9_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg10_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg11_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg12_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg13_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg14_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg15_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg16_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg17_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg18_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg19_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg20_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg21_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg22_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg23_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg24_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg25_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg26_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg27_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg28_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg29_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg30_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg31_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg32_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg33_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg34_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg35_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg36_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg37_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg38_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg39_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg40_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg41_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg42_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg43_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg44_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg45_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg46_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg47_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg48_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg49_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg50_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg51_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg52_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg53_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg54_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg55_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg56_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg57_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg58_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg59_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg60_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg61_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg62_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg63_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg64_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg65_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg66_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg67_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg68_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg69_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg70_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg71_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg72_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg73_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg74_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg75_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg76_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg77_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg78_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg79_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg80_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg81_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg82_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg83_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg84_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg85_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg86_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg87_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg88_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg89_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg90_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg91_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg92_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg93_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg94_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg95_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg96_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg97_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg98_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg99_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg100_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg101_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg102_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg103_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg104_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg105_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg106_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg107_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg108_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg109_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg110_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg111_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg112_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg113_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg114_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg115_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg116_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg117_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg118_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg119_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg120_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg121_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg122_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg123_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg124_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg125_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg126_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg127_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg128_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg129_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg130_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg131_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg132_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg133_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg134_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg135_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg136_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg137_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg138_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg139_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg140_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg141_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg142_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg143_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg144_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg145_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg146_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg147_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg148_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg149_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg150_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg151_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg152_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg153_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg154_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg155_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg156_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg157_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg158_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg159_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg160_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg161_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg162_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg163_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg164_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg165_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg166_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg167_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg168_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg169_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg170_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg171_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg172_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg173_1 = rand_strided((768, 768), (1, 768), device="cuda:0", dtype=torch.int8)
    arg174_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg175_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg176_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg177_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg178_1 = rand_strided((768, 3072), (1, 768), device="cuda:0", dtype=torch.int8)
    arg179_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg180_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg181_1 = rand_strided((3072, 768), (1, 3072), device="cuda:0", dtype=torch.int8)
    arg182_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg183_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg184_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg185_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg186_1 = rand_strided((768, 1000), (1, 768), device="cuda:0", dtype=torch.int8)
    arg187_1 = rand_strided((1000,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg188_1 = rand_strided((1000,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg189_1 = rand_strided(
        (1, 3, 224, 224), (150528, 50176, 224, 1), device="cuda:0", dtype=torch.bfloat16
    )
    fn = lambda: call(
        [
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            arg8_1,
            arg9_1,
            arg10_1,
            arg11_1,
            arg12_1,
            arg13_1,
            arg14_1,
            arg15_1,
            arg16_1,
            arg17_1,
            arg18_1,
            arg19_1,
            arg20_1,
            arg21_1,
            arg22_1,
            arg23_1,
            arg24_1,
            arg25_1,
            arg26_1,
            arg27_1,
            arg28_1,
            arg29_1,
            arg30_1,
            arg31_1,
            arg32_1,
            arg33_1,
            arg34_1,
            arg35_1,
            arg36_1,
            arg37_1,
            arg38_1,
            arg39_1,
            arg40_1,
            arg41_1,
            arg42_1,
            arg43_1,
            arg44_1,
            arg45_1,
            arg46_1,
            arg47_1,
            arg48_1,
            arg49_1,
            arg50_1,
            arg51_1,
            arg52_1,
            arg53_1,
            arg54_1,
            arg55_1,
            arg56_1,
            arg57_1,
            arg58_1,
            arg59_1,
            arg60_1,
            arg61_1,
            arg62_1,
            arg63_1,
            arg64_1,
            arg65_1,
            arg66_1,
            arg67_1,
            arg68_1,
            arg69_1,
            arg70_1,
            arg71_1,
            arg72_1,
            arg73_1,
            arg74_1,
            arg75_1,
            arg76_1,
            arg77_1,
            arg78_1,
            arg79_1,
            arg80_1,
            arg81_1,
            arg82_1,
            arg83_1,
            arg84_1,
            arg85_1,
            arg86_1,
            arg87_1,
            arg88_1,
            arg89_1,
            arg90_1,
            arg91_1,
            arg92_1,
            arg93_1,
            arg94_1,
            arg95_1,
            arg96_1,
            arg97_1,
            arg98_1,
            arg99_1,
            arg100_1,
            arg101_1,
            arg102_1,
            arg103_1,
            arg104_1,
            arg105_1,
            arg106_1,
            arg107_1,
            arg108_1,
            arg109_1,
            arg110_1,
            arg111_1,
            arg112_1,
            arg113_1,
            arg114_1,
            arg115_1,
            arg116_1,
            arg117_1,
            arg118_1,
            arg119_1,
            arg120_1,
            arg121_1,
            arg122_1,
            arg123_1,
            arg124_1,
            arg125_1,
            arg126_1,
            arg127_1,
            arg128_1,
            arg129_1,
            arg130_1,
            arg131_1,
            arg132_1,
            arg133_1,
            arg134_1,
            arg135_1,
            arg136_1,
            arg137_1,
            arg138_1,
            arg139_1,
            arg140_1,
            arg141_1,
            arg142_1,
            arg143_1,
            arg144_1,
            arg145_1,
            arg146_1,
            arg147_1,
            arg148_1,
            arg149_1,
            arg150_1,
            arg151_1,
            arg152_1,
            arg153_1,
            arg154_1,
            arg155_1,
            arg156_1,
            arg157_1,
            arg158_1,
            arg159_1,
            arg160_1,
            arg161_1,
            arg162_1,
            arg163_1,
            arg164_1,
            arg165_1,
            arg166_1,
            arg167_1,
            arg168_1,
            arg169_1,
            arg170_1,
            arg171_1,
            arg172_1,
            arg173_1,
            arg174_1,
            arg175_1,
            arg176_1,
            arg177_1,
            arg178_1,
            arg179_1,
            arg180_1,
            arg181_1,
            arg182_1,
            arg183_1,
            arg184_1,
            arg185_1,
            arg186_1,
            arg187_1,
            arg188_1,
            arg189_1,
        ]
    )
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
