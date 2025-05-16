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


# kernel path: /tmp/torchinductor_cpuhrsch/un/cunk3wqofix3bpzsfdo7acyque4mldycmrexd6slyevong5r5x4s.py
# Source Nodes: [cat_1, input_1, x_6, x_7, y], Original ATen: [aten.add, aten.cat, aten.clone, aten.native_layer_norm]
# cat_1 => cat
# input_1 => add
# x_6 => clone_2
# x_7 => add_3
# y => add_4, add_5, convert_element_type_8, convert_element_type_9, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_per_fused_add_cat_clone_native_layer_norm_6 = async_compile.triton(
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_clone_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 >= tmp6
    tmp12 = tl.full([1], 197, tl.int64)
    tmp13 = tmp3 < tmp12
    tmp14 = tl.load(in_ptr2 + ((196*r1) + (((-1) + x0) % 196)), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp11, tmp16, tmp17)
    tmp19 = tl.where(tmp7, tmp10, tmp18)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tl.full([1], 768, tl.int32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 / tmp32
    tmp34 = tmp24 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = tmp23 - tmp33
    tmp41 = 768.0
    tmp42 = tmp39 / tmp41
    tmp43 = 1e-06
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp40 * tmp45
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp53, rmask & xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ov/covg5yeigiq2fhtxx42dah2r4u7lksvcwdlsxivokx5mpjvxck3b.py
# Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_1], Original ATen: [aten.gelu]
# l__self___encoder_layers_encoder_layer_0_mlp_1 => add_6, convert_element_type_13, convert_element_type_14, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_7 = async_compile.triton(
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
    size_hints=[1048576],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp12, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/p6/cp6msnkgbybdxhqixazfcuyduc476ws6y24btzy6kvuc3tb3lfrp.py
# Source Nodes: [add_2, x_8], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_7
# x_8 => add_8, add_9, convert_element_type_18, convert_element_type_19, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_native_layer_norm_8 = async_compile.triton(
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp5 - tmp15
    tmp23 = 768.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/ul/culnd7xrtza242hyrp7hnarnsdk5o7ovwfjdy56jyxubon7uxhu2.py
# Source Nodes: [add_2, x_10, x_11, y_2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_7
# x_10 => clone_6
# x_11 => add_10
# y_2 => add_11, add_12, convert_element_type_26, convert_element_type_27, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
triton_per_fused_add_clone_native_layer_norm_9 = async_compile.triton(
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 768.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-06
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_cpuhrsch/3a/c3a3shxprppl7kcfrlppv4wfxya4gbxnk6txrb6ht4c5m3ehr4qg.py
# Source Nodes: [x_54], Original ATen: [aten.addmm]
# x_54 => addmm_48
triton_tem_fused_addmm_10 = async_compile.triton(
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
    num_warps=2,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_10', 'backend_hash': 'a33067c4979f4f58aaa689554e57cd7781667c4ec621f76e3aabedec9456e44b'},
)
@triton.jit
def triton_(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 1
    N = 1000
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 0
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
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), tmp1, mask)
""",
    device_str="cuda",
)
import torch._inductor.kernel.mm_common

meta0 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.float32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 16,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
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
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768,), (1,))
    assert_size_stride(arg10_1, (768,), (1,))
    assert_size_stride(arg11_1, (768,), (1,))
    assert_size_stride(arg12_1, (3072, 768), (768, 1))
    assert_size_stride(arg13_1, (3072,), (1,))
    assert_size_stride(arg14_1, (768, 3072), (3072, 1))
    assert_size_stride(arg15_1, (768,), (1,))
    assert_size_stride(arg16_1, (768,), (1,))
    assert_size_stride(arg17_1, (768,), (1,))
    assert_size_stride(arg18_1, (2304, 768), (768, 1))
    assert_size_stride(arg19_1, (2304,), (1,))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768,), (1,))
    assert_size_stride(arg22_1, (768,), (1,))
    assert_size_stride(arg23_1, (768,), (1,))
    assert_size_stride(arg24_1, (3072, 768), (768, 1))
    assert_size_stride(arg25_1, (3072,), (1,))
    assert_size_stride(arg26_1, (768, 3072), (3072, 1))
    assert_size_stride(arg27_1, (768,), (1,))
    assert_size_stride(arg28_1, (768,), (1,))
    assert_size_stride(arg29_1, (768,), (1,))
    assert_size_stride(arg30_1, (2304, 768), (768, 1))
    assert_size_stride(arg31_1, (2304,), (1,))
    assert_size_stride(arg32_1, (768, 768), (768, 1))
    assert_size_stride(arg33_1, (768,), (1,))
    assert_size_stride(arg34_1, (768,), (1,))
    assert_size_stride(arg35_1, (768,), (1,))
    assert_size_stride(arg36_1, (3072, 768), (768, 1))
    assert_size_stride(arg37_1, (3072,), (1,))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768,), (1,))
    assert_size_stride(arg40_1, (768,), (1,))
    assert_size_stride(arg41_1, (768,), (1,))
    assert_size_stride(arg42_1, (2304, 768), (768, 1))
    assert_size_stride(arg43_1, (2304,), (1,))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768,), (1,))
    assert_size_stride(arg46_1, (768,), (1,))
    assert_size_stride(arg47_1, (768,), (1,))
    assert_size_stride(arg48_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (3072,), (1,))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768,), (1,))
    assert_size_stride(arg52_1, (768,), (1,))
    assert_size_stride(arg53_1, (768,), (1,))
    assert_size_stride(arg54_1, (2304, 768), (768, 1))
    assert_size_stride(arg55_1, (2304,), (1,))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768,), (1,))
    assert_size_stride(arg58_1, (768,), (1,))
    assert_size_stride(arg59_1, (768,), (1,))
    assert_size_stride(arg60_1, (3072, 768), (768, 1))
    assert_size_stride(arg61_1, (3072,), (1,))
    assert_size_stride(arg62_1, (768, 3072), (3072, 1))
    assert_size_stride(arg63_1, (768,), (1,))
    assert_size_stride(arg64_1, (768,), (1,))
    assert_size_stride(arg65_1, (768,), (1,))
    assert_size_stride(arg66_1, (2304, 768), (768, 1))
    assert_size_stride(arg67_1, (2304,), (1,))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768,), (1,))
    assert_size_stride(arg70_1, (768,), (1,))
    assert_size_stride(arg71_1, (768,), (1,))
    assert_size_stride(arg72_1, (3072, 768), (768, 1))
    assert_size_stride(arg73_1, (3072,), (1,))
    assert_size_stride(arg74_1, (768, 3072), (3072, 1))
    assert_size_stride(arg75_1, (768,), (1,))
    assert_size_stride(arg76_1, (768,), (1,))
    assert_size_stride(arg77_1, (768,), (1,))
    assert_size_stride(arg78_1, (2304, 768), (768, 1))
    assert_size_stride(arg79_1, (2304,), (1,))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768,), (1,))
    assert_size_stride(arg82_1, (768,), (1,))
    assert_size_stride(arg83_1, (768,), (1,))
    assert_size_stride(arg84_1, (3072, 768), (768, 1))
    assert_size_stride(arg85_1, (3072,), (1,))
    assert_size_stride(arg86_1, (768, 3072), (3072, 1))
    assert_size_stride(arg87_1, (768,), (1,))
    assert_size_stride(arg88_1, (768,), (1,))
    assert_size_stride(arg89_1, (768,), (1,))
    assert_size_stride(arg90_1, (2304, 768), (768, 1))
    assert_size_stride(arg91_1, (2304,), (1,))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768,), (1,))
    assert_size_stride(arg94_1, (768,), (1,))
    assert_size_stride(arg95_1, (768,), (1,))
    assert_size_stride(arg96_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (3072,), (1,))
    assert_size_stride(arg98_1, (768, 3072), (3072, 1))
    assert_size_stride(arg99_1, (768,), (1,))
    assert_size_stride(arg100_1, (768,), (1,))
    assert_size_stride(arg101_1, (768,), (1,))
    assert_size_stride(arg102_1, (2304, 768), (768, 1))
    assert_size_stride(arg103_1, (2304,), (1,))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768,), (1,))
    assert_size_stride(arg106_1, (768,), (1,))
    assert_size_stride(arg107_1, (768,), (1,))
    assert_size_stride(arg108_1, (3072, 768), (768, 1))
    assert_size_stride(arg109_1, (3072,), (1,))
    assert_size_stride(arg110_1, (768, 3072), (3072, 1))
    assert_size_stride(arg111_1, (768,), (1,))
    assert_size_stride(arg112_1, (768,), (1,))
    assert_size_stride(arg113_1, (768,), (1,))
    assert_size_stride(arg114_1, (2304, 768), (768, 1))
    assert_size_stride(arg115_1, (2304,), (1,))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768,), (1,))
    assert_size_stride(arg118_1, (768,), (1,))
    assert_size_stride(arg119_1, (768,), (1,))
    assert_size_stride(arg120_1, (3072, 768), (768, 1))
    assert_size_stride(arg121_1, (3072,), (1,))
    assert_size_stride(arg122_1, (768, 3072), (3072, 1))
    assert_size_stride(arg123_1, (768,), (1,))
    assert_size_stride(arg124_1, (768,), (1,))
    assert_size_stride(arg125_1, (768,), (1,))
    assert_size_stride(arg126_1, (2304, 768), (768, 1))
    assert_size_stride(arg127_1, (2304,), (1,))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768,), (1,))
    assert_size_stride(arg130_1, (768,), (1,))
    assert_size_stride(arg131_1, (768,), (1,))
    assert_size_stride(arg132_1, (3072, 768), (768, 1))
    assert_size_stride(arg133_1, (3072,), (1,))
    assert_size_stride(arg134_1, (768, 3072), (3072, 1))
    assert_size_stride(arg135_1, (768,), (1,))
    assert_size_stride(arg136_1, (768,), (1,))
    assert_size_stride(arg137_1, (768,), (1,))
    assert_size_stride(arg138_1, (2304, 768), (768, 1))
    assert_size_stride(arg139_1, (2304,), (1,))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768,), (1,))
    assert_size_stride(arg142_1, (768,), (1,))
    assert_size_stride(arg143_1, (768,), (1,))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072,), (1,))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768,), (1,))
    assert_size_stride(arg148_1, (768,), (1,))
    assert_size_stride(arg149_1, (768,), (1,))
    assert_size_stride(arg150_1, (1000, 768), (768, 1))
    assert_size_stride(arg151_1, (1000,), (1,))
    assert_size_stride(arg152_1, (1, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(
            arg152_1,
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
        del arg152_1
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
        buf19 = reinterpret_tensor(buf12, (197, 768), (768, 1), 0)
        del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf14, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0),
            out=buf19,
        )
        del arg8_1
        buf20 = reinterpret_tensor(buf19, (1, 197, 768), (151296, 768, 1), 0)
        del buf19  # reuse
        buf24 = reinterpret_tensor(buf14, (1, 197, 768), (151296, 768, 1), 0)
        del buf14  # reuse
        # Source Nodes: [cat_1, input_1, x_6, x_7, y], Original ATen: [aten.add, aten.cat, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_cat_clone_native_layer_norm_6.run(
            buf20,
            arg9_1,
            arg0_1,
            buf0,
            arg3_1,
            arg1_1,
            arg10_1,
            arg11_1,
            buf24,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg0_1
        del arg10_1
        del arg11_1
        del arg1_1
        del arg3_1
        del arg9_1
        del buf0
        buf25 = empty_strided_cuda((197, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf24, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg12_1, (768, 3072), (1, 768), 0),
            out=buf25,
        )
        del arg12_1
        buf26 = reinterpret_tensor(buf25, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf25  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_0_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf26, arg13_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg13_1
        buf27 = reinterpret_tensor(buf24, (197, 768), (768, 1), 0)
        del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf26, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg14_1, (3072, 768), (1, 3072), 0),
            out=buf27,
        )
        del arg14_1
        buf31 = reinterpret_tensor(buf11, (1, 197, 768), (151296, 768, 1), 0)
        del buf11  # reuse
        # Source Nodes: [add_2, x_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf20,
            buf27,
            arg15_1,
            arg16_1,
            arg17_1,
            buf31,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg16_1
        del arg17_1
        buf32 = buf9
        del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf31, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg18_1, (768, 2304), (1, 768), 0),
            out=buf32,
        )
        del arg18_1
        buf33 = reinterpret_tensor(buf31, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf31  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf32, arg19_1, buf33, 151296, grid=grid(151296), stream=stream0
        )
        buf34 = buf10
        del buf10  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf32, arg19_1, buf34, 151296, grid=grid(151296), stream=stream0
        )
        buf35 = empty_strided_cuda(
            (1, 12, 197, 64), (151296, 64, 768, 1), torch.bfloat16
        )
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf32, arg19_1, buf35, 151296, grid=grid(151296), stream=stream0
        )
        del arg19_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf36 = aten._scaled_dot_product_flash_attention.default(
            buf33, buf34, buf35, scale=0.125
        )
        del buf33
        buf37 = buf36[0]
        del buf36
        buf42 = reinterpret_tensor(buf35, (197, 768), (768, 1), 0)
        del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf37, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0),
            out=buf42,
        )
        del arg20_1
        buf43 = reinterpret_tensor(buf42, (1, 197, 768), (151296, 768, 1), 0)
        del buf42  # reuse
        buf47 = reinterpret_tensor(buf37, (1, 197, 768), (151296, 768, 1), 0)
        del buf37  # reuse
        # Source Nodes: [add_2, x_10, x_11, y_2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf43,
            arg21_1,
            buf20,
            buf27,
            arg15_1,
            arg22_1,
            arg23_1,
            buf47,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg15_1
        del arg21_1
        del arg22_1
        del arg23_1
        buf48 = reinterpret_tensor(buf26, (197, 3072), (3072, 1), 0)
        del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf47, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg24_1, (768, 3072), (1, 768), 0),
            out=buf48,
        )
        del arg24_1
        buf49 = reinterpret_tensor(buf48, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf48  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_1_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf49, arg25_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg25_1
        buf50 = reinterpret_tensor(buf47, (197, 768), (768, 1), 0)
        del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf49, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg26_1, (3072, 768), (1, 3072), 0),
            out=buf50,
        )
        del arg26_1
        buf54 = reinterpret_tensor(buf27, (1, 197, 768), (151296, 768, 1), 0)
        del buf27  # reuse
        # Source Nodes: [add_4, x_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf43,
            buf50,
            arg27_1,
            arg28_1,
            arg29_1,
            buf54,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg28_1
        del arg29_1
        buf55 = buf32
        del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf54, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg30_1, (768, 2304), (1, 768), 0),
            out=buf55,
        )
        del arg30_1
        buf56 = reinterpret_tensor(buf54, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf54  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf55, arg31_1, buf56, 151296, grid=grid(151296), stream=stream0
        )
        buf57 = reinterpret_tensor(buf20, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf20  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf55, arg31_1, buf57, 151296, grid=grid(151296), stream=stream0
        )
        buf58 = buf34
        del buf34  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf55, arg31_1, buf58, 151296, grid=grid(151296), stream=stream0
        )
        del arg31_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf59 = aten._scaled_dot_product_flash_attention.default(
            buf56, buf57, buf58, scale=0.125
        )
        del buf56
        buf60 = buf59[0]
        del buf59
        buf65 = reinterpret_tensor(buf58, (197, 768), (768, 1), 0)
        del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf60, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg32_1, (768, 768), (1, 768), 0),
            out=buf65,
        )
        del arg32_1
        buf66 = reinterpret_tensor(buf65, (1, 197, 768), (151296, 768, 1), 0)
        del buf65  # reuse
        buf70 = reinterpret_tensor(buf60, (1, 197, 768), (151296, 768, 1), 0)
        del buf60  # reuse
        # Source Nodes: [add_4, x_14, x_15, y_4], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf66,
            arg33_1,
            buf43,
            buf50,
            arg27_1,
            arg34_1,
            arg35_1,
            buf70,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg27_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf71 = reinterpret_tensor(buf49, (197, 3072), (3072, 1), 0)
        del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf70, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg36_1, (768, 3072), (1, 768), 0),
            out=buf71,
        )
        del arg36_1
        buf72 = reinterpret_tensor(buf71, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf71  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_2_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf72, arg37_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg37_1
        buf73 = reinterpret_tensor(buf70, (197, 768), (768, 1), 0)
        del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf72, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg38_1, (3072, 768), (1, 3072), 0),
            out=buf73,
        )
        del arg38_1
        buf77 = reinterpret_tensor(buf50, (1, 197, 768), (151296, 768, 1), 0)
        del buf50  # reuse
        # Source Nodes: [add_6, x_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf66,
            buf73,
            arg39_1,
            arg40_1,
            arg41_1,
            buf77,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg40_1
        del arg41_1
        buf78 = buf55
        del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf77, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg42_1, (768, 2304), (1, 768), 0),
            out=buf78,
        )
        del arg42_1
        buf79 = reinterpret_tensor(buf77, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf77  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf78, arg43_1, buf79, 151296, grid=grid(151296), stream=stream0
        )
        buf80 = reinterpret_tensor(buf43, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf43  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf78, arg43_1, buf80, 151296, grid=grid(151296), stream=stream0
        )
        buf81 = buf57
        del buf57  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf78, arg43_1, buf81, 151296, grid=grid(151296), stream=stream0
        )
        del arg43_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf82 = aten._scaled_dot_product_flash_attention.default(
            buf79, buf80, buf81, scale=0.125
        )
        del buf79
        buf83 = buf82[0]
        del buf82
        buf88 = reinterpret_tensor(buf81, (197, 768), (768, 1), 0)
        del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf83, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0),
            out=buf88,
        )
        del arg44_1
        buf89 = reinterpret_tensor(buf88, (1, 197, 768), (151296, 768, 1), 0)
        del buf88  # reuse
        buf93 = reinterpret_tensor(buf83, (1, 197, 768), (151296, 768, 1), 0)
        del buf83  # reuse
        # Source Nodes: [add_6, x_18, x_19, y_6], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf89,
            arg45_1,
            buf66,
            buf73,
            arg39_1,
            arg46_1,
            arg47_1,
            buf93,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg39_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf94 = reinterpret_tensor(buf72, (197, 3072), (3072, 1), 0)
        del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf93, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0),
            out=buf94,
        )
        del arg48_1
        buf95 = reinterpret_tensor(buf94, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf94  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_3_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf95, arg49_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg49_1
        buf96 = reinterpret_tensor(buf93, (197, 768), (768, 1), 0)
        del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf95, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0),
            out=buf96,
        )
        del arg50_1
        buf100 = reinterpret_tensor(buf73, (1, 197, 768), (151296, 768, 1), 0)
        del buf73  # reuse
        # Source Nodes: [add_8, x_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf89,
            buf96,
            arg51_1,
            arg52_1,
            arg53_1,
            buf100,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg52_1
        del arg53_1
        buf101 = buf78
        del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf100, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg54_1, (768, 2304), (1, 768), 0),
            out=buf101,
        )
        del arg54_1
        buf102 = reinterpret_tensor(buf100, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf100  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf101, arg55_1, buf102, 151296, grid=grid(151296), stream=stream0
        )
        buf103 = reinterpret_tensor(buf66, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf66  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf101, arg55_1, buf103, 151296, grid=grid(151296), stream=stream0
        )
        buf104 = buf80
        del buf80  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf101, arg55_1, buf104, 151296, grid=grid(151296), stream=stream0
        )
        del arg55_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf105 = aten._scaled_dot_product_flash_attention.default(
            buf102, buf103, buf104, scale=0.125
        )
        del buf102
        buf106 = buf105[0]
        del buf105
        buf111 = reinterpret_tensor(buf104, (197, 768), (768, 1), 0)
        del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf106, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0),
            out=buf111,
        )
        del arg56_1
        buf112 = reinterpret_tensor(buf111, (1, 197, 768), (151296, 768, 1), 0)
        del buf111  # reuse
        buf116 = reinterpret_tensor(buf106, (1, 197, 768), (151296, 768, 1), 0)
        del buf106  # reuse
        # Source Nodes: [add_8, x_22, x_23, y_8], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf112,
            arg57_1,
            buf89,
            buf96,
            arg51_1,
            arg58_1,
            arg59_1,
            buf116,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg51_1
        del arg57_1
        del arg58_1
        del arg59_1
        buf117 = reinterpret_tensor(buf95, (197, 3072), (3072, 1), 0)
        del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf116, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg60_1, (768, 3072), (1, 768), 0),
            out=buf117,
        )
        del arg60_1
        buf118 = reinterpret_tensor(buf117, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf117  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_4_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf118, arg61_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg61_1
        buf119 = reinterpret_tensor(buf116, (197, 768), (768, 1), 0)
        del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf118, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg62_1, (3072, 768), (1, 3072), 0),
            out=buf119,
        )
        del arg62_1
        buf123 = reinterpret_tensor(buf96, (1, 197, 768), (151296, 768, 1), 0)
        del buf96  # reuse
        # Source Nodes: [add_10, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf112,
            buf119,
            arg63_1,
            arg64_1,
            arg65_1,
            buf123,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg64_1
        del arg65_1
        buf124 = buf101
        del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf123, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg66_1, (768, 2304), (1, 768), 0),
            out=buf124,
        )
        del arg66_1
        buf125 = reinterpret_tensor(buf123, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf123  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf124, arg67_1, buf125, 151296, grid=grid(151296), stream=stream0
        )
        buf126 = reinterpret_tensor(buf89, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf89  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf124, arg67_1, buf126, 151296, grid=grid(151296), stream=stream0
        )
        buf127 = buf103
        del buf103  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf124, arg67_1, buf127, 151296, grid=grid(151296), stream=stream0
        )
        del arg67_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf128 = aten._scaled_dot_product_flash_attention.default(
            buf125, buf126, buf127, scale=0.125
        )
        del buf125
        buf129 = buf128[0]
        del buf128
        buf134 = reinterpret_tensor(buf127, (197, 768), (768, 1), 0)
        del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf129, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0),
            out=buf134,
        )
        del arg68_1
        buf135 = reinterpret_tensor(buf134, (1, 197, 768), (151296, 768, 1), 0)
        del buf134  # reuse
        buf139 = reinterpret_tensor(buf129, (1, 197, 768), (151296, 768, 1), 0)
        del buf129  # reuse
        # Source Nodes: [add_10, x_26, x_27, y_10], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf135,
            arg69_1,
            buf112,
            buf119,
            arg63_1,
            arg70_1,
            arg71_1,
            buf139,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg63_1
        del arg69_1
        del arg70_1
        del arg71_1
        buf140 = reinterpret_tensor(buf118, (197, 3072), (3072, 1), 0)
        del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf139, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg72_1, (768, 3072), (1, 768), 0),
            out=buf140,
        )
        del arg72_1
        buf141 = reinterpret_tensor(buf140, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf140  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_5_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf141, arg73_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg73_1
        buf142 = reinterpret_tensor(buf139, (197, 768), (768, 1), 0)
        del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf141, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg74_1, (3072, 768), (1, 3072), 0),
            out=buf142,
        )
        del arg74_1
        buf146 = reinterpret_tensor(buf119, (1, 197, 768), (151296, 768, 1), 0)
        del buf119  # reuse
        # Source Nodes: [add_12, x_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf135,
            buf142,
            arg75_1,
            arg76_1,
            arg77_1,
            buf146,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg76_1
        del arg77_1
        buf147 = buf124
        del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf146, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg78_1, (768, 2304), (1, 768), 0),
            out=buf147,
        )
        del arg78_1
        buf148 = reinterpret_tensor(buf146, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf146  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf147, arg79_1, buf148, 151296, grid=grid(151296), stream=stream0
        )
        buf149 = reinterpret_tensor(buf112, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf112  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf147, arg79_1, buf149, 151296, grid=grid(151296), stream=stream0
        )
        buf150 = buf126
        del buf126  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf147, arg79_1, buf150, 151296, grid=grid(151296), stream=stream0
        )
        del arg79_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf151 = aten._scaled_dot_product_flash_attention.default(
            buf148, buf149, buf150, scale=0.125
        )
        del buf148
        buf152 = buf151[0]
        del buf151
        buf157 = reinterpret_tensor(buf150, (197, 768), (768, 1), 0)
        del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf152, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0),
            out=buf157,
        )
        del arg80_1
        buf158 = reinterpret_tensor(buf157, (1, 197, 768), (151296, 768, 1), 0)
        del buf157  # reuse
        buf162 = reinterpret_tensor(buf152, (1, 197, 768), (151296, 768, 1), 0)
        del buf152  # reuse
        # Source Nodes: [add_12, x_30, x_31, y_12], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf158,
            arg81_1,
            buf135,
            buf142,
            arg75_1,
            arg82_1,
            arg83_1,
            buf162,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg75_1
        del arg81_1
        del arg82_1
        del arg83_1
        buf163 = reinterpret_tensor(buf141, (197, 3072), (3072, 1), 0)
        del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf162, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg84_1, (768, 3072), (1, 768), 0),
            out=buf163,
        )
        del arg84_1
        buf164 = reinterpret_tensor(buf163, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf163  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_6_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf164, arg85_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg85_1
        buf165 = reinterpret_tensor(buf162, (197, 768), (768, 1), 0)
        del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf164, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg86_1, (3072, 768), (1, 3072), 0),
            out=buf165,
        )
        del arg86_1
        buf169 = reinterpret_tensor(buf142, (1, 197, 768), (151296, 768, 1), 0)
        del buf142  # reuse
        # Source Nodes: [add_14, x_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf158,
            buf165,
            arg87_1,
            arg88_1,
            arg89_1,
            buf169,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg88_1
        del arg89_1
        buf170 = buf147
        del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf169, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg90_1, (768, 2304), (1, 768), 0),
            out=buf170,
        )
        del arg90_1
        buf171 = reinterpret_tensor(buf169, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf169  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf170, arg91_1, buf171, 151296, grid=grid(151296), stream=stream0
        )
        buf172 = reinterpret_tensor(buf135, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf135  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf170, arg91_1, buf172, 151296, grid=grid(151296), stream=stream0
        )
        buf173 = buf149
        del buf149  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf170, arg91_1, buf173, 151296, grid=grid(151296), stream=stream0
        )
        del arg91_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf174 = aten._scaled_dot_product_flash_attention.default(
            buf171, buf172, buf173, scale=0.125
        )
        del buf171
        buf175 = buf174[0]
        del buf174
        buf180 = reinterpret_tensor(buf173, (197, 768), (768, 1), 0)
        del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf175, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0),
            out=buf180,
        )
        del arg92_1
        buf181 = reinterpret_tensor(buf180, (1, 197, 768), (151296, 768, 1), 0)
        del buf180  # reuse
        buf185 = reinterpret_tensor(buf175, (1, 197, 768), (151296, 768, 1), 0)
        del buf175  # reuse
        # Source Nodes: [add_14, x_34, x_35, y_14], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf181,
            arg93_1,
            buf158,
            buf165,
            arg87_1,
            arg94_1,
            arg95_1,
            buf185,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg87_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf186 = reinterpret_tensor(buf164, (197, 3072), (3072, 1), 0)
        del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf185, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0),
            out=buf186,
        )
        del arg96_1
        buf187 = reinterpret_tensor(buf186, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf186  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_7_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf187, arg97_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg97_1
        buf188 = reinterpret_tensor(buf185, (197, 768), (768, 1), 0)
        del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf187, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0),
            out=buf188,
        )
        del arg98_1
        buf192 = reinterpret_tensor(buf165, (1, 197, 768), (151296, 768, 1), 0)
        del buf165  # reuse
        # Source Nodes: [add_16, x_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf181,
            buf188,
            arg99_1,
            arg100_1,
            arg101_1,
            buf192,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg100_1
        del arg101_1
        buf193 = buf170
        del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf192, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg102_1, (768, 2304), (1, 768), 0),
            out=buf193,
        )
        del arg102_1
        buf194 = reinterpret_tensor(buf192, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf192  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf193, arg103_1, buf194, 151296, grid=grid(151296), stream=stream0
        )
        buf195 = reinterpret_tensor(buf158, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf158  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf193, arg103_1, buf195, 151296, grid=grid(151296), stream=stream0
        )
        buf196 = buf172
        del buf172  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf193, arg103_1, buf196, 151296, grid=grid(151296), stream=stream0
        )
        del arg103_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf197 = aten._scaled_dot_product_flash_attention.default(
            buf194, buf195, buf196, scale=0.125
        )
        del buf194
        buf198 = buf197[0]
        del buf197
        buf203 = reinterpret_tensor(buf196, (197, 768), (768, 1), 0)
        del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf198, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0),
            out=buf203,
        )
        del arg104_1
        buf204 = reinterpret_tensor(buf203, (1, 197, 768), (151296, 768, 1), 0)
        del buf203  # reuse
        buf208 = reinterpret_tensor(buf198, (1, 197, 768), (151296, 768, 1), 0)
        del buf198  # reuse
        # Source Nodes: [add_16, x_38, x_39, y_16], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf204,
            arg105_1,
            buf181,
            buf188,
            arg99_1,
            arg106_1,
            arg107_1,
            buf208,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg105_1
        del arg106_1
        del arg107_1
        del arg99_1
        buf209 = reinterpret_tensor(buf187, (197, 3072), (3072, 1), 0)
        del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf208, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg108_1, (768, 3072), (1, 768), 0),
            out=buf209,
        )
        del arg108_1
        buf210 = reinterpret_tensor(buf209, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf209  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_8_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf210, arg109_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg109_1
        buf211 = reinterpret_tensor(buf208, (197, 768), (768, 1), 0)
        del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf210, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg110_1, (3072, 768), (1, 3072), 0),
            out=buf211,
        )
        del arg110_1
        buf215 = reinterpret_tensor(buf188, (1, 197, 768), (151296, 768, 1), 0)
        del buf188  # reuse
        # Source Nodes: [add_18, x_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf204,
            buf211,
            arg111_1,
            arg112_1,
            arg113_1,
            buf215,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg112_1
        del arg113_1
        buf216 = buf193
        del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf215, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg114_1, (768, 2304), (1, 768), 0),
            out=buf216,
        )
        del arg114_1
        buf217 = reinterpret_tensor(buf215, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf215  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf216, arg115_1, buf217, 151296, grid=grid(151296), stream=stream0
        )
        buf218 = reinterpret_tensor(buf181, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf181  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf216, arg115_1, buf218, 151296, grid=grid(151296), stream=stream0
        )
        buf219 = buf195
        del buf195  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf216, arg115_1, buf219, 151296, grid=grid(151296), stream=stream0
        )
        del arg115_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf220 = aten._scaled_dot_product_flash_attention.default(
            buf217, buf218, buf219, scale=0.125
        )
        del buf217
        buf221 = buf220[0]
        del buf220
        buf226 = reinterpret_tensor(buf219, (197, 768), (768, 1), 0)
        del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf221, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0),
            out=buf226,
        )
        del arg116_1
        buf227 = reinterpret_tensor(buf226, (1, 197, 768), (151296, 768, 1), 0)
        del buf226  # reuse
        buf231 = reinterpret_tensor(buf221, (1, 197, 768), (151296, 768, 1), 0)
        del buf221  # reuse
        # Source Nodes: [add_18, x_42, x_43, y_18], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf227,
            arg117_1,
            buf204,
            buf211,
            arg111_1,
            arg118_1,
            arg119_1,
            buf231,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg111_1
        del arg117_1
        del arg118_1
        del arg119_1
        buf232 = reinterpret_tensor(buf210, (197, 3072), (3072, 1), 0)
        del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf231, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg120_1, (768, 3072), (1, 768), 0),
            out=buf232,
        )
        del arg120_1
        buf233 = reinterpret_tensor(buf232, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf232  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_9_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf233, arg121_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg121_1
        buf234 = reinterpret_tensor(buf231, (197, 768), (768, 1), 0)
        del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf233, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg122_1, (3072, 768), (1, 3072), 0),
            out=buf234,
        )
        del arg122_1
        buf238 = reinterpret_tensor(buf211, (1, 197, 768), (151296, 768, 1), 0)
        del buf211  # reuse
        # Source Nodes: [add_20, x_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf227,
            buf234,
            arg123_1,
            arg124_1,
            arg125_1,
            buf238,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg124_1
        del arg125_1
        buf239 = buf216
        del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf238, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg126_1, (768, 2304), (1, 768), 0),
            out=buf239,
        )
        del arg126_1
        buf240 = reinterpret_tensor(buf238, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf238  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf239, arg127_1, buf240, 151296, grid=grid(151296), stream=stream0
        )
        buf241 = reinterpret_tensor(buf204, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf204  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf239, arg127_1, buf241, 151296, grid=grid(151296), stream=stream0
        )
        buf242 = buf218
        del buf218  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf239, arg127_1, buf242, 151296, grid=grid(151296), stream=stream0
        )
        del arg127_1
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf243 = aten._scaled_dot_product_flash_attention.default(
            buf240, buf241, buf242, scale=0.125
        )
        del buf240
        buf244 = buf243[0]
        del buf243
        buf249 = reinterpret_tensor(buf242, (197, 768), (768, 1), 0)
        del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf244, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0),
            out=buf249,
        )
        del arg128_1
        buf250 = reinterpret_tensor(buf249, (1, 197, 768), (151296, 768, 1), 0)
        del buf249  # reuse
        buf254 = reinterpret_tensor(buf244, (1, 197, 768), (151296, 768, 1), 0)
        del buf244  # reuse
        # Source Nodes: [add_20, x_46, x_47, y_20], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf250,
            arg129_1,
            buf227,
            buf234,
            arg123_1,
            arg130_1,
            arg131_1,
            buf254,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg123_1
        del arg129_1
        del arg130_1
        del arg131_1
        buf255 = reinterpret_tensor(buf233, (197, 3072), (3072, 1), 0)
        del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf254, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg132_1, (768, 3072), (1, 768), 0),
            out=buf255,
        )
        del arg132_1
        buf256 = reinterpret_tensor(buf255, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf255  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_10_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf256, arg133_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg133_1
        buf257 = reinterpret_tensor(buf254, (197, 768), (768, 1), 0)
        del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf256, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg134_1, (3072, 768), (1, 3072), 0),
            out=buf257,
        )
        del arg134_1
        buf261 = reinterpret_tensor(buf234, (1, 197, 768), (151296, 768, 1), 0)
        del buf234  # reuse
        # Source Nodes: [add_22, x_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf250,
            buf257,
            arg135_1,
            arg136_1,
            arg137_1,
            buf261,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg136_1
        del arg137_1
        buf262 = buf239
        del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf261, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg138_1, (768, 2304), (1, 768), 0),
            out=buf262,
        )
        del arg138_1
        buf263 = reinterpret_tensor(buf261, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf261  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_3.run(
            buf262, arg139_1, buf263, 151296, grid=grid(151296), stream=stream0
        )
        buf264 = reinterpret_tensor(buf227, (1, 12, 197, 64), (151296, 64, 768, 1), 0)
        del buf227  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(
            buf262, arg139_1, buf264, 151296, grid=grid(151296), stream=stream0
        )
        buf265 = buf241
        del buf241  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(
            buf262, arg139_1, buf265, 151296, grid=grid(151296), stream=stream0
        )
        del arg139_1
        del buf262
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_self_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf266 = aten._scaled_dot_product_flash_attention.default(
            buf263, buf264, buf265, scale=0.125
        )
        del buf263
        del buf264
        buf267 = buf266[0]
        del buf266
        buf272 = reinterpret_tensor(buf265, (197, 768), (768, 1), 0)
        del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf267, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0),
            out=buf272,
        )
        del arg140_1
        buf273 = reinterpret_tensor(buf272, (1, 197, 768), (151296, 768, 1), 0)
        del buf272  # reuse
        buf277 = reinterpret_tensor(buf267, (1, 197, 768), (151296, 768, 1), 0)
        del buf267  # reuse
        # Source Nodes: [add_22, x_50, x_51, y_22], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_9.run(
            buf273,
            arg141_1,
            buf250,
            buf257,
            arg135_1,
            arg142_1,
            arg143_1,
            buf277,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg135_1
        del arg141_1
        del arg142_1
        del arg143_1
        del buf250
        buf278 = reinterpret_tensor(buf256, (197, 3072), (3072, 1), 0)
        del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf277, (197, 768), (768, 1), 0),
            reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0),
            out=buf278,
        )
        del arg144_1
        buf279 = reinterpret_tensor(buf278, (1, 197, 3072), (605184, 3072, 1), 0)
        del buf278  # reuse
        # Source Nodes: [l__self___encoder_layers_encoder_layer_11_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(
            buf279, arg145_1, 605184, grid=grid(605184), stream=stream0
        )
        del arg145_1
        buf280 = reinterpret_tensor(buf277, (197, 768), (768, 1), 0)
        del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(
            reinterpret_tensor(buf279, (197, 3072), (3072, 1), 0),
            reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0),
            out=buf280,
        )
        del arg146_1
        del buf279
        buf284 = reinterpret_tensor(buf257, (1, 197, 768), (151296, 768, 1), 0)
        del buf257  # reuse
        # Source Nodes: [add_24, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(
            buf273,
            buf280,
            arg147_1,
            arg148_1,
            arg149_1,
            buf284,
            197,
            768,
            grid=grid(197),
            stream=stream0,
        )
        del arg147_1
        del arg148_1
        del arg149_1
        del buf273
        del buf280
        buf285 = empty_strided_cuda((1, 1000), (1000, 1), torch.bfloat16)
        # Source Nodes: [x_54], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_10.run(
            arg151_1,
            buf284,
            arg150_1,
            buf285,
            grid=torch._inductor.kernel.mm_common.mm_grid(1, 1000, meta0),
            stream=stream0,
        )
        del arg150_1
        del arg151_1
        del buf284
    return (buf285,)


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
    arg8_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg9_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg10_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg11_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg12_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg13_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg14_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg15_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg16_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg17_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg18_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg19_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg20_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg21_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg22_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg23_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg24_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg25_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg26_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg27_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg28_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg29_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg30_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg31_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg32_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg33_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg34_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg35_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg36_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg37_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg38_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg39_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg40_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg41_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg42_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg43_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg44_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg45_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg46_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg47_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg48_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg49_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg50_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg51_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg52_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg53_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg54_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg55_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg56_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg57_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg58_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg59_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg60_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg61_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg62_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg63_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg64_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg65_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg66_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg67_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg68_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg69_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg70_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg71_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg72_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg73_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg74_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg75_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg76_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg77_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg78_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg79_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg80_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg81_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg82_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg83_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg84_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg85_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg86_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg87_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg88_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg89_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg90_1 = rand_strided((2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg91_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg92_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg93_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg94_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg95_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg96_1 = rand_strided((3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg97_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg98_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg99_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg100_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg101_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg102_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg103_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg104_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg105_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg106_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg107_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg108_1 = rand_strided(
        (3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg109_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg110_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg111_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg112_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg113_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg114_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg115_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg116_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg117_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg118_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg119_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg120_1 = rand_strided(
        (3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg121_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg122_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg123_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg124_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg125_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg126_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg127_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg128_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg129_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg130_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg131_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg132_1 = rand_strided(
        (3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg133_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg134_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg135_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg136_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg137_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg138_1 = rand_strided(
        (2304, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg139_1 = rand_strided((2304,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg140_1 = rand_strided((768, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16)
    arg141_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg142_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg143_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg144_1 = rand_strided(
        (3072, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg145_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg146_1 = rand_strided(
        (768, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg147_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg148_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg149_1 = rand_strided((768,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg150_1 = rand_strided(
        (1000, 768), (768, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg151_1 = rand_strided((1000,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg152_1 = rand_strided(
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
        ]
    )
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
