
# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_blackroot/b2/cb2pk32dkh7fzpcyu2ziiu5kddauyyziy3mm3pwtmnndoq3ip4nt.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*u8', 1: '*i8', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '4cd5bd8106358e28b51e7312c873e76da8dfa4b9aa3bf91f435911b0de1494e2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], 6, tl.uint8)
    tmp7 = tmp5 >> tmp6
    tmp8 = tl.full([1], 3, tl.uint8)
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9.to(tl.int8)
    tmp11 = tl.full([1], 1, tl.int8)
    tmp12 = tmp10 - tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + (x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full([1], 4, tl.uint8)
    tmp21 = tmp19 >> tmp20
    tmp22 = tmp21 & tmp8
    tmp23 = tmp22.to(tl.int8)
    tmp24 = tmp23 - tmp11
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp18, tmp24, tmp25)
    tmp27 = tmp0 >= tmp16
    tmp28 = tl.full([1], 3, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr0 + (x1), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full([1], 2, tl.uint8)
    tmp33 = tmp31 >> tmp32
    tmp34 = tmp33 & tmp8
    tmp35 = tmp34.to(tl.int8)
    tmp36 = tmp35 - tmp11
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp30, tmp36, tmp37)
    tmp39 = tmp0 >= tmp28
    tmp40 = tl.full([1], 4, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr0 + (x1), tmp39, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp42 & tmp8
    tmp44 = tmp43.to(tl.int8)
    tmp45 = tmp44 - tmp11
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp39, tmp45, tmp46)
    tmp48 = tl.where(tmp30, tmp38, tmp47)
    tmp49 = tl.where(tmp18, tmp26, tmp48)
    tmp50 = tl.where(tmp4, tmp14, tmp49)
    tl.store(out_ptr0 + (x2), tmp50, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 16, 2), (32, 2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 16, 2, 4), (128, 8, 4, 1), torch.int8)
        # Source Nodes: [stack], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_0.run(arg0_1, buf0, 131072, grid=grid(131072), stream=stream0)
        del arg0_1
    return (reinterpret_tensor(buf0, (1024, 16, 8), (128, 8, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 16, 2), (32, 2, 1), device='cuda:0', dtype=torch.uint8)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
