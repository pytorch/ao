# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import collections
import enum
import json
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import do_bench_using_profiling

from torch.profiler import profile, ProfilerActivity, record_function

def profiler_output_to_filtered_time_by_kernel_name(
    prof, 
    num_iter: int,
    num_leaf_tensors: int,
):
    """
    Input: 
      * `prof`: a profiler with captured events
      * `num_iter`: number of iterations used to capture `prof`
      * `num_leaf_tensors`: number of leaf tensors to accumulate gradients to
    Output: a deduplicated list of GPU time in nanoseconds grouped by CPU kernel name,
      with the microbenchmark overhead filtered out

    Currently assumes that `prof` captured events from a microbenchmark which was
    set up as follows:

        #
        # Forward pass 
        #

        # Expected GPU kernel overhead: none
        y = func(...)

        # Convenient way to set up the backward pass without caring about shapes
        y_sum = y.sum()

        # Expected GPU kernel overhead:
        # * the call to `sum`

        #
        # Backward pass
        #
        y_sum.backward()

        # Expected GPU kernel overhead:
        # * the call to `aten.fill_` to put a tensor with a single 1.0 value as the input to the backward
        # * the call to `aten.copy_` to fill the first `grad_output` tensor with 1.0
        # * the call to `aten.add_` to accumulate grads, once per leaf tensor

    Note that if there are user_annotations in the captured events, `torch.profiler`
    will include their time in the total GPU time displayed at the bottom of
    `key_averages.table()`. The filter below excludes them to prevent double
    counting.
    """
    key_averages = prof.key_averages()
    thresh = 1e-10
    kernel_name_to_gpu_time_us = collections.defaultdict(float)
    for e in key_averages:

        # manually filter top-level CPU events with attributed CUDA time
        # example CPU event row from printing `key_averages`:
        #                                               aten::addmm         0.83%      76.554us         0.98%      90.846us      90.846us       1.022ms        31.82%       1.022ms       1.022ms             1
        # and it maps to this CUDA event:
        #   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64...         0.00%       0.000us         0.00%       0.000us       0.000us       1.022ms        31.82%       1.022ms       1.022ms             1
        if not (e.self_cpu_time_total > thresh and e.self_device_time_total > thresh):
            continue

        kernel_name_to_gpu_time_us[e.key] = e.self_device_time_total
    return kernel_name_to_gpu_time_us


def profiler_output_to_gpu_time_for_key(prof, key):
    """
    Input: an event name
    Output: sum of GPU time of all events with that name in `prof`

    This is useful to get the total time of a user annotation
    """
    total = 0
    for e in prof.profiler.function_events:
        if e.key == key:
            total += e.device_time_total
    return total


def kernel_name_to_category(k):
    # number prefix is for easy sorting
    if k in ("aten::mm", "aten::addmm", "aten::_scaled_mm"):
        return "0_gemm"
    elif (
        # max(abs(tensor))
        ("abs" in k and "max" in k)
        or
        # casting pointwise to float8
        ("clamp" in k)
        or
        # things related to scaled_mm
        ("scaled_mm" in k)
        or
        # syncing amaxes and scales
        ("roll" in k)
    ):
        # note: the above filter is approximate and will give false
        # positives if model code contains other code to abs/max/clamp
        return "1_f8_overhead"
    return "2_other"


def parse_bw_and_kernel_name(line):
    """
    Input: a single line of stdout of TORCHINDUCTOR_PROFILE=1 output, such as
        0.257ms         0.537 GB         2092.43GB/s     triton_red_fused_native_layer_norm_0
    Output: the bandwidth value and the kernel name, or None and None
    """
    result = re.search(".* ([0-9\.]+)GB/s.*(triton_[a-z_0-9]+)", line)
    if result:
        return result.group(1), result.group(2)
    else:
        return None, None


def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int],
    K: Optional[int],
    N: Optional[int],
):
    if shape_gen_name == 'llama':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        # LLaMa 2 70B single-node weight shapes
        # assumes fused attn.wqkv and ffn.w13
        # source: https://fburl.com/gsheet/g8onr7rh
        name_to_shapes_70b = {
            "attn.wqkv": (M, 8192, 1280),
            "attn.w0": (M, 1024, 8192),
            "ffn.w13": (M, 8192, 7168),
            "ffn.w2": (M, 3584, 8192),
        }
        return name_to_shapes_70b.items()

    elif shape_gen_name == 'square':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_power_of_2 = 8  # 256
        max_power_of_2 = 15  # 32,768
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val = 2 ** power_of_2
            name_to_shapes[idx] = val, val, val
        return name_to_shapes.items()

    elif shape_gen_name == 'sweep':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_p2 = 8  # 256
        max_p2 = 15  # 32,768
        counter = 0
        for M_p2 in range(min_p2, max_p2 + 1):
            M = 2 ** M_p2
            for K_p2 in range(min_p2, max_p2 + 1):
                K = 2 ** K_p2
                for N_p2 in range(min_p2, max_p2 + 1):
                    N = 2 ** N_p2
                    name_to_shapes[counter] = M, K, N
                    counter += 1
        return name_to_shapes.items()

    elif shape_gen_name == 'custom':
        assert M is not None and K is not None and N is not None, \
            'M, K, N must be specified for custom shape_gen'
        name_to_shapes = {
            1: (M, K, N),
        }
        return name_to_shapes.items()

    raise AssertionError(f'unknown shape_gen_name {shape_gen_name}')

# copy-pasta from https://github.com/vkuzo/pytorch_scripts/blob/main/add_inductor_metadata_to_perf_trace.py
def update_triton_kernels_in_prof_chome_trace_with_torch_logs(
    perf_trace_file: str,
    torch_logs_file: str,
    modified_perf_trace_file: str,
):
    """
    Input 1: a perf trace generated by using `torch.profiler.profile` inside of 
      some_program.py, and containing torch.compile + inductor kernels
    Input 2: a text file with the output of 
      TORCH_LOGS="output_code" python some_program.py
    Input 3: filename for the modified perf trace

    This script does the following for each triton kernel in input 1:
    - navigate to the kernel information in the logs from input 2
    - copy over the kernel metadata (aten graph, triton code, etc) to the JSON 
      in input 1

    The end result is that Input 1 is modified so that the kernel metadata is 
    directly visible in tools like chrome://tracing and perfetto.
    """

    external_id_to_cpu_ops = dict()
    external_id_to_kernels = dict()

    # open the torch logs file
    torch_logs_str = None
    with open(torch_logs_file, 'r') as f:
        torch_logs_str = f.readlines()

    # strip away the torch_logs prefix
    torch_logs_only = []
    for line in torch_logs_str:
        line = line.replace('\n', '')
        match = re.match('.* \[__output_code\] (.*)', line)
        if match:
            torch_logs_only.append(match.group(1))

    # Find the locations of the kernel metadata in the logs.
    # metadata format, haven't been extensively tested so may be brittle:
    #
    #   ...[__output_code]: # kernel_path: /tmp/torchinductor_...
    #   ...[__output_code]: ...
    #   ...[__output_code]: triton_red_fused_LayerNorm_3 = async_compile.triton('triton_', '''
    #   ...[__output_code]: ...
    #   ...[__output_code]: ''', device_str='cuda')
    #
    # We look for the first and last line and save everything in between
    name_to_start_end = {}
    cur_start, cur_end, cur_name = None, None, None
    for line_num, line in enumerate(torch_logs_only):
        match_start = re.match('\# kernel path: .*', line)
        if match_start:
            cur_start = line_num

        # triton_red_fused_LayerNorm_3 = async_compile.triton('triton_', '''
        match_name = re.match("([\w_]+) = async_compile.*", line)
        if match_name:
            cur_name = match_name.group(1)

        match_end = re.match("''', device_str='cuda'\)", line)
        if match_end:
            cur_end = line_num

            # populate the mapping and reset
            name_to_start_end[cur_name] = (cur_start, cur_end)
            cur_start, cur_end, cur_name = None, None, None

    # ensure matching didn't have loose ends
    assert cur_start is None and cur_end is None and cur_name is None

    # Now, go through the JSON file and populate the extra metadata
    # Format of the relevant parts of the perf trace JSON:
    # {
    #   ...
    #   // CPU ops, with names matchable to triton kernels from inductor output code
    #   {
    #     # "cat": "cpu_op", 
    #     # "name": "triton_red_fused_LayerNorm_abs_max_0",
    #     # "args": {"External id": 1030, ...},
    #     # ...
    #   },
    #   // Inductor kernels, with wall time
    #   {
    #     # "cat": "kernel", 
    #     # "name": "triton_",  // we don't depend on this name, including for context
    #     # "args": {"External id": 1030, ...},
    #     # "ts": 4275686082015.124, // start time
    #     # "dur": 208.640,  // duration
    #     # ...
    #   },
    # }
    #
    # We can't assume any ordering, so we do two passes:
    # 1. Find mapping of cpu_op to external_id
    # 2. Using 1, add the metadata to triton kernels

    # open the perf trace json
    with open(perf_trace_file, 'r') as f:
        perf_trace = json.load(f)

    # find mapping of cpu_op to external_id
    external_id_to_cpu_op = dict()
    for record in perf_trace['traceEvents']:
        # print(record)
        is_cpu_op = record.get('cat') == 'cpu_op'
        if is_cpu_op:
            external_id_to_cpu_op[record['args']['External id']] = record['name']

    # add the metadata to triton kernels
    for record in perf_trace['traceEvents']:
        is_triton_kernel = record.get('cat') == 'kernel' and 'triton' in record.get('name', '')
        if not is_triton_kernel:
            continue
        op_name = external_id_to_cpu_op.get(record['args']['External id'])
        if op_name is None:
            continue
        start, end = name_to_start_end[op_name]
        triton_code = torch_logs_only[start:end+1]
        s = ''
        for line in triton_code:
            s += f'{line}\n'
        record['args']['triton_code'] = s

    # write the modified file
    # out_file = perf_trace_file.replace('.json', '') + '_with_metadata.json'
    with open(modified_perf_trace_file, 'w') as f:
        json.dump(perf_trace, f)


def get_gpu_kernel_gemm_time_s(f, *args, **kwargs):
    # warmup
    f(*args, **kwargs)
    n_iter = 5
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for idx in range(n_iter):
            f(*args, **kwargs) 
    data = profiler_output_to_filtered_time_by_kernel_name(prof, n_iter, num_leaf_tensors=0) 
    # there is only 1 key, aten::mm or aten::_scaled_mm, with unit nanoseconds
    assert len(data) == 1
    if "aten::mm" in data:
        return data["aten::mm"] / 1e6 / n_iter
    elif "aten::_scaled_mm" in data:
        return data["aten::_scaled_mm"] / 1e6 / n_iter
    else:
        raise AssertionError("unexpected format of data")


class LNLinear(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py
class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class NormFFNResidualNorm(nn.Module):
    """
    A fragment representing the end of TransformerBlock n and the start
    of TransformerBlock n + 1, intended to include the fusions relevant
    to float8 gemms in the FFN module in forward and backward.
    """

    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.attn_norm = RMSNorm(dim)

    def forward(self, h):
        # end of transformer block n
        x = self.ffn_norm(h)
        x = self.ffn(x)
        x = h + x
        # start of transformer block n + 1
        x = self.attn_norm(x)
        return x

#
# The following model definitions intend to capture all the 
# relevant (prev_op -> linear -> [next_op, ...]) subgraphs from the LLaMa model
# family.  We use https://github.com/pytorch/torchtitan/tree/main/torchtitan/models/llama
# as the reference code structure, and manually define all the subgraphs
# around linear layers.
#
# The goal here is to set up microbenchmarks on the fusion of float8 
# scaling/casting that are relevant to transformer models, and break
# them up in such a way that we can easily reason about fusions regarding
# each linear layer without having to dig through large traces / long log files.
# There is some duplication here (attn_norm and ffn_norm appear in multiple
# places), and that's ok.
#
# Simplifying assumptions (may be relaxed in the future):
# 1. don't model fusion with rotary embeddings and repeat-transpose
# 2. don't model fusion with SDPA
#
# Patterns:
# 1. add -> attn_norm -> attn.{wq|wk|wv}
#   - note: rotary embedding, repeat-transpose, SDPA not modeled for simplicity
#   - note: model just one of (q, k, v) and assume the others are similar
# 2. transpose-contiguous -> attn.wo -> add -> ffn_norm
# 3. add -> ffn_norm -> ffn.w1 -> silu -> mul
#   - note: assume that `add -> ffn_norm -> ffn.w3 -> mul` is covered by above
# 4. silu -> mul -> ffn.w2 -> add -> attn_norm
#   - note: in today's torchtitan codebase, the transformer blocks are compiled 
#     separately, so `add` and `attn_norm` are not in the same graph
#

class AttnWQKVSubgraph(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.attn_norm = RMSNorm(dim1)
        self.fc = nn.Linear(dim1, dim2, bias=False)

    def forward(self, x, y):
        x = x + y
        x = self.attn_norm(x)
        x = self.fc(x)
        return x


class AttnWOSubgraph(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc = nn.Linear(dim1, dim2, bias=False)
        self.ffn_norm = RMSNorm(dim2)

    def forward(self, x, y):
        x = x.transpose(0, 1).contiguous()
        x = self.fc(x)
        x = x + y
        x = self.ffn_norm(x)
        return x


class FFNW13Subgraph(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.ffn_norm = RMSNorm(dim1)
        self.fc = nn.Linear(dim1, dim2, bias=False)

    def forward(self, x, y):
        x = x + y
        x = self.ffn_norm(x)
        x = self.fc(x)
        x = F.silu(x)
        x = x * x
        return x


class FFNW2Subgraph(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc = nn.Linear(dim1, dim2, bias=False)
        self.attn_norm = RMSNorm(dim2)

    def forward(self, x, y):
        x = F.silu(x)
        x = x * x
        x = self.fc(x)
        x = x + y
        x = self.attn_norm(x)
        return x


class ModelType(enum.Enum):
    LINEAR = "linear"
    LN_LINEAR = "ln_linear"
    NORM_FFN_NORM = "norm_ffn_norm"
    NORM_FFN_NORM_SMALL = "norm_ffn_norm_small"
    ATTN_WQKV_SUBGRAPH = "attn_wqkv_subgraph"
    ATTN_WO_SUBGRAPH = "attn_wo_subgraph"
    FFN_W13_SUBGRAPH = "ffn_w13_subgraph"
    FFN_W2_SUBGRAPH = "ffn_w12_subgraph"


def subgraph_name_to_subgraph_and_inputs(
    model_type: str,
    mkn_override: Optional[Tuple[int, int, int]] = None,
):
    try:
        model_type = ModelType(model_type)
    except ValueError as e:
        valid_values = [e.value for e in ModelType]
        print(f"invalid model_type {model_type}, valid values: {valid_values}")
        raise e

    device = "cuda"
    ref_dtype = torch.bfloat16
    if model_type is ModelType.LN_LINEAR:
        M, K, N = 4 * 4096, 8192, 7168
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = LNLinear(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )
        input_tensors = (input_tensor,)

    elif model_type is ModelType.NORM_FFN_NORM:
        assert mkn_override is None, "unsupported"
        m_ref = NormFFNResidualNorm(
            dim=4096,
            hidden_dim=16384,
            multiple_of=1024,
            ffn_dim_multiplier=1.3,
        )
        input_tensor = torch.randn(
            1, 8192, 4096, device=device, dtype=ref_dtype
        ).requires_grad_()
        input_tensors = (input_tensor,)

    elif model_type is ModelType.NORM_FFN_NORM_SMALL:
        assert mkn_override is None, "unsupported"
        m_ref = NormFFNResidualNorm(
            dim=4096,
            hidden_dim=4096,
            multiple_of=1024,
            ffn_dim_multiplier=1.0,
        )
        input_tensor = torch.randn(
            1, 2048, 4096, device=device, dtype=ref_dtype
        ).requires_grad_()
        input_tensors = (input_tensor,)

    elif model_type is ModelType.ATTN_WQKV_SUBGRAPH:
        M, K, N = 1024, 2048, 4096
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = AttnWQKVSubgraph(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype).requires_grad_()
        input_tensor2 = torch.randn(
            M, K, device=device, dtype=ref_dtype).requires_grad_()
        input_tensors = (input_tensor, input_tensor2)

    elif model_type is ModelType.ATTN_WO_SUBGRAPH:
        M, K, N = 1024, 2048, 4096
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = AttnWOSubgraph(K, N)
        input_tensor = torch.randn(
            K, M, device=device, dtype=ref_dtype).requires_grad_()
        input_tensor2 = torch.randn(
            M, N, device=device, dtype=ref_dtype).requires_grad_()
        input_tensors = (input_tensor, input_tensor2)

    elif model_type is ModelType.FFN_W13_SUBGRAPH:
        M, K, N = 1024, 2048, 4096
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = FFNW13Subgraph(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype).requires_grad_()
        input_tensor2 = torch.randn(
            M, K, device=device, dtype=ref_dtype).requires_grad_()
        input_tensors = (input_tensor, input_tensor2)

    elif model_type is ModelType.FFN_W2_SUBGRAPH:
        M, K, N = 1024, 2048, 4096
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = FFNW2Subgraph(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype).requires_grad_()
        input_tensor2 = torch.randn(
            M, N, device=device, dtype=ref_dtype).requires_grad_()
        input_tensors = (input_tensor, input_tensor2)

    else:
        assert ModelType is ModelType.LINEAR
        M, K, N = 4096, 4096, 4096
        if mkn_override is not None:
            M, K, N = mkn_override
        m_ref = torch.nn.Sequential(
            torch.nn.Linear(K, N, bias=False),
        )
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )
        input_tensors = (input_tensor,)

    m_ref = m_ref.to(device).to(ref_dtype)
    return m_ref, input_tensors


def do_bench_using_profiling_wrapper_s(fn, *args, **kwargs):
    def fn_inner():
        fn(*args, **kwargs)
    res = do_bench_using_profiling(fn_inner) / 1e3
    return res
