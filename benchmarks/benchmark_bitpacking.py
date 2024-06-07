# from torchao.quantization.quant_primitives import dynamically_quantize_per_channel
from torchao.prototype.common.bitpacking import pack, unpack
from math import log
# from torchao.utils import benchmark_utils
import torch
pack = torch.compile(pack, fullgraph=True)

def benchmark(function, num_runs, *args, **kwargs):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs):
        function(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs



def load4x(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (1, 4*scale,scale), dtype=torch.uint8).cuda()
    
def load2x(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (1, 2*scale,scale), dtype=torch.uint8).cuda()

def loadx(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (1, scale,scale), dtype=torch.uint8).cuda()

def unpack8to2(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (1, scale,scale), dtype=torch.uint8).cuda()
    unpacked_tensor = unpack_c(fake_tensor, 2, dim=1)
    
def unpack8to4(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (1, scale,scale), dtype=torch.uint8).cuda()
    unpacked_tensor = unpack_c(fake_tensor, 4, dim=1)

def t8to4wmm(scale=1024):
    fake_tensor = torch.randint(0, 2**8, (8, 1024,1024), dtype=torch.uint8).cuda()
    unpacked_tensor = unpack_c(fake_tensor, 4, dim=1)
    
def test_iso_bitpack():    
    torch._dynamo.config.specialize_int = True   
    # _unpack_c = torch.compile(_unpack, fullgraph=True)
    unpack_c = torch.compile(unpack, fullgraph=True)

    scale = [16,64,256,1024,4096]
    load4x_times = []
    unpack8to2_times = []
    load2x_times = []
    unpack8to4_times = []
    for s in scale:
        res = benchmark(load4x, 50, scale=s)
        load4x_times.append(res)
        print(f"load(1, {4*s},{s}) time: {res} ms")
        
        res=benchmark(unpack8to2, 50, scale=s)
        unpack8to2_times.append(res)
        print(f"load(1, {s},{s}) unpack uint2 time: {res} ms")
        
        res = benchmark(load2x, 50, scale=s)
        load2x_times.append(res)
        print(f"load(1, {2*s},{s}) time: {res} ms")
        
        res = benchmark(unpack8to4, 50, scale=s)
        unpack8to4_times.append(res)
        print(f"load(1, {s},{s}) unpack uint4 time: {res} ms")
        print()

    # import matplotlib.pyplot as plt
    # plt.plot(scale, load4x_times, label="load(1, 4x, x)")
    # plt.plot(scale, unpack8to2_times, label="unpack uint8 to uint2")
    # plt.plot(scale, load2x_times, label="load(1, 2x, x)")
    # plt.plot(scale, unpack8to4_times, label="unpack uint8 to uint4")
    # plt.xlabel("scale")
    # plt.ylabel("time (ms)")
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig("benchmark_bitpacking.png")

import hqq
import hqq.core.quantize as hqq_quantize
HQQLinear = hqq_quantize.HQQLinear
BaseQuantizeConfig = hqq_quantize.BaseQuantizeConfig

import itertools
from torchao.prototype.hqq import pack_2xint4, triton_mixed_mm

# Test configs
SHAPES = [
    [16, 128, 128],
    [16, 4096, 4096],
]

DTYPES = [torch.bfloat16, torch.float16]
GROUP_SIZES = [64, 128]
AXES = [1]  # Only axis = 1 supported
TRANSPOSED = [False, True]
TRITON_KERNEL_TYPE = ["compute_bound"]  # ["max_autotune", "compute_bound"]

TEST_CONFIGS = list(
    itertools.product(SHAPES, GROUP_SIZES, AXES, DTYPES, TRANSPOSED, TRITON_KERNEL_TYPE)
)

BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}


def check(expected, actual, msg="", max_diff=1e-3, verbose=False):
    passed = torch.allclose(expected, actual, atol=max_diff, rtol=max_diff)
    if verbose:
        max_err = (expected - actual).abs().max()
        if not passed:
            print_msg = f"{msg}:\nFailed! Max error: {max_err}"
            try:
                from termcolor import colored
            except ImportError:
                print(print_msg)
            else:
                print(colored(print_msg, "red", attrs=["bold"]))

        else:
            print_msg = f"{msg}:\nPassed! Max error: {max_err}"
            try:
                from termcolor import colored
            except ImportError:
                print(print_msg)
            else:
                print(colored(print_msg, "green", attrs=["bold"]))

    return passed


def test_mixed_mm(
    shape, group_size, axis, dtype, transposed, kernel_type, quant_dtype=torch.uint8, pack_fn = True
):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    W_q = W_q.to(dtype=quant_dtype)
    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    W_dq = hqq_linear.dequantize()

    scales, zeros = meta["scale"], meta["zero"]
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)
    if pack_fn:
        packed_w = pack(W_q.T,4,dim=0,order=False)
    else:
        packed_w = pack_2xint4(W_q.T)
    # print(W_q.T[0:5,0:5], W_q.T.shape)
    # print(packed_w[0:5,0:5], W_q.T.shape)
    # print(packed_w2[0:5,0:5], W_q.T.shape)
    if transposed:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=True,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )

    else:
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq.T

        tt_out = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            transposed=False,
            group_size=group_size,
            fp8_fast_accum=False,
            kernel_type=kernel_type,
        )
    # assert check(
    #     hqq_out,
    #     tt_out,
    #     max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3,
    #     verbose=True,
    # )

    
if __name__ == "__main__":
    # _test_mixed_mm(transposed=False)
    shapes = [
    [16, 128, 128],
    [16, 4096, 4096],
    ]
    group_sizes = [64, 128]
    shape = [16, 128, 128]
    group_size = 64
    
    for i in range(2):
        shape = shapes[i]
        group_size = group_sizes[i]
        print("linear layer size: ", shape)
        print("group size: ", group_size)
        # run once to compile
        test_mixed_mm(
            shape,
            group_size,
            1,
            torch.float16,
            True,
            "compute_bound",
            torch.uint8,
        )
        # shape, group_size, axis, dtype, transposed, kernel_type, quant_dtype=torch.uint8
        print("pack time (ms): ", benchmark(test_mixed_mm, 10, 
                                                shape, 
                                                group_size,
                                                1,
                                                torch.float16,
                                                True,
                                                "compute_bound",
                                                torch.uint8))
        
        print("pack_2xint4 time (ms): ", benchmark(test_mixed_mm, 10, 
                                                shape, 
                                                group_size,
                                                1,
                                                torch.float16,
                                                True,
                                                "compute_bound", #max autotune doesnt work?
                                                torch.uint8,
                                                pack_fn=False))
        # print("pack_2xint4 time (ms): ", benchmark(test_mixed_mm, 10, 
        #                                         shape, 
        #                                         group_size,
        #                                         1,
        #                                         torch.float16,
        #                                         True,
        #                                         "compute_bound", #max autotune doesnt work?
        #                                         torch.uint8,
        #                                         pack_fn=False))
        # print("pack time (ms): ", benchmark(test_mixed_mm, 10, 
        #                                         shape, 
        #                                         group_size,
        #                                         1,
        #                                         torch.float16,
        #                                         True,
        #                                         "compute_bound",
        #                                         torch.uint8))
        print("")

