from math import log
import torch

from torchao.prototype.common.bitpacking import pack, unpack
from torchao.dtypes.uint4 import unpack_uint4, pack_uint4


def benchmark(function, num_runs, setup =None):
    args = setup()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs):
        function(*args)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def test_vs_existing():
    def new_():
        fake_tensor = torch.randint(0, 2**8, (1, 1024,1024), dtype=torch.uint8).cuda()
        packed = pack(fake_tensor, 4, dim=1)
        unpacked = unpack(packed, 4, dim=1)
    def old_():
        fake_tensor = torch.randint(0, 2**8, (1, 1024,1024), dtype=torch.uint8).cuda()
        packed = pack_uint4(fake_tensor)
        unpacked = unpack_uint4(packed)
    new_ = torch.compile(new_, fullgraph=True)
    old_ = torch.compile(old_, fullgraph=True)
    new_()
    old_()
    print(f"new: {benchmark(new_, 1000)} ms ")
    print(f"old: {benchmark(old_, 1000)} ms")

    
def test_iso_bitpack():
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


def test_vs_hqqpack():
    #requires hqq to be installed
    import hqq
    import hqq.core.quantize as hqq_quantize
    HQQLinear = hqq_quantize.HQQLinear
    BaseQuantizeConfig = hqq_quantize.BaseQuantizeConfig
    from torchao.prototype.hqq import pack_2xint4, triton_mixed_mm
    
    BASE_QUANT_CONFIG = {
        "optimize": True,
        "view_as_float": False,
        "nbits": 4,
        "bitpack": False,
        "axis": 1,
    }
    
    def mixed_mm(
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
    
    shapes = [
    [16, 128, 128],
    [16, 4096, 4096],
    ]
    group_sizes = [64, 128]
    shape = [16, 128, 128]
    group_size = 64
    pack = torch.compile(pack, fullgraph=True)
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
        print("pack time (ms): ", benchmark(test_mixed_mm, 100, 
                                                shape, 
                                                group_size,
                                                1,
                                                torch.float16,
                                                True,
                                                "compute_bound",
                                                torch.uint8))
        
        print("pack_2xint4 time (ms): ", benchmark(test_mixed_mm, 100, 
                                                shape, 
                                                group_size,
                                                1,
                                                torch.float16,
                                                True,
                                                "compute_bound", #max autotune doesnt work?
                                                torch.uint8,
                                                pack_fn=False))
        print("")


if __name__ == "__main__":
    test_vs_existing()

