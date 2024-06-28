from math import log
from copy import deepcopy
import torch

from torchao.prototype.intx.bitpacking import pack, unpack
from torchao.dtypes.uint4 import unpack_uint4, pack_uint4
from torchao.quantization.quant_api import quantize, intx_weight_only, int8_weight_only


def benchmark(function, args, num_runs):
    function(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs):
        function(*args)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def test_bitpack_iso():
    def new_(scale):
        fake_tensor = torch.randint(2**8, (1, scale,scale), dtype=torch.uint8).cuda()
        packed = pack(fake_tensor, 4, dim=1)
        unpacked = unpack(packed, 4, dim=1)
    def old_(scale):
        fake_tensor = torch.randint(2**8, (1, scale,scale), dtype=torch.uint8).cuda()
        packed = pack_uint4(fake_tensor)
        unpacked = unpack_uint4(packed)
        
    results = []
    for scale in [256,512, 1024, 2048,4096, 8192]:
        new_ = torch.compile(new_,  fullgraph=True)
        old_ = torch.compile(old_,  fullgraph=True)
        f1 = benchmark(new_,[scale], 100)
        f2 = benchmark(old_,[scale], 100)
        results.append((scale, f1, f2))
    for scale, f1, f2 in results:
        print(f"scale: {scale}\tnew speedup: {f2/f1 : .03f}x")


def intx_vs_fp16():
    class Linear16(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(scale * 2, scale, bias=False).cuda(),
                torch.nn.Linear(scale, scale, bias=False).cuda(),
                torch.nn.Linear(scale, scale, bias=False).cuda(),
                torch.nn.Linear(scale, scale, bias=False).cuda(),
                torch.nn.Linear(scale, scale, bias=False).cuda(),
                torch.nn.Linear(scale, scale//2, bias=False).cuda(),
            )

        def forward(self, x):
            return self.net(x)
    
    

    results  = []
    for scale in [256, 512, 1024, 2048, 4096]:
        print("scale: ", scale)
        test_input = torch.randn(scale*2).cuda()
        forward_args = [test_input]
        
        fp16 = Linear16(scale)
        
        int1 = deepcopy(fp16)
        int1 = quantize(int1, intx_weight_only(1))
        int1 = torch.compile(int1, fullgraph=True)
        int1_time = benchmark(int1.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int2 = deepcopy(fp16)
        int2 = quantize(int2, intx_weight_only(2))
        int2 = torch.compile(int2, fullgraph=True)
        int2_time = benchmark(int2.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int3 = deepcopy(fp16)
        int3 = quantize(int3, intx_weight_only(3))
        int3 = torch.compile(int3, fullgraph=True)
        int3_time = benchmark(int3.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int4 = deepcopy(fp16)
        int4 = quantize(int4, intx_weight_only(4))
        int4 = torch.compile(int4, fullgraph=True)
        int4_time = benchmark(int4.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int5 = deepcopy(fp16)
        int5 = quantize(int5, intx_weight_only(5))
        int5 = torch.compile(int5, fullgraph=True)
        int5_time = benchmark(int5.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int6 = deepcopy(fp16)
        int6 = quantize(int6, intx_weight_only(6))
        int6 = torch.compile(int6, fullgraph=True)
        int6_time = benchmark(int6.forward, forward_args, 100)
        torch._dynamo.reset()
        
        int7 = deepcopy(fp16)
        int7 = quantize(int7, intx_weight_only(7)) 
        int7 = torch.compile(int7, fullgraph=True)
        int7_time = benchmark(int7.forward, forward_args, 100)
        torch._dynamo.reset()
        
        fp16 = torch.compile(fp16, fullgraph=True)
        fp16_time = benchmark(fp16.forward, forward_args, 100)
        torch._dynamo.reset()
        
        # results.append((scale, fp16_time, int3_time, int6_time))
        results.append((scale, fp16_time, int1_time, int2_time, int3_time, int4_time, int5_time, int6_time, int7_time, int8_time))
        
    for scale, fp16_time, int1_time, int2_time, int3_time, int4_time, int5_time, int6_time, int7_time, int8_time in results:
        print(f"scale: {scale} fp16 time: {fp16_time : 05f}" 
              f"\nint1: {fp16_time/int1_time : .05f}x"
              f"\nint2: {fp16_time/int2_time : .05f}x"
              f"\nint3: {fp16_time/int3_time : .05f}x"
              f"\nint4: {fp16_time/int4_time : .05f}x"
              f"\nint5: {fp16_time/int5_time : .05f}x"
              f"\nint6: {fp16_time/int6_time : .05f}x"
              f"\nint7: {fp16_time/int7_time : .05f}x"
              f"\nint8: {fp16_time/int8_time : .05f}x"
            )
    
        
        
if __name__ == "__main__"
    torch._dynamo.config.specialize_int = True
    intx_vs_fp16()
    # test_vs_existing()
    