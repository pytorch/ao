from math import log
from copy import deepcopy

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from torchao.prototype.intx.bitpacking import pack, unpack, pack_cpu, unpack_cpu
from torchao.dtypes.uint4 import unpack_uint4, pack_uint4
from torchao.quantization.quant_api import quantize, intx_weight_only, int8_weight_only


def benchmark(function, args, num_runs):
    # warmup
    for i in range(100):
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

def profile_function(function, args, num_runs):
    function(*args)
    torch.cuda.synchronize()
    with profile(activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
        with record_function("model_inference"):
            for _ in range(num_runs):
                function(*args)
        
    # Print a summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

def profile_bitpack():
        
    fake_tensor = [torch.randint(2**8, (512,512), dtype=torch.uint8).cuda()]
    func = torch.compile(unpack_cpu, fullgraph=True)
    with profile(activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
        ) as prof:
        
        for _ in range(1000):
            unpacked = func(fake_tensor, 4)
        
    # Print a summary
    with open("profile-bitpack.txt", "a") as f:
        print(f'{func}',file=f)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10), file=f)
    prof.export_chrome_trace("trace.json")
    '''
    gpu unpack  on cpu Self CPU time total: 602.501ms
    cpu unpack on cpu Self CPU time total: 415.469ms
    
    gpu unpack on gpu: 
    Self CPU time total: 58.512ms
    Self CUDA time total: 5.083ms
    
    cpu unpack on gpu:
    Self CPU time total: 96.947ms
    Self CUDA time total: 5.253ms
    '''
            
def intx_vs_fp16(nbits= [1,2,3,4,5,6,7], scales=[256, 512, 1024],layouts=["plain","packed"], repeats=30):
    class Linear16(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(scale * 2, scale, bias=False, dtype=torch.float16).cuda(),
                torch.nn.Linear(scale, scale, bias=False, dtype=torch.float16).cuda(),
                torch.nn.Linear(scale, scale//2, bias=False, dtype=torch.float16).cuda(),
            )

        def forward(self, x):
            return self.net(x)
    
    results  = []
    nbits.sort()
    scales.sort()
    for scale in scales:
        print("scale: ", scale)
        test_input = torch.randn(scale*2, dtype=torch.float16).cuda()
        forward_args = [test_input]
        times = [scale]
        
        fp16 = Linear16(scale)
        fp16c = torch.compile(fp16, fullgraph=True)
        fp16_time = benchmark(fp16c.forward, forward_args, repeats)
        times.append(fp16_time)
        print('fp16 done')
        for bit_size in nbits:
            for layout in layouts:
                intx = deepcopy(fp16)
                intx = quantize(intx, intx_weight_only(bit_size, group_size=64, layout=layout))
                intx = torch.compile(intx, fullgraph=True)
                intx_time = benchmark(intx.forward, forward_args, repeats)
                times.append(intx_time)
                print(f'int{bit_size}-{layout} done')
            torch._dynamo.reset()
        results.append(times)
    print("----------- benchmark results -----------")
    for result in results:
        result_str = "\n".join([f"int{nbits[i]}: {result[1]/result[2+len(layouts)*i]:.2f}x\t{result[1]/result[3+len(layouts)*i]:.2f}x\t{result[1]/result[4+i]:.2f}x" for i in range(len(nbits))])
        print(f"scale: {result[0]} fp16 time:{result[1]: .2f}ms {layouts} speedups:\n{result_str}")
    
        
        
if __name__ == "__main__":
    # test_bitpack_iso()
    # profile_intx(4)
    # profile_intx(6)
    # profile_intx()
    intx_vs_fp16(nbits=[5,6,7],scales=[4096], layouts = ["plain","packed"], repeats =10000)
