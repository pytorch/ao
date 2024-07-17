from math import log
from copy import deepcopy

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchao.utils import unwrap_tensor_subclass
from torchao.prototype.intx import IntxTensor, to_intx, pack, unpack, pack_cpu, unpack_cpu
from torchao.dtypes import to_affine_quantized
from torchao.dtypes.affine_quantized_tensor import IntxLayoutType
from torchao.quantization.quant_api import quantize_, int8_weight_only
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
)


def test_intx(bit_size): 
    input_float = torch.randn((1,8), dtype=torch.float16)
    print('input_float', input_float)
    mapping_type = MappingType.SYMMETRIC
    quant_min = 0
    quant_max = 2**bit_size - 1
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int32
    zero_point_domain = ZeroPointDomain.INT
    target_dtype = torch.uint8
    block_size = (1, input_float.shape[1])
    
    scale, zero_point = choose_qparams_affine(
            input_float, mapping_type, block_size,
            target_dtype, quant_min, quant_max, eps, torch.float32, 
            zero_point_dtype, True, zero_point_domain
    )
    
    aqt = quantize_affine(
        input_float, block_size, scale,
        zero_point, target_dtype,
        quant_min = quant_min,
        quant_max = quant_max,
        zero_point_domain = zero_point_domain
        )
        
    q =  to_intx(aqt, bit_size, -1)

    print('quantized', q)
    deqaunt = dequantize_affine(
        q, block_size, scale,
        zero_point, target_dtype,
        quant_min = quant_min,
        quant_max = quant_max,
        zero_point_domain = zero_point_domain
    )
    print("dequantized", deqaunt)
    
    print("difference: ", torch.sum(torch.abs(input_float - deqaunt))/8)
    
class Linear16(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(scale*2, scale*2, bias=False, dtype=torch.float16).cuda(),
            # torch.nn.Linear(scale, scale, bias=False, dtype=torch.float16).cuda(),
            # torch.nn.Linear(scale, scale//2, bias=False, dtype=torch.float16).cuda(),
        )

    def forward(self, x):
        return self.net(x)

def intx_weight_only(bit_size, group_size=64, pack_dim=-1):
    """
    Applies intx weight-only asymmetric per-group quantization to linear layers, using intx quantization where 
    x is the number of bits specified by the `nbits` argument
    """
    
    def apply_intx_weight_only_quant(weight):
        layout_type = IntxLayoutType(bit_size=bit_size, pack_dim=pack_dim) 
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = 0
        quant_max = 2**bit_size - 1
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        
        return to_affine_quantized(
            weight, mapping_type, block_size, torch.uint8, quant_min = quant_min,
            quant_max = quant_max, eps = eps, 
            zero_point_dtype=zero_point_dtype,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type,
        )
    
    return apply_intx_weight_only_quant
    
    
def benchmark(function, args, num_runs):
    # warmup
    torch._dynamo.reset()
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
            
def intx_vs_fp16(nbits= [1,2,3,4,5,6,7], scales=[256, 512, 1024], repeats=30):
    
    
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
            print(bit_size)
            m = deepcopy(fp16)
            # quantize_(m, int8_weight_only())
            quantize_(m, intx_weight_only(bit_size)) 
            m = unwrap_tensor_subclass(m)
            m = torch.compile(m, fullgraph=True)
            print("compiled")
            intx_time = benchmark(m.forward, forward_args, repeats)
            times.append(intx_time)
            print(f'int{bit_size} done')
            
        results.append(times)
    print("----------- benchmark results -----------")
    for result in results:
        result_str = "\n".join([f"int{nbits[i]}: {result[1]/result[2]:.2f}x" for i in range(len(nbits))])
        print(f"scale: {result[0]} fp16 time:{result[1]: .2f}ms speedups:\n{result_str}")
    
        
        
if __name__ == "__main__":
    # test_intx(4)    
    intx_vs_fp16(nbits=[4])
    
    