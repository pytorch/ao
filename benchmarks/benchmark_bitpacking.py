from math import log
import torch

from torchao.prototype.common.bitpacking import pack, unpack
from torchao.dtypes.uint4 import unpack_uint4, pack_uint4


def benchmark(function, args, num_runs):
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
    def new_(scale):
        fake_tensor = torch.randint(2**8-1, (1, scale,scale), dtype=torch.uint8).cuda()
        packed = pack(fake_tensor, 4, dim=1)
        unpacked = unpack(packed, 4, dim=1)
    def old_(scale):
        fake_tensor = torch.randint(2**8-1, (1, scale,scale), dtype=torch.uint8).cuda()
        packed = pack_uint4(fake_tensor)
        unpacked = unpack_uint4(packed)
        
    
    for scale in [256,512, 1024, 2048,4096, 8192]:
        new_ = torch.compile(new_,  fullgraph=True)
        old_ = torch.compile(old_,  fullgraph=True)
        new_(scale)
        old_(scale)
        print("scale: ", scale)
        print(f"new: {benchmark(new_,[scale], 10)} ms ")
        print(f"old: {benchmark(old_,[scale], 10)} ms")


def compare_to_fp16():
    class Linear16(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            scale += scale % 2
            self.l1 = torch.nn.Linear(scale * 2, scale, bias=False,dtype=torch.float16).cuda()
            self.l2 = torch.nn.Linear(scale, scale//2, bias=False,dtype=torch.float16).cuda()

        def forward(self, x):
            return self.l2(self.l1(x))
    
    class W4A16_symmetric_weight_only(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            assert scale % 4 == 0
            self.l1 = torch.randint(2**8-1,(scale, scale), dtype=torch.uint8).cuda()
            self.s1 = torch.tensor((scale),dtype=torch.float16).cuda()
            self.l2 = torch.randint(2**8-1,(scale//2, scale//4), dtype=torch.uint8).cuda()
            self.s2 = torch.tensor((scale//4),dtype=torch.float16).cuda()

        
        def forward(self, x):
            w = unpack(self.l1.detach(), 4, output_dtype=torch.float16)
            x = x * self.s1
            x = x @ w
            w = unpack(self.l2.detach(), 4, output_dtype=torch.float16)
            x = x * self.s2
            x = x @ w
            
            return x

    torch._dynamo.config.specialize_int = True
    for scale in [256,512, 1024, 2048,4096, 8192]:
        a = Linear16(scale)
        b = W4A16_symmetric_weight_only(scale)
        # a = torch.compile(a, fullgraph=True)
        b = torch.compile(b, fullgraph=True)
        
        test_input = torch.randn(scale*2, dtype=torch.float16).cuda()
        forward_args = [test_input]
        b.forward(test_input)
        print("scale: ", scale)
        print("fp16 time: ", benchmark(a.forward, forward_args, 100))
        print("uint4 time: ", benchmark(b.forward, forward_args, 100))
    
        
        
if __name__ == "__main__":
    compare_to_fp16()
    test_vs_existing()
    