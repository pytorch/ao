# %%
import torchao
import torch


# %%
from distutils.sysconfig import get_python_lib
prefix = get_python_lib()

# %%
import sys; print(sys.path)

torch.ops.load_library(f"{prefix}/torchao-0.3.0-py3.11-macosx-11.1-arm64.egg/torchao/libexecutorch_kernels.dylib")

# %%
from executorch.extension.pybindings import portable_lib

# %%
print("\n".join(portable_lib._get_operator_names()))

# %%
class Demo(torch.nn.Module):
    def forward(self, A, B, scalesAndZeros):
        return torch.ops.torchao.int4mv.default(A, B, 32, scalesAndZeros)

# %%
M = 1
K = 4096
N = 4096
A = torch.randn(M, K, device="mps", dtype=torch.float16) # M x K
B = torch.randn(N, K // 2, device="mps", dtype=torch.int8) # N x K / 2
scalesAndZeros = torch.randn(N, K // 32, 2, device="mps", dtype=torch.float16) # N x K / 32 x 2

# %%
module = Demo()
module.forward(A, B, scalesAndZeros)

# %%
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge, EdgeCompileConfig
ep = torch.export.export(module, (A, B, scalesAndZeros))

# %%
executorch_program = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()

# %%
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
m = _load_for_executorch_from_buffer(executorch_program.buffer)
out = m.forward([A.to(device="cpu"), B.to(device="cpu"), scalesAndZeros.to(device="cpu")])
# %%
print(executorch_program.exported_program())

# %%
print(out)

# %%



