import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm
from torchao.ops import rowwise_scaled_linear_cutlass_s8s4


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3

def get_rowwise_problem(m: int, n: int, k: int, A_nbits: int, B_nbits: int):
    assert A_nbits in (4, 8) and B_nbits in (4, 8)

    dev = torch.device("cuda")
    A = torch.randint(-128, 127, (m, k * A_nbits // 8), dtype=torch.int8, device=dev)
    A_scale = torch.randn((m,), dtype=torch.half, device=dev)
    B = torch.randint(
        -128, 127, size=(n, k * B_nbits // 8), dtype=torch.int8, device=dev
    )
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A, A_scale, B, B_scale, C

def get_blockwise_problem(m: int, n: int, k: int, block_size: int):
    assert n % block_size == 0 and k % block_size == 0, "N and K dims must be divisible by block_size"
    dev = torch.device("cuda")
    A = (448.0 * (2 * torch.rand(m, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    A_scale = torch.randn((m, k // block_size), dtype=torch.half, device=dev)
    B = (448.0 * (2 * torch.rand(n, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    B_scale = torch.randn((n // block_size, k // block_size), dtype=torch.half, device=dev)

    return A, A_scale, B, B_scale

def benchmark(m: int, k: int, n: int, block_size: int):
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((n, k), dtype=torch.half, device=dev)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale, C = get_rowwise_problem(m, n, k, 8, 8)
    rowwise_scaled_linear_cutlass_s8s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s8s4, A, A_scale, B, B_scale, C
    )

    A, A_scale, B, B_scale = get_blockwise_problem(m, n, k, block_size)
    blockwise_fp8_gemm_time = benchmark_microseconds(
        blockwise_fp8_gemm, A, A_scale, B, B_scale
    )

    # Add precision tests
    # On prend 2 sets de matrices aléatoires
    # On les quantise en int8/int4 rowwise
    # On les quantise en en float8 blockwise
    # 


    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "rowwise_scaled_linear_cutlass_s8s4 latency (ms)": rowwise_scaled_linear_cutlass_s8s4_time,
        "rowwise s8s4 speedup (d/s)": fp16_time / rowwise_scaled_linear_cutlass_s8s4_time,
        "blockwise_fp8_gemm latency (ms)": blockwise_fp8_gemm_time,
        "blockwise fp8 speedup (d/s)": fp16_time / blockwise_fp8_gemm_time,
    }


from torchao.prototype.blockwise_fp8.blockwise_quantization import fp8_blockwise_weight_quant, fp8_blockwise_act_quant, fp8_blockwise_weight_dequant
from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm

def test_quant_dequant():
    torch.manual_seed(0)
    x = torch.randn(256, 256).cuda()
    qx, s = fp8_blockwise_weight_quant(x, block_size=128)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s, block_size=128)

    error = torch.norm(x - x_reconstructed) / torch.norm(x)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.05, "Quant-Dequant error too high!"

def test_blockwise_fp8_gemm():
    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()

    C = A @ B.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size=128)
    B_q, B_s = fp8_blockwise_weight_quant(B, block_size=128)

    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)

    error = torch.norm(C - C_q) / torch.norm(C)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.05, "Quantized GEMM error is too high!"


test_quant_dequant()
test_blockwise_fp8_gemm()


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)
    block_size_vals = (128, 128, 128, 128)

    results = []
    for m in tqdm([1 << i for i in range(10)]):
        for n, k, block_size in zip(n_vals, k_vals, block_size_vals):
            results.append(benchmark(m, k, n, block_size))

    df = pd.DataFrame(results)
    df.to_csv("blockwise_scaled_linear_triton_time_results.csv", index=False)
    print(df.to_markdown(index=False))