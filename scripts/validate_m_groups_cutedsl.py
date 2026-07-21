import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_rearrange_2d_m_groups import (
    mx_block_rearrange_2d_m_groups_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    mx_block_rearrange_2d_M_groups_cuda,
    triton_mx_block_rearrange_2d_M_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs


def benchmark_cuda_graph_function_in_microseconds(f, *args, iters=1000, **kwargs):
    for _ in range(10):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f(*args, **kwargs)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def benchmark_function(f, *args, use_cuda_graph=False, graph_iters=1000, **kwargs):
    if use_cuda_graph:
        return benchmark_cuda_graph_function_in_microseconds(
            f,
            *args,
            iters=graph_iters,
            **kwargs,
        )
    return benchmark_cuda_function_in_microseconds(f, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--cols", type=int, default=224)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--multiple-of", type=int, default=128)
    parser.add_argument(
        "--chunk-width", type=int, default=0, choices=(0, 16, 32, 64, 128)
    )
    parser.add_argument(
        "--bench-only",
        choices=("cuda", "triton", "cutedsl"),
        default=None,
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit(f"Requires SM 10.x; got {torch.cuda.get_device_capability()}")
    if not _mxfp8_cutedsl_kernels_available:
        raise SystemExit("CuteDSL mxfp8 kernels are unavailable")

    if args.deterministic:
        scales = (
            torch.arange(args.rows * args.cols, device="cuda", dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(args.rows, args.cols)
            .view(torch.float8_e8m0fnu)
        )
    else:
        scales = torch.randint(
            0,
            255,
            (args.rows, args.cols),
            device="cuda",
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)
    scales_ref = scales.contiguous()
    offsets = generate_jagged_offs(
        args.groups,
        args.rows,
        multiple_of=args.multiple_of,
        device="cuda",
        dtype=torch.int32,
    )

    chunk_width = None if args.chunk_width == 0 else args.chunk_width

    print(
        f"shape={tuple(scales.shape)} strides={scales.stride()} groups={args.groups} chunk_width={chunk_width or 'auto'}"
    )
    if args.bench_only is None:
        out_cuda = mx_block_rearrange_2d_M_groups_cuda(scales_ref, offsets)
        out_triton = triton_mx_block_rearrange_2d_M_groups(scales_ref, offsets)
        out_cutedsl = mx_block_rearrange_2d_m_groups_cutedsl(
            scales,
            offsets,
            chunk_width=chunk_width,
        )
        torch.cuda.synchronize()

        print(f"triton_equal={torch.equal(out_cuda, out_triton)}")
        print(f"cutedsl_equal={torch.equal(out_cuda, out_cutedsl)}")
        if not torch.equal(out_cuda, out_cutedsl):
            diff = out_cuda.view(torch.uint8) != out_cutedsl.view(torch.uint8)
            print(f"cutedsl_diff_count={diff.sum().item()}")
            coords = diff.nonzero()[:16]
            idx = coords[:, 0] * out_cuda.shape[1] + coords[:, 1]
            cuda_flat = out_cuda.view(torch.uint8).flatten()
            cutedsl_flat = out_cutedsl.view(torch.uint8).flatten()
            print(f"first_diff_coords_v2={coords.cpu().tolist()}")
            print(f"first_diff_flat_indices_v2={idx.cpu().tolist()}")
            print(f"cuda_values={cuda_flat[idx].cpu().tolist()}")
            print(f"cutedsl_values={cutedsl_flat[idx].cpu().tolist()}")
            print(f"offsets={offsets.cpu().tolist()}")
            if args.deterministic:
                first_flat = int(idx[0].item())
                sf_tile = first_flat // 512
                tile_byte = first_flat % 512
                sf_tiles_per_row = out_cuda.shape[1] // 4
                row_chunk = sf_tile // sf_tiles_per_row
                sf_col_tile = sf_tile % sf_tiles_per_row
                local_col = sf_col_tile * 4
                local_row = None
                local_byte = None
                for r in range(128):
                    off = (r % 32) * 16 + (r // 32) * 4
                    if off <= tile_byte < off + 4:
                        local_row = r
                        local_byte = tile_byte - off
                        break
                offsets_cpu = offsets.cpu().tolist()
                prev = 0
                chunk_base = 0
                input_row = None
                for g, end in enumerate(offsets_cpu):
                    chunks = (end - prev + 127) // 128
                    if row_chunk < chunk_base + chunks:
                        input_row = prev + (row_chunk - chunk_base) * 128 + local_row
                        break
                    chunk_base += chunks
                    prev = end
                print(
                    f"decoded_first_diff=row_chunk:{row_chunk} sf_col_tile:{sf_col_tile} local_row:{local_row} local_col:{local_col} local_byte:{local_byte} input_row:{input_row}"
                )
                if input_row is not None:
                    src = scales.view(torch.uint8)[
                        input_row, local_col : local_col + 16
                    ]
                    print(f"decoded_source16={src.cpu().tolist()}")

    if args.bench_only in (None, "cuda"):
        cuda_us = benchmark_function(
            mx_block_rearrange_2d_M_groups_cuda,
            scales_ref,
            offsets,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cuda_us={cuda_us:.2f}")
    if args.bench_only in (None, "triton"):
        triton_us = benchmark_function(
            triton_mx_block_rearrange_2d_M_groups,
            scales_ref,
            offsets,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"triton_us={triton_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_function(
            mx_block_rearrange_2d_m_groups_cutedsl,
            scales,
            offsets,
            chunk_width,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
