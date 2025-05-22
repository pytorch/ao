import os
from datetime import datetime

import torch
import torch.nn.functional as F

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8


def _dump_metadata_format_cutlass(
    rows=128, cols=256, device=torch.device("cuda"), dtype=torch.float8_e4m3fn
):
    """
    This is a helper function to dump the metadata packing format for 2:4 sparse GEMMS.

    We create a 2:4 sparse tensor by tiling the same pattern and then changing a singular 1x4 strip of the metadata at a time.
    This will allow us to find the corresponding location in the metadata that changes.
    """

    # We tile the same pattern [0, 0, 1, 1] which yields 238 for all metadata values.
    dense_reference_tensor = (
        torch.Tensor([0, 0, 1, 1])
        .to(device=device, dtype=dtype)
        .tile((rows, cols // 4))
        .contiguous()
    )
    _, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(
        dense_reference_tensor
    )
    print("INITIAL")
    print(meta_reference)
    print(meta_reference.shape, meta_reference.is_contiguous(), meta_reference.dtype)

    metadata_list = meta_reference.tolist()

    # The probe pattern yields the value 68 in the metadata
    probe_pattern = [1, 1, 0, 0]
    for i in range(rows):
        num_per_tb = 8
        for j in range(cols // num_per_tb):
            dense_reference_tensor_c = dense_reference_tensor.clone()
            dense_reference_tensor_c[i, j * num_per_tb : (j + 1) * num_per_tb] = (
                torch.Tensor(probe_pattern)
                .to(device=device, dtype=dtype)
                .tile((1, 2))
                .contiguous()
            )
            # use the reference cutlass function to pack metadata
            _, meta_refernece_probe = to_sparse_semi_structured_cutlass_sm9x_f8(
                dense_reference_tensor_c
            )

            # find where the reference packed metadata is equal to 68
            indicies = (meta_refernece_probe == 68).nonzero()

            for r_i, c_i in indicies:
                metadata_list[r_i][c_i] = (
                    f"a[{i:2d}, {j * num_per_tb:2d}:{(j + 1) * num_per_tb:2d}]"
                )

    print("METADATA FORMAT")
    for line in metadata_list:
        print(line)
        print()


class SquaredReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


def profiler_runner(path, fn, *args, **kwargs):
    if path is None:
        path = os.path.join(
            os.path.expanduser("~/traces"),
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json.gz",
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    print(f"Exported trace to {path}")
    return result

    # input = create_semi_structured_tensor(4096, 8192, dtype=torch.bfloat16).to(device)
    # print(input)

    # ffn_clone = copy.deepcopy(test_ffn)
    # quantize_(ffn_clone.w1, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    # ffn_clone.w2 = FP8SemiSparseActivationLinear.from_dense(ffn_clone.w2)
    # # quantize_(ffn_clone.w2, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    # ffn_clone.forward = torch.compile(ffn_clone.forward, mode="max-autotune", fullgraph=True)
    # # warmup
    # def test():
    #     for i in range(10):
    #         ffn_clone(input)
    # test()
    # fp8_c_activation_sparse_time = benchmark_microseconds(test)
    # print(fp8_c_activation_sparse_time / 10)

    # profiler_runner(None, test)

    # test_linear = nn.Linear(8192, 8192).cuda().to(torch.bfloat16)
    # test_linear.weight.data = torch.ones(8192, 8192).cuda().to(torch.bfloat16)
    # print(test_linear(input))
    # sparse_fp8_linear = FP8SemiSparseActivationLinear.from_dense(test_linear)
    # print(sparse_fp8_linear(input))
