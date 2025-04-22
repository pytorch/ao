import torch
import torch.nn.functional as F

def dump_metadata_format_cutlass():
    r, c = 128, 256
    # 238 in binary
    W_ref_asdf = torch.Tensor([0, 0, 1, 1]).to(device=device, dtype=torch.float8_e4m3fn).tile((r, c // 4)).contiguous()
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref_asdf)
    # W_quant_func = _float8_cutlass_quant_sparse
    # W_aqt = W_quant_func(W_ref_asdf, dtypeq_W)
    # W_meta = W_aqt.tensor_impl.meta
    print("INITIAL")
    print(meta_reference)
    print(meta_reference.shape, meta_reference.is_contiguous(), meta_reference.dtype)
    breakpoint()
    garbanzo_beans = meta_reference.tolist()


    pattern = [1, 1, 0, 0] # 68
    for i in range(r):
        num_per_tb = 8
        for j in range(c // num_per_tb):
            W_ref = W_ref_asdf.clone()
            W_ref[i, j*num_per_tb:(j+1)*num_per_tb] = torch.Tensor(pattern).to(device=device, dtype=dtype).tile((1, 2)).contiguous()
            _, W_meta = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)

            indicies = (W_meta == 68).nonzero()

            # print(indicies, i, j, W_meta)
            # breakpoint()

            for (r_i, c_i) in indicies:
                garbanzo_beans[r_i][c_i] = f"a[{i:2d}, {j*num_per_tb:2d}:{(j+1)*num_per_tb:2d}]"

    # from pprint import pprint
    print("METADATA FORMAT")
    for line in garbanzo_beans:
        print(line)
        print()
        # print(line[:4])
        # print(line[4:])

    assert False
    # torch.testing.assert_close(W_meta, W_subclass_sparse.meta.view(torch.uint8))

class SquaredReLU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


def profiler_runner(path, fn, *args, **kwargs):
    if path is None:
        path = os.path.join(
            os.path.expanduser("~/traces"),
            f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json.gz',
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
