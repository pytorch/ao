
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
