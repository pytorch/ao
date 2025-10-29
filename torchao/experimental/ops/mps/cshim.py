import torch

# List of ops and their c-shim declarations used for AOTInductor
# Check out TestUIntxWeightOnlyLinearQuantizer.test_export_accuracy on how to use it
torchao_op_c_shim: dict[torch.ops.OpOverload, list[str]] = {}

for nbit in range(1, 8):
    op_name = f"_linear_fp_act_{nbit}bit_weight"
    torchao_op_c_shim[getattr(torch.ops.torchao, op_name).default] = [
        f"AOTITorchError aoti_torch_mps_{op_name}(AtenTensorHandle A, AtenTensorHandle B, int64_t group_size, AtenTensorHandle S, AtenTensorHandle Z, AtenTensorHandle* ret)",
    ]
