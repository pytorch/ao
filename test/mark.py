import os

# List of all test files
files = [
    "./dora/test_dora_fusion.py",
    "./dora/test_dora_layer.py",
    "./dtypes/test_affine_quantized.py",
    "./dtypes/test_nf4.py",
    "./dtypes/test_bitnet.py",
    "./dtypes/test_fp8.py",
    "./dtypes/test_uint2.py",
    "./dtypes/test_uint4.py",
    "./galore/README.md",
    "./galore/memory_analysis_utils.py",
    "./galore/model_configs.py",
    "./galore/profile_memory_usage.py",
    "./galore/profiling_utils.py",
    "./hqq/test_triton_mm.py",
    "./hqq/test_triton_qkv_fused.py",
    "./integration/test_integration.py",
    "./kernel/galore_test_utils.py",
    "./kernel/test_autotuner.py",
    "./kernel/test_fused_kernels.py",
    "./kernel/test_galore_downproj.py",
    "./prototype/mx_formats/test_custom_cast.py",
    "./prototype/mx_formats/test_mx_linear.py",
    "./prototype/mx_formats/test_mx_tensor.py",
    "./prototype/test_bitpacking_gen.py",
    "./prototype/test_quant_llm.py",
    "./prototype/test_bitpacking.py",
    "./prototype/test_low_bit_optim.py",
    "./quantization/test_galore_quant.py",
    "./quantization/test_qat.py",
    "./quantization/test_quant_api.py",
    "./quantization/test_quant_primitives.py",
    "./smoke_tests/smoke_tests.py",
    "./sparsity/test_parametrization.py",
    "./sparsity/test_scheduler.py",
    "./sparsity/test_sparsifier.py",
    "./sparsity/test_sparsity_utils.py",
    "./sparsity/test_structured_sparsifier.py",
    "./sparsity/test_wanda.py",
    "./sparsity/test_fast_sparse_training.py",
    "./sparsity/test_sparse_api.py",
    "./test_ops.py"
]

# Split the files into two groups
mid_point = len(files) // 2
group1 = files[:mid_point]
group2 = files[mid_point:]

# Function to add pytestmark to a file
def add_pytestmark(file_path, marker):
    if not file_path.endswith('.py'):
        return
    with open(file_path, 'r') as f:
        content = f.read()

    pytestmark_line = f'pytestmark = pytest.mark.{marker}\n'
    import_pytest_line = 'import pytest\n'

    if 'import pytest' not in content:
        content = import_pytest_line + pytestmark_line + content
    else:
        if 'pytestmark' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'import pytest' in line:
                    lines.insert(i + 1, pytestmark_line)
                    break
            content = '\n'.join(lines)
        else:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'pytestmark' in line:
                    lines[i] = pytestmark_line
                    break
            content = '\n'.join(lines)

    with open(file_path, 'w') as f:
        f.write(content)

# Add pytestmark to each group of files
for file in group1:
    add_pytestmark(file, 'group1')

for file in group2:
    add_pytestmark(file, 'group2')

print(f"Group 1 files: {group1}")
print(f"Group 2 files: {group2}")
