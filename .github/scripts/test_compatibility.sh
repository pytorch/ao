# Testing compatibility
# We know that torchao .so files built using PyTorch 2.8.0 are not ABI compatible with PyTorch 2.9+. (see #2919)
# If the version of torch is not compatible with the version of torchao,
# we expect to skip loading the .so files and a warning should be logged but no error

# Function to check torchao import with configurable expectations
check_torchao_import() {
    local expect_warning="$1"
    local warning_text="$2"
    local torch_incompatible="${3:-}"

    if [ -n "$torch_incompatible" ]; then
        output=$(TORCH_INCOMPATIBLE=1 python -c "import torchao" 2>&1)
    else
        output=$(python -c "import torchao" 2>&1)
    fi
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Failed to import torchao"
        echo "Output: $output"
        exit 1
    fi

    warning_found=false
    if [ -n "$warning_text" ] && echo "$output" | grep -i "$warning_text" > /dev/null; then
        echo "Output: $output"
        warning_found=true
    fi

    if [ "$expect_warning" != "$warning_found" ]; then
        echo echo "FAILURE: expect_warning is $expect_warning but warning_found is $warning_found with message $output"
        exit 1
    fi
}

# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install compatible version of torch (nightly 2.9.0dev)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
# Installs nightly torchao (0.14.0dev...)
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu129
# Test 1: Should import successfully without warning
check_torchao_import "false" ""

# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install compatible version of torch (nightly 2.9.0dev)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
# Build torchao from source
python setup.py develop
# Test 2: Should import with warning because optional env var is set to true
check_torchao_import "true" "Skipping import of cpp extensions due to incompatible torch version" "TORCH_INCOMPATBILE=1"

# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install non-ABI stable torch version with current version of torchao
pip install torch==2.8.0
# Build torchao
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu129
# Test 4: Should import with specific warning
check_torchao_import "true" "Skipping import of cpp extensions due to incompatible torch version"
