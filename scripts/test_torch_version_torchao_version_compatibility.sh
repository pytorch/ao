# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Testing compatibility
# We know that torchao .so files built using PyTorch 2.8.0 are not ABI compatible with PyTorch 2.9+. (see #2919)
# If the version of torch is not compatible with the version of torchao,
# we expect to skip loading the .so files and a warning should be logged but no error

PREV_TORCH_VERSION = 2.8.0
PREV_TORCHAO_VERSION = 0.13.0

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

## prev torch version, prev torchao version
# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install prev compatible version of torch
pip install PREV_TORCH_VERSION
# Installs prev compatible version of torchao
pip install PREV_TORCHAO_VERSION
# hould import successfully without warning
check_torchao_import "false" ""

## current torch, current torchao
# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install specific compatible version of torch (nightly 2.9.0dev)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
# Build torchao from source
python setup.py develop
# Should import successfully without warning
check_torchao_import "false" ""
## prev torch, torchao from source (do not rebuild), env var = True
# Uninstall torch
pip uninstall torch
# Install incompatible version of torch
pip install torch==PREV_TORCH_VERSION
# Should import with warning because optional env var is set to true
check_torchao_import "true" "Skipping import of cpp extensions due to incompatible torch version" "TORCHAO_SKIP_LOADING_SO_FILES=1"


# current torch, prev torchao
# Uninstall torch
pip uninstall torch
# Uninstall torchao
pip uninstall torchao
# Install non-ABI stable torch version
pip install torch==2.9.0
# Installs incompatible torchao
pip install torchao==PREV_TORCHAO_VERSION
# Should import with specific warning
check_torchao_import "true" "Skipping import of cpp extensions due to incompatible torch version"
