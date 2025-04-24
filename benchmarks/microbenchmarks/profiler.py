# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
from torch.profiler import ProfilerActivity


def generate_model_profile(model, input_data, profile_file_path):
    """Function to benchmark model evaluation with profiling.

    Args:
        model: The model to profile
        input_data: Input data for the model
        profile_file_path: Path to save the profiler output

    Returns:
        profile_file_path
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(profile_file_path), exist_ok=True)

    # Set up profiler activities based on device
    activities = [ProfilerActivity.CPU]
    device = next(model.parameters()).device
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_data)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Run profiler with minimal settings to ensure compatibility
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,  # Experimental; might be unreliable for some layers
    ) as prof:
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # Save profiling details
    prof.export_chrome_trace(profile_file_path)
    print(f"Chrome trace saved at: {profile_file_path}")
    print("You can now visualize it using:")
    print("1. Chrome Trace Viewer: chrome://tracing")
    print("2. Perfetto UI: https://ui.perfetto.dev")

    return profile_file_path
