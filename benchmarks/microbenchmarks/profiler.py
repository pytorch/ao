# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import os
import pickle

import torch
from torch.profiler import ProfilerActivity


def _validate_pickle_file(file_path):
    """Validate if the pickle file is valid and can be read."""
    try:
        with open(file_path, "rb") as f:
            pickle.load(f)
    except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
        print(f"Error: Pickle file {file_path} is invalid or cannot be read. {e}")
        return False
    return True


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


def generate_memory_profile(model, input_data, profile_file_path):
    """Function to generate CUDA memory profile.

    Args:
        model: The model to profile
        input_data: Input data for the model
        profile_file_path: Path to save the memory profile (.pickle)

    Returns:
        str: Path to the saved profile file.
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Memory profiling requires CUDA.")
        return None
    if model is None or input_data is None:
        raise ValueError("Model and input_data must not be None.")

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(profile_file_path), exist_ok=True)
    memory_stats = dict()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Reset memory history to ensure clean slate
        torch.cuda.memory._record_memory_history(enabled=False)
        torch.cuda.memory._record_memory_history(max_entries=100000)

        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
                torch.cuda.synchronize()

        for i in range(5):
            try:
                # Reset again to avoid warm-up effects in final stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.memory._record_memory_history(enabled=False)
                torch.cuda.memory._record_memory_history(max_entries=100000)

                # Run actual profiled inference
                with torch.no_grad():
                    _ = model(input_data)
                    torch.cuda.synchronize()

                # Take memory snapshot after inference and save to temporary pickle file
                torch.cuda.memory._dump_snapshot(profile_file_path)

                if _validate_pickle_file(profile_file_path):
                    print(f"Saved memory profile to {profile_file_path}")
                    break
            except ValueError as e:
                import time

                print(f"Attempt {i + 1}/5: {e}, retrying...")
                time.sleep(3.0)

        # Record memory stats
        _memory_stats = torch.cuda.memory_stats()
        memory_stats = {
            "allocated_bytes.all.peak": _memory_stats["allocated_bytes.all.peak"] / 1e6,
            "active_bytes.all.peak": _memory_stats["active_bytes.all.peak"] / 1e6,
            "reserved_bytes.all.peak": _memory_stats["reserved_bytes.all.peak"] / 1e6,
        }

    except Exception as e:
        print(f"Error in memory profiling: {e}")

    # Return the file path for consistency with other profiler functions
    return profile_file_path, memory_stats


def visualize_memory_profile(profile_file_path):
    """Visualize memory profile using matplotlib.

    Args:
        profile_file_path: Path to the memory profile file (.pickle)

    Returns:
        str: Path to the visualization HTML file
    """
    # Create parent directory if it doesn't exist
    memory_visualization_path = profile_file_path.replace("pickle", "html")
    os.makedirs(os.path.dirname(memory_visualization_path), exist_ok=True)
    try:
        # For pickle files (from actual profiling), use torch's visualization
        from torch.cuda._memory_viz import trace_plot

        with open(profile_file_path, "rb") as f:
            data = pickle.load(f)
        with open(memory_visualization_path, "w") as f:
            f.write(trace_plot(data))

        print(f"Memory visualization saved to: {memory_visualization_path}")

    except Exception as e:
        print(
            f"Error in generating visualization: {e}\n",
            "To view the memory visualization, upload the pickle file to https://pytorch.org/memory_viz or run the following command to convert that to a html file:\n",
            "python pytorch/torch/cuda/_memory_viz.py trace_plot <pickle file> -o <desired output name>.html",
        )

    return memory_visualization_path
