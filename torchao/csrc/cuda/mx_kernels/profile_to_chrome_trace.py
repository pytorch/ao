"""
Profile MX block rearrange kernel and export to Chrome trace format.

This script demonstrates how to:
1. Run the kernel with profiling enabled
2. Collect profiling data from GPU
3. Convert to Chrome trace JSON format
4. Visualize in chrome://tracing or Perfetto UI
"""

import json

import torch

from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mx_block_rearrange_2d_M_groups_cuda,
)

# Must match ProfilerTag enum in profiler.h
PROFILER_TAGS = [
    "TMALoad",  # Tag 0: Full TMA load time (issue -> wait complete)
    "ProcessThenTMAStore",  # Tag 1: Process + store time
]


def profile_mx_kernel(
    input_tensor,
    input_group_end_offsets,
    chunk_width=64,
    chunks_per_tb=4,
    num_profile_entries=10000,
):
    """
    Run the MX block rearrange kernel with profiling enabled.

    Args:
        input_tensor: Input scale factors (uint8 tensor)
        input_group_end_offsets: Group boundaries (int32 tensor)
        chunk_width: 64 or 128
        chunks_per_tb: 4, 8, or 16
        num_profile_entries: Max profiler entries per thread block

    Returns:
        output_tensor: Rearranged output tensor
        profiler_data: Raw profiler data from GPU (list of lists)
    """
    # Calculate number of thread blocks
    scale_rows = input_tensor.size(0)
    scale_cols = input_tensor.size(1)
    num_groups = input_group_end_offsets.size(0)
    rows_per_superblock = 128 * chunks_per_tb
    num_row_superblocks = (
        scale_rows + rows_per_superblock - 1
    ) // rows_per_superblock + num_groups
    num_col_chunks = (scale_cols + chunk_width - 1) // chunk_width
    num_blocks = num_col_chunks * num_row_superblocks

    # Allocate profiler buffer
    # Each block gets: 1 entry for count + num_profile_entries * 5 values per entry
    # Format: [count, sm_id, tag, chunk_idx, start, dur, ...]
    profiler = torch.zeros(
        num_blocks, 1 + num_profile_entries * 5, dtype=torch.int64, device="cuda"
    )

    output_tensor = mx_block_rearrange_2d_M_groups_cuda(
        input_tensor,
        input_group_end_offsets,
        chunk_width=chunk_width,
        chunks_per_tb=chunks_per_tb,
        profiler_tensor=profiler,
        num_profile_entries=num_profile_entries,
    )

    torch.cuda.synchronize()

    # Copy profiler data to host
    profiler_data = profiler.tolist()
    return output_tensor, profiler_data


def export_chrome_trace(
    profiler_data, output_path="mx_kernel_profile.json", verbose=False
):
    """Convert profiler data to Chrome trace format."""
    events = []

    for bid, data in enumerate(profiler_data):
        num_entries = data[0]
        if num_entries == 0:
            continue

        for i in range(num_entries):
            idx_start = 1 + i * 5
            sm_id, tag, chunk_idx, start_time, duration = data[
                idx_start : idx_start + 5
            ]

            # if tag < 0 or tag >= len(PROFILER_TAGS):
            #     continue

            # if duration == 0 and start_time == 0:
            #     continue

            base_tid = bid * 100 + chunk_idx * 10
            event = {
                "name": PROFILER_TAGS[tag],
                "ph": "X",
                "ts": start_time,
                "dur": duration,
                "pid": sm_id,
                "tid": base_tid + tag,
                "args": {
                    "block_id": bid,
                    "sm_id": sm_id,
                    "tag_id": tag,
                    "chunk_idx": chunk_idx,
                },
            }
            events.append(event)

    if not events:
        print("Warning: No profiling events recorded!")
        return

    # Normalize timestamps
    min_ts = min(evt["ts"] for evt in events)
    for evt in events:
        evt["ts"] = int(evt["ts"] - min_ts)
        evt["dur"] = int(evt["dur"])

    # Add metadata
    metadata_events = []
    unique_sms = sorted(set(evt["pid"] for evt in events))
    unique_blocks = sorted(set(evt["tid"] for evt in events))

    for sm_id in unique_sms:
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": sm_id,
                "args": {"name": f"SM {sm_id}"},
            }
        )

    for tid in unique_blocks:
        block_id = tid // 100
        chunk_idx = (tid % 100) // 10
        event_type_offset = tid % 10
        event_type_name = "TMALoad" if event_type_offset == 0 else "ProcessAndStore"

        block_events = [e for e in events if e["tid"] == tid]
        if block_events:
            sm_id = block_events[0]["pid"]
            metadata_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": sm_id,
                    "tid": tid,
                    "args": {
                        "name": f"Block {block_id} Chunk {chunk_idx} {event_type_name}"
                    },
                }
            )

    trace_data = {
        "traceEvents": metadata_events + events,
        "displayTimeUnit": "ns",
        "metadata": {
            "kernel": "mx_blocked_layout_2d_M_groups_kernel",
            "num_blocks": len(profiler_data),
            "num_events": len(events),
            "num_sms": len(unique_sms),
        },
    }

    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"Exported {len(events)} events to {output_path}")


if __name__ == "__main__":
    scale_rows = 2048
    scale_cols = 512
    num_groups = 4

    input_scales = torch.randint(
        0, 256, (scale_rows, scale_cols), dtype=torch.uint8, device="cuda"
    )

    # Create group boundaries (4 equal groups)
    group_size = scale_rows // num_groups
    input_group_end_offsets = torch.tensor(
        [group_size * (i + 1) for i in range(num_groups)],
        dtype=torch.int32,
        device="cuda",
    )

    print("Running kernel with profiling...")
    output_scales, profiler_data = profile_mx_kernel(
        input_scales, input_group_end_offsets, chunk_width=64, chunks_per_tb=4
    )

    print("Exporting to Chrome trace format...")
    export_chrome_trace(profiler_data, "mx_kernel_profile.json", verbose=False)
