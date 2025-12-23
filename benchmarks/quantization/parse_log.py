#!/usr/bin/env python3
"""
Parse measure_accuracy_and_performance.sh log file and extract metrics.

Usage:
    python parse_log.py <log_file> [--format csv|json|table]

Example:
    python parse_log.py benchmarks/data/measure_accuracy_and_performance_log.txt --format csv
"""

import argparse
import re
import sys
from typing import Dict, List, Optional

from tabulate import tabulate


def parse_header(lines: List[str]) -> Dict[str, str]:
    """Parse the first 4 lines to extract library versions."""
    header = {}
    for line in lines[:4]:
        # Format: torch.__version__='2.9.0+cu128'
        # or: torch.cuda.get_device_name()='NVIDIA H100'
        match = re.match(r"([^=]+)='([^']+)'", line)
        if match:
            key = match.group(1)
            value = match.group(2)
            header[key] = value
    return header


def parse_accuracy_metrics(section_lines: List[str]) -> Dict[str, Optional[float]]:
    """Parse accuracy metrics from lm_eval output table."""
    metrics = {
        "wikitext_bits_per_byte": None,
        "winogrande_acc": None,
        "winogrande_acc_stderr": None,
    }

    for line in section_lines:
        # Parse wikitext metrics
        # Format: |wikitext  |      2|none  |     0|bits_per_byte  |↓  |0.5452|±  |   N/A|
        if "|wikitext" in line and "|bits_per_byte" in line:
            parts = [p.strip() for p in line.split("|")]
            # Find the value (should be after bits_per_byte)
            try:
                idx = parts.index("bits_per_byte")
                if idx + 2 < len(parts):
                    metrics["wikitext_bits_per_byte"] = float(parts[idx + 2])
            except (ValueError, IndexError):
                pass

        # Parse winogrande accuracy
        # Format: |winogrande|      1|none  |     0|acc            |↑  |0.7419|±  |0.0123|
        if "|winogrande" in line and "|acc" in line:
            parts = [p.strip() for p in line.split("|")]
            try:
                idx = parts.index("acc")
                if idx + 2 < len(parts):
                    metrics["winogrande_acc"] = float(parts[idx + 2])
                if idx + 4 < len(parts) and parts[idx + 4] != "N/A":
                    metrics["winogrande_acc_stderr"] = float(parts[idx + 4])
            except (ValueError, IndexError):
                pass

    return metrics


def parse_throughput_metrics(
    section_lines: List[str],
) -> Dict[str, Optional[Dict[str, float]]]:
    """Parse vLLM throughput metrics."""
    metrics = {
        "prefill": None,
        "decode": None,
    }

    # Find throughput lines
    for i, line in enumerate(section_lines):
        # Format: Throughput: 7.50 requests/s, 30939.86 total tokens/s, 239.84 output tokens/s
        if line.startswith("Throughput:"):
            match = re.match(
                r"Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+total tokens/s,\s+([\d.]+)\s+output tokens/s",
                line,
            )
            if match:
                throughput_data = {
                    "requests_per_sec": float(match.group(1)),
                    "total_tokens_per_sec": float(match.group(2)),
                    "output_tokens_per_sec": float(match.group(3)),
                }

                # Determine if this is prefill or decode by looking backwards for the benchmark command
                # Prefill has --input_len 4096 --output_len 32
                # Decode has --input_len 32 --output_len 2048
                for j in range(max(0, i - 50), i):
                    if "benchmarking vllm prefill performance" in section_lines[j]:
                        metrics["prefill"] = throughput_data
                        break
                    elif "benchmarking vllm decode performance" in section_lines[j]:
                        metrics["decode"] = throughput_data
                        break
                else:
                    # If we can't find the marker, assign based on order
                    if metrics["prefill"] is None:
                        metrics["prefill"] = throughput_data
                    elif metrics["decode"] is None:
                        metrics["decode"] = throughput_data

    return metrics


def parse_recipe_section(section_lines: List[str], recipe_name: str) -> Dict:
    """Parse a single recipe section."""
    result = {
        "recipe": recipe_name,
    }

    # Parse checkpoint size
    checkpoint_size_gb = None
    for line in section_lines:
        # Format: checkpoint size: 16.077915292 GB
        match = re.match(r"checkpoint size:\s+([\d.]+)\s+GB", line)
        if match:
            checkpoint_size_gb = float(match.group(1))
            break
    result["checkpoint_size_gb"] = checkpoint_size_gb

    # Parse accuracy metrics
    accuracy = parse_accuracy_metrics(section_lines)
    result.update(accuracy)

    # Parse throughput metrics
    throughput = parse_throughput_metrics(section_lines)

    # Flatten throughput metrics
    if throughput["prefill"]:
        result["prefill_total_tokens_per_sec"] = throughput["prefill"][
            "total_tokens_per_sec"
        ]
    else:
        result["prefill_total_tokens_per_sec"] = None

    if throughput["decode"]:
        result["decode_total_tokens_per_sec"] = throughput["decode"][
            "total_tokens_per_sec"
        ]
    else:
        result["decode_total_tokens_per_sec"] = None

    return result


def parse_log_file(log_file_path: str) -> Dict:
    """Parse the entire log file."""
    with open(log_file_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # Parse header
    header = parse_header(lines)

    # Find recipe sections
    recipe_sections = []
    current_section = None
    current_recipe = None

    for i, line in enumerate(lines):
        # Look for recipe markers
        match = re.match(r"processing quant_recipe (.+)", line)
        if match:
            # Save previous section if exists
            if current_section is not None:
                recipe_sections.append((current_recipe, current_section))

            # Start new section
            current_recipe = match.group(1).strip()
            current_section = []
        elif current_section is not None:
            current_section.append(line)

    # Don't forget the last section
    if current_section is not None:
        recipe_sections.append((current_recipe, current_section))

    # Parse each recipe section
    results = []
    for recipe_name, section_lines in recipe_sections:
        result = parse_recipe_section(section_lines, recipe_name)
        results.append(result)

    # Calculate speedups relative to baseline ("None")
    baseline = None
    for result in results:
        if result["recipe"] == "None":
            baseline = result
            break

    if baseline:
        baseline_prefill = baseline["prefill_total_tokens_per_sec"]
        baseline_decode = baseline["decode_total_tokens_per_sec"]

        for result in results:
            # Calculate prefill speedup
            if (
                result["prefill_total_tokens_per_sec"] is not None
                and baseline_prefill is not None
            ):
                result["speedup_prefill"] = (
                    result["prefill_total_tokens_per_sec"] / baseline_prefill
                )
            else:
                result["speedup_prefill"] = None

            # Calculate decode speedup
            if (
                result["decode_total_tokens_per_sec"] is not None
                and baseline_decode is not None
            ):
                result["speedup_decode"] = (
                    result["decode_total_tokens_per_sec"] / baseline_decode
                )
            else:
                result["speedup_decode"] = None
    else:
        # No baseline found, set all speedups to None
        for result in results:
            result["speedup_prefill"] = None
            result["speedup_decode"] = None

    return {
        "header": header,
        "results": results,
    }


def format_as_csv(data: Dict) -> str:
    """Format parsed data as CSV."""
    lines = []

    # Header comment with library versions
    lines.append("# Library Versions:")
    for key, value in data["header"].items():
        lines.append(f"# {key}={value}")
    lines.append("")

    # CSV header
    fieldnames = [
        "recipe",
        "checkpoint_size_gb",
        "wikitext_bits_per_byte",
        "winogrande_acc",
        "winogrande_acc_stderr",
        "prefill_total_tokens_per_sec",
        "decode_total_tokens_per_sec",
        "speedup_prefill",
        "speedup_decode",
    ]
    lines.append(",".join(fieldnames))

    # Data rows
    for result in data["results"]:
        row = [
            str(result.get(field, "")) if result.get(field) is not None else ""
            for field in fieldnames
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def format_as_table(data: Dict) -> str:
    """Format parsed data as a human-readable table using tabulate."""
    lines = []

    # Header with library versions
    lines.append("Library Versions:")
    lines.append("=" * 80)
    for key, value in data["header"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    # Prepare table data
    table_data = []
    headers = [
        "Recipe",
        "Checkpoint\n(GB)",
        "Wikitext\nbits/byte",
        "Winogrande\nAcc",
        "Winogrande\nStderr",
        "Prefill\ntoks/s",
        "Decode\ntoks/s",
        "Speedup\nPrefill",
        "Speedup\nDecode",
    ]

    for result in data["results"]:
        row = [
            result["recipe"],
            f"{result['checkpoint_size_gb']:.2f}"
            if result["checkpoint_size_gb"] is not None
            else None,
            f"{result['wikitext_bits_per_byte']:.4f}"
            if result["wikitext_bits_per_byte"] is not None
            else None,
            f"{result['winogrande_acc']:.4f}"
            if result["winogrande_acc"] is not None
            else None,
            f"{result['winogrande_acc_stderr']:.4f}"
            if result["winogrande_acc_stderr"] is not None
            else None,
            f"{result['prefill_total_tokens_per_sec']:.2f}"
            if result["prefill_total_tokens_per_sec"] is not None
            else None,
            f"{result['decode_total_tokens_per_sec']:.2f}"
            if result["decode_total_tokens_per_sec"] is not None
            else None,
            f"{result['speedup_prefill']:.3f}"
            if result["speedup_prefill"] is not None
            else None,
            f"{result['speedup_decode']:.3f}"
            if result["speedup_decode"] is not None
            else None,
        ]
        table_data.append(row)

    # Generate table
    lines.append("Quantization Recipe Results:")
    lines.append("=" * 80)
    lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse measure_accuracy_and_performance.sh log file"
    )
    parser.add_argument("log_file", help="Path to the log file to parse")
    parser.add_argument(
        "--format",
        choices=["csv", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Parse the log file
    data = parse_log_file(args.log_file)

    # Format output
    if args.format == "csv":
        output = format_as_csv(data)
    else:  # table
        output = format_as_table(data)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
