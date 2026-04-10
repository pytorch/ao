#!/usr/bin/env python3
"""
Parse TorchAO CI test logs and generate markdown summary.

This script parses pytest output from GitHub Actions job logs and produces
a readable summary in markdown table format.

Usage:
    python parse_ci_test_logs.py <logs_directory>
    python parse_ci_test_logs.py <single_log_file>
    
Examples:
    python parse_ci_test_logs.py acutal-logs/logs_54334293034/
    python parse_ci_test_logs.py acutal-logs/job-logs.txt
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    """Represents test results from a single CI job."""
    job_name: str
    pytorch_version: str
    backend: str  # "cuda" or "cpu"
    runner: str
    collected: int
    passed: int
    failed: int
    skipped: int
    warnings: int
    duration_seconds: float
    duration_human: str
    collection_skipped: int = 0  # Tests skipped during collection


def parse_job_name(filename: str) -> tuple[str, str, str]:
    """
    Extract PyTorch version, backend, and runner from job filename.
    
    Examples:
        "0_test (CUDA 2.8, linux.g5.12xlarge.nvidia.gpu, torch==2.8.0, cuda, 12.6) _ linux-job.txt"
        -> ("2.8", "cuda", "linux.g5.12xlarge.nvidia.gpu")
    """
    # Match patterns like "CUDA 2.8" or "CPU 2.7" or "CUDA Nightly" or "CPU Nightly"
    version_match = re.search(r'(CUDA|CPU)\s+([\d.]+|Nightly)', filename)
    if version_match:
        backend = version_match.group(1).lower()
        version = version_match.group(2)
    else:
        backend = "unknown"
        version = "unknown"
    
    # Match runner type
    runner_match = re.search(r'(linux\.[^,\)]+)', filename)
    runner = runner_match.group(1) if runner_match else "unknown"
    
    return version, backend, runner


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def parse_pytest_summary(line: str) -> Optional[dict]:
    """
    Parse pytest summary line.
    
    Examples:
        "======== 2542 passed, 6428 skipped, 694 warnings in 3764.96s (1:02:44) ========="
        "======== 2542 passed, 3 failed, 6428 skipped, 694 warnings in 3764.96s (1:02:44) ========="
    """
    # Strip ANSI color codes first
    line = strip_ansi_codes(line)
    
    # Pattern: passed, [failed,] skipped, warnings in Xs (H:MM:SS)
    pattern = r'=+\s+(\d+)\s+passed(?:,\s+(\d+)\s+failed)?(?:,\s+(\d+)\s+skipped)?(?:,\s+(\d+)\s+warnings)?\s+in\s+([\d.]+)s\s+\(([^)]+)\)'
    
    match = re.search(pattern, line)
    if match:
        return {
            'passed': int(match.group(1)),
            'failed': int(match.group(2)) if match.group(2) else 0,
            'skipped': int(match.group(3)) if match.group(3) else 0,
            'warnings': int(match.group(4)) if match.group(4) else 0,
            'duration_seconds': float(match.group(5)),
            'duration_human': match.group(6),
        }
    return None


def parse_collection_line(line: str) -> Optional[dict]:
    """
    Parse pytest collection summary line.
    
    Examples:
        "collected 8953 items / 17 skipped"
        "collected 5946 items"
    """
    # Strip ANSI color codes first
    line = strip_ansi_codes(line)
    
    pattern = r'collected\s+(\d+)\s+items(?:\s*/\s*(\d+)\s+skipped)?'
    match = re.search(pattern, line)
    if match:
        return {
            'collected': int(match.group(1)),
            'collection_skipped': int(match.group(2)) if match.group(2) else 0,
        }
    return None


def parse_log_file(filepath: Path) -> Optional[TestResult]:
    """Parse a single log file and extract test results."""
    
    filename = filepath.name
    version, backend, runner = parse_job_name(filename)
    
    collected = 0
    collection_skipped = 0
    summary = None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Check for collection line
                coll = parse_collection_line(line)
                if coll:
                    collected = coll['collected']
                    collection_skipped = coll['collection_skipped']
                
                # Check for summary line (take the last one)
                summ = parse_pytest_summary(line)
                if summ:
                    summary = summ
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None
    
    if summary is None:
        print(f"Warning: No pytest summary found in {filepath}", file=sys.stderr)
        return None
    
    # Create a cleaner job name
    job_name = f"{backend.upper()} {version}"
    if "nightly" in filename.lower():
        job_name = f"{backend.upper()} Nightly"
    
    return TestResult(
        job_name=job_name,
        pytorch_version=version,
        backend=backend,
        runner=runner,
        collected=collected,
        passed=summary['passed'],
        failed=summary['failed'],
        skipped=summary['skipped'],
        warnings=summary['warnings'],
        duration_seconds=summary['duration_seconds'],
        duration_human=summary['duration_human'],
        collection_skipped=collection_skipped,
    )


def parse_logs_directory(logs_dir: Path) -> list[TestResult]:
    """Parse all log files in a directory."""
    results = []
    seen_jobs = set()  # Track unique jobs to avoid duplicates
    
    # Find all .txt files (combined job logs) - prefer numbered files
    log_files = sorted(logs_dir.glob("[0-9]*_test*.txt"))
    
    if not log_files:
        # Try finding individual .txt files directly
        log_files = [f for f in logs_dir.glob("*.txt") if "test" in f.name.lower()]
    
    for log_file in sorted(log_files):
        result = parse_log_file(log_file)
        if result:
            # Create a unique key to avoid duplicates
            job_key = (result.pytorch_version, result.backend)
            if job_key not in seen_jobs:
                seen_jobs.add(job_key)
                results.append(result)
    
    return results


def generate_markdown_summary(results: list[TestResult], title: str = "CI Test Results Summary") -> str:
    """Generate markdown summary from test results."""
    
    lines = [
        f"# {title}",
        "",
        f"**Generated from CI logs**",
        "",
    ]
    
    # Summary statistics
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_collected = sum(r.collected for r in results)
    
    lines.extend([
        "## Overall Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Jobs | {len(results)} |",
        f"| Total Tests Collected | {total_collected:,} |",
        f"| Total Passed | {total_passed:,} |",
        f"| Total Failed | {total_failed:,} |",
        f"| Total Skipped | {total_skipped:,} |",
        "",
    ])
    
    # Detailed results table
    lines.extend([
        "## Detailed Results by Job",
        "",
        "| Job | Backend | PyTorch | Collected | Passed | Failed | Skipped | Pass Rate | Duration |",
        "|-----|---------|---------|-----------|--------|--------|---------|-----------|----------|",
    ])
    
    # Sort by backend (CUDA first) then by version
    def sort_key(r):
        version_num = 999 if r.pytorch_version == "Nightly" else float(r.pytorch_version)
        return (0 if r.backend == "cuda" else 1, version_num)
    
    for r in sorted(results, key=sort_key):
        executed = r.passed + r.failed
        pass_rate = (r.passed / executed * 100) if executed > 0 else 0
        fail_marker = "❌" if r.failed > 0 else "✅"
        
        lines.append(
            f"| {r.job_name} | {r.backend.upper()} | {r.pytorch_version} | "
            f"{r.collected:,} | {r.passed:,} | {r.failed} {fail_marker} | {r.skipped:,} | "
            f"{pass_rate:.1f}% | {r.duration_human} |"
        )
    
    lines.append("")
    
    # CUDA vs CPU comparison
    cuda_results = [r for r in results if r.backend == "cuda"]
    cpu_results = [r for r in results if r.backend == "cpu"]
    
    if cuda_results and cpu_results:
        lines.extend([
            "## CUDA vs CPU Comparison",
            "",
            "| Metric | CUDA (avg) | CPU (avg) | Difference |",
            "|--------|------------|-----------|------------|",
        ])
        
        cuda_avg_passed = sum(r.passed for r in cuda_results) / len(cuda_results)
        cpu_avg_passed = sum(r.passed for r in cpu_results) / len(cpu_results)
        cuda_avg_skipped = sum(r.skipped for r in cuda_results) / len(cuda_results)
        cpu_avg_skipped = sum(r.skipped for r in cpu_results) / len(cpu_results)
        
        lines.append(f"| Avg Passed | {cuda_avg_passed:,.0f} | {cpu_avg_passed:,.0f} | {cuda_avg_passed - cpu_avg_passed:+,.0f} |")
        lines.append(f"| Avg Skipped | {cuda_avg_skipped:,.0f} | {cpu_avg_skipped:,.0f} | {cuda_avg_skipped - cpu_avg_skipped:+,.0f} |")
        lines.append("")
    
    # Version comparison for CUDA
    if cuda_results:
        lines.extend([
            "## CUDA Results by PyTorch Version",
            "",
            "| PyTorch Version | Collected | Passed | Skipped | Pass Rate |",
            "|-----------------|-----------|--------|---------|-----------|",
        ])
        
        for r in sorted(cuda_results, key=lambda x: (999 if x.pytorch_version == "Nightly" else float(x.pytorch_version))):
            executed = r.passed + r.failed
            pass_rate = (r.passed / executed * 100) if executed > 0 else 0
            lines.append(f"| {r.pytorch_version} | {r.collected:,} | {r.passed:,} | {r.skipped:,} | {pass_rate:.1f}% |")
        
        lines.append("")
    
    # Version comparison for CPU
    if cpu_results:
        lines.extend([
            "## CPU Results by PyTorch Version",
            "",
            "| PyTorch Version | Collected | Passed | Skipped | Pass Rate |",
            "|-----------------|-----------|--------|---------|-----------|",
        ])
        
        for r in sorted(cpu_results, key=lambda x: (999 if x.pytorch_version == "Nightly" else float(x.pytorch_version))):
            executed = r.passed + r.failed
            pass_rate = (r.passed / executed * 100) if executed > 0 else 0
            lines.append(f"| {r.pytorch_version} | {r.collected:,} | {r.passed:,} | {r.skipped:,} | {pass_rate:.1f}% |")
        
        lines.append("")
    
    # Notes
    lines.extend([
        "## Notes",
        "",
        "- **Collected**: Total number of test items discovered by pytest",
        "- **Passed**: Tests that completed successfully",
        "- **Failed**: Tests that failed (assertion errors, exceptions)",
        "- **Skipped**: Tests skipped due to missing hardware, dependencies, or markers",
        "- **Pass Rate**: Passed / (Passed + Failed) - does not include skipped tests",
        "",
        "### Key Observations",
        "",
        "1. **CUDA vs CPU**: CUDA runs more tests because GPU-specific tests are not skipped",
        "2. **Skipped Tests**: High skip count on CPU is expected (GPU-only tests)",
        "3. **PyTorch Version**: Newer versions may have more tests due to new features",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse TorchAO CI test logs and generate markdown summary"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to logs directory or single log file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="CI Test Results Summary",
        help="Title for the markdown report"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        results = [parse_log_file(path)]
        results = [r for r in results if r is not None]
    elif path.is_dir():
        results = parse_logs_directory(path)
    else:
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    
    if not results:
        print("Error: No test results found", file=sys.stderr)
        sys.exit(1)
    
    markdown = generate_markdown_summary(results, title=args.title)
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(markdown)
        print(f"Report written to: {output_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
