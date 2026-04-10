#!/usr/bin/env python3
"""
Parse TorchAO CI test logs and generate detailed test results.

This script parses pytest output from GitHub Actions job logs and produces
a detailed report of individual tests with their pass/fail/skip status,
grouped by test module/file.

Usage:
    python parse_ci_detailed_tests.py <log_file> [options]
    
Examples:
    # Parse CUDA 2.9 log and output detailed test list
    python parse_ci_detailed_tests.py "acutal-logs/logs_54334293034/2_test (CUDA 2.9, ...).txt"
    
    # Save to markdown file
    python parse_ci_detailed_tests.py <log_file> -o detailed-tests.md
    
    # Filter by status (passed, failed, skipped)
    python parse_ci_detailed_tests.py <log_file> --status passed
    
    # Filter by module
    python parse_ci_detailed_tests.py <log_file> --module quantization
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestCase:
    """Represents a single test case result."""
    full_name: str  # Full pytest node id
    module: str     # Test module/file path (e.g., test/dtypes/test_nf4.py)
    class_name: Optional[str]  # Test class name if any
    test_name: str  # Test function name
    parameters: Optional[str]  # Parametrized test parameters
    status: str     # PASSED, FAILED, SKIPPED
    failure_reason: Optional[str] = None  # For failed tests


@dataclass 
class TestModule:
    """Represents tests grouped by module."""
    name: str
    tests: list = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    
    def add_test(self, test: TestCase):
        self.tests.append(test)
        if test.status == "PASSED":
            self.passed += 1
        elif test.status == "FAILED":
            self.failed += 1
        elif test.status == "SKIPPED":
            self.skipped += 1


@dataclass
class DetailedTestResults:
    """Container for all parsed test results."""
    job_name: str
    pytorch_version: str
    backend: str
    total_collected: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    modules: dict = field(default_factory=dict)  # module_name -> TestModule
    
    def add_test(self, test: TestCase):
        if test.module not in self.modules:
            self.modules[test.module] = TestModule(name=test.module)
        self.modules[test.module].add_test(test)
        
        if test.status == "PASSED":
            self.total_passed += 1
        elif test.status == "FAILED":
            self.total_failed += 1
        elif test.status == "SKIPPED":
            self.total_skipped += 1


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def parse_job_info(filename: str) -> tuple[str, str, str]:
    """
    Extract job name, PyTorch version, and backend from job filename.
    
    Examples:
        "2_test (CUDA 2.9, linux.g5.12xlarge.nvidia.gpu, torch==2.9.1, cuda, 12.6) _ linux-job.txt"
        -> ("CUDA 2.9", "2.9", "cuda")
    """
    version_match = re.search(r'(CUDA|CPU)\s+([\d.]+|Nightly)', filename)
    if version_match:
        backend = version_match.group(1).lower()
        version = version_match.group(2)
        job_name = f"{version_match.group(1)} {version}"
    else:
        backend = "unknown"
        version = "unknown"
        job_name = "Unknown Job"
    
    return job_name, version, backend


def parse_test_line(line: str) -> Optional[TestCase]:
    """
    Parse a single test result line from pytest verbose output.
    
    Examples (with timestamp prefix from CI logs):
        "2026-01-14T08:32:25.7524826Z test/core/test_config.py::test_name[config0] PASSED"
        "2026-01-14T08:32:25.7524826Z test/dtypes/test_nf4.py::TestClass::test_name SKIPPED"
        
    Also handles wrapped lines where status appears alone:
        "2026-01-14T08:32:34.8524766Z PASSED"
    """
    line = strip_ansi_codes(line).strip()
    
    # Remove timestamp prefix if present (GitHub Actions log format)
    # Format: 2026-01-14T08:32:25.7524826Z 
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+'
    line = re.sub(timestamp_pattern, '', line)
    
    # Pattern: test_path::[[class::]test_name[params]] STATUS
    # Status can be PASSED, FAILED, SKIPPED (sometimes with reason after)
    pattern = r'^(test/[^\s]+::[^\s]+)\s+(PASSED|FAILED|SKIPPED)(?:\s*-.*)?$'
    
    match = re.match(pattern, line)
    if not match:
        return None
    
    full_name = match.group(1)
    status = match.group(2)
    
    # Parse the test path components
    # Format: test/path/file.py::ClassName::test_name[params]
    # or:     test/path/file.py::test_name[params]
    
    parts = full_name.split("::")
    module = parts[0]  # test/path/file.py
    
    class_name = None
    test_name = None
    parameters = None
    
    if len(parts) == 2:
        # No class: test/file.py::test_name[params]
        test_part = parts[1]
    elif len(parts) >= 3:
        # With class: test/file.py::ClassName::test_name[params]
        class_name = parts[1]
        test_part = parts[2]
    else:
        test_part = parts[-1] if parts else ""
    
    # Extract parameters from test name
    param_match = re.match(r'^([^\[]+)(?:\[(.+)\])?$', test_part)
    if param_match:
        test_name = param_match.group(1)
        parameters = param_match.group(2)
    else:
        test_name = test_part
    
    return TestCase(
        full_name=full_name,
        module=module,
        class_name=class_name,
        test_name=test_name,
        parameters=parameters,
        status=status,
    )


def parse_collection_line(line: str) -> Optional[int]:
    """Parse pytest collection summary to get total collected count."""
    line = strip_ansi_codes(line)
    # Remove timestamp prefix if present
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+'
    line = re.sub(timestamp_pattern, '', line)
    pattern = r'collected\s+(\d+)\s+items'
    match = re.search(pattern, line)
    return int(match.group(1)) if match else None


def clean_line(line: str) -> str:
    """Clean a log line by removing ANSI codes and timestamp."""
    line = strip_ansi_codes(line).strip()
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+'
    return re.sub(timestamp_pattern, '', line)


def parse_wrapped_test_line(prev_line: str, curr_line: str) -> Optional[TestCase]:
    """
    Try to parse a test result that spans two lines.
    
    Some long test names wrap, e.g.:
        Line 1: "test/path/file.py::ClassName::test_very_long_name_with_params[param1-param2-param"
        Line 2: "3] PASSED"
    """
    prev_clean = clean_line(prev_line)
    curr_clean = clean_line(curr_line)
    
    # Check if current line is just a status (possibly with trailing part of test name)
    status_only_pattern = r'^([^\s]*)\s*(PASSED|FAILED|SKIPPED)$'
    status_match = re.match(status_only_pattern, curr_clean)
    
    if not status_match:
        return None
    
    trailing_part = status_match.group(1)  # Could be empty or end of test name
    status = status_match.group(2)
    
    # Check if previous line starts with a test path but doesn't have a status
    if not prev_clean.startswith('test/'):
        return None
    
    # Previous line shouldn't already have a status
    if re.search(r'\s+(PASSED|FAILED|SKIPPED)\s*$', prev_clean):
        return None
    
    # Combine the lines
    full_test_line = prev_clean + trailing_part + " " + status
    
    # Now try to parse it
    pattern = r'^(test/[^\s]+::[^\s]+)\s+(PASSED|FAILED|SKIPPED)$'
    match = re.match(pattern, full_test_line)
    
    if not match:
        return None
    
    full_name = match.group(1)
    status = match.group(2)
    
    # Parse the test path components
    parts = full_name.split("::")
    module = parts[0]
    
    class_name = None
    test_name = None
    parameters = None
    
    if len(parts) == 2:
        test_part = parts[1]
    elif len(parts) >= 3:
        class_name = parts[1]
        test_part = parts[2]
    else:
        test_part = parts[-1] if parts else ""
    
    param_match = re.match(r'^([^\[]+)(?:\[(.+)\])?$', test_part)
    if param_match:
        test_name = param_match.group(1)
        parameters = param_match.group(2)
    else:
        test_name = test_part
    
    return TestCase(
        full_name=full_name,
        module=module,
        class_name=class_name,
        test_name=test_name,
        parameters=parameters,
        status=status,
    )


def parse_log_file(filepath: Path) -> Optional[DetailedTestResults]:
    """Parse a single log file and extract detailed test results."""
    
    filename = filepath.name
    job_name, version, backend = parse_job_info(filename)
    
    results = DetailedTestResults(
        job_name=job_name,
        pytorch_version=version,
        backend=backend,
    )
    
    prev_line = ""
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Check for collection line
                collected = parse_collection_line(line)
                if collected:
                    results.total_collected = collected
                    prev_line = line
                    continue
                
                # Check for test result line (single line)
                test = parse_test_line(line)
                if test:
                    results.add_test(test)
                    prev_line = line
                    continue
                
                # Check for wrapped test result (two lines)
                test = parse_wrapped_test_line(prev_line, line)
                if test:
                    results.add_test(test)
                
                prev_line = line
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None
    
    if not results.modules:
        print(f"Warning: No test results found in {filepath}", file=sys.stderr)
        return None
    
    return results


def generate_summary_table(results: DetailedTestResults) -> str:
    """Generate a summary statistics table."""
    parsed_total = results.total_passed + results.total_failed + results.total_skipped
    
    lines = [
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total Collected | {results.total_collected:,} |",
        f"| **Parsed Tests** | {parsed_total:,} |",
        f"| Passed | {results.total_passed:,} |",
        f"| Failed | {results.total_failed:,} |",
        f"| Skipped | {results.total_skipped:,} |",
        f"| Test Modules | {len(results.modules)} |",
        "",
    ]
    
    # Add note if we couldn't parse all tests
    if parsed_total < results.total_collected:
        not_parsed = results.total_collected - parsed_total
        lines.extend([
            f"> **Note**: {not_parsed:,} tests could not be parsed due to multi-line test output.",
            "> This happens when test names span multiple lines or produce verbose output.",
            "",
        ])
    
    return "\n".join(lines)


def generate_modules_summary(
    results: DetailedTestResults,
    module_filter: Optional[str] = None,
) -> str:
    """Generate a summary table by module."""
    lines = [
        "## Results by Module",
        "",
        "| Module | Passed | Failed | Skipped | Total |",
        "|--------|--------|--------|---------|-------|",
    ]
    
    # Sort modules by name
    for module_name in sorted(results.modules.keys()):
        # Apply module filter
        if module_filter and module_filter.lower() not in module_name.lower():
            continue
            
        module = results.modules[module_name]
        total = module.passed + module.failed + module.skipped
        
        # Add visual indicators
        failed_str = f"**{module.failed}** ❌" if module.failed > 0 else str(module.failed)
        passed_str = f"{module.passed} ✅" if module.passed > 0 and module.failed == 0 else str(module.passed)
        
        lines.append(
            f"| {module_name} | {passed_str} | {failed_str} | {module.skipped} | {total} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def generate_detailed_test_list(
    results: DetailedTestResults, 
    status_filter: Optional[str] = None,
    module_filter: Optional[str] = None,
    show_params: bool = True,
) -> str:
    """Generate detailed test list grouped by module."""
    lines = ["## Detailed Test List", ""]
    
    # Sort modules by name
    for module_name in sorted(results.modules.keys()):
        # Apply module filter
        if module_filter and module_filter.lower() not in module_name.lower():
            continue
        
        module = results.modules[module_name]
        
        # Filter tests by status if specified
        if status_filter:
            tests = [t for t in module.tests if t.status.upper() == status_filter.upper()]
        else:
            tests = module.tests
        
        if not tests:
            continue
        
        # Module header with stats
        lines.append(f"### {module_name}")
        lines.append(f"*{module.passed} passed, {module.failed} failed, {module.skipped} skipped*")
        lines.append("")
        
        # Group tests by class (if any)
        by_class = defaultdict(list)
        for test in tests:
            key = test.class_name or "(module-level)"
            by_class[key].append(test)
        
        for class_name in sorted(by_class.keys()):
            class_tests = by_class[class_name]
            
            if class_name != "(module-level)":
                lines.append(f"#### `{class_name}`")
                lines.append("")
            
            # Create test list table
            lines.append("| Test | Status |")
            lines.append("|------|--------|")
            
            for test in sorted(class_tests, key=lambda t: t.test_name):
                # Format test name
                if show_params and test.parameters:
                    test_display = f"`{test.test_name}[{test.parameters}]`"
                else:
                    test_display = f"`{test.test_name}`"
                
                # Format status with emoji
                if test.status == "PASSED":
                    status_display = "✅ PASSED"
                elif test.status == "FAILED":
                    status_display = "❌ FAILED"
                else:
                    status_display = "⏭️ SKIPPED"
                
                lines.append(f"| {test_display} | {status_display} |")
            
            lines.append("")
    
    return "\n".join(lines)


def generate_markdown_report(
    results: DetailedTestResults,
    title: Optional[str] = None,
    status_filter: Optional[str] = None,
    module_filter: Optional[str] = None,
    summary_only: bool = False,
) -> str:
    """Generate complete markdown report."""
    
    if title is None:
        title = f"Detailed Test Results: {results.job_name}"
    
    lines = [
        f"# {title}",
        "",
        f"**Job**: {results.job_name}  ",
        f"**PyTorch Version**: {results.pytorch_version}  ",
        f"**Backend**: {results.backend.upper()}  ",
        "",
    ]
    
    # Add filter info if applied
    filters = []
    if status_filter:
        filters.append(f"Status: {status_filter.upper()}")
    if module_filter:
        filters.append(f"Module contains: '{module_filter}'")
    if filters:
        lines.append(f"**Filters applied**: {', '.join(filters)}")
        lines.append("")
    
    # Summary statistics
    lines.append(generate_summary_table(results))
    
    # Module summary
    lines.append(generate_modules_summary(results, module_filter=module_filter))
    
    # Detailed test list (unless summary_only)
    if not summary_only:
        lines.append(generate_detailed_test_list(
            results, 
            status_filter=status_filter,
            module_filter=module_filter,
        ))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse TorchAO CI test logs and generate detailed test results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Parse CUDA 2.9 log
    python parse_ci_detailed_tests.py "logs/2_test (CUDA 2.9, ...).txt"
    
    # Show only failed tests
    python parse_ci_detailed_tests.py <log_file> --status failed
    
    # Show only tests from quantization module
    python parse_ci_detailed_tests.py <log_file> --module quantization
    
    # Save to file
    python parse_ci_detailed_tests.py <log_file> -o results.md
"""
    )
    
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the CI log file to parse"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        help="Custom title for the report"
    )
    parser.add_argument(
        "-s", "--status",
        type=str,
        choices=["passed", "failed", "skipped"],
        help="Filter tests by status"
    )
    parser.add_argument(
        "-m", "--module",
        type=str,
        help="Filter tests by module name (substring match)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, not detailed test list"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.log_file.exists():
        print(f"Error: File not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse log file
    results = parse_log_file(args.log_file)
    if results is None:
        print("Failed to parse log file", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    report = generate_markdown_report(
        results,
        title=args.title,
        status_filter=args.status,
        module_filter=args.module,
        summary_only=args.summary_only,
    )
    
    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
