# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch


def parse_version(version_string):
    """
    Parse version string representing pre-release with -1.

    Examples:
    - "2.5.0" -> [2, 5, 0]
    - "2.5.0.dev" -> [2, 5, -1]
    """
    # Remove local version identifier (everything after +)
    clean_version = re.sub(r"\+.*$", "", version_string)

    # Check for pre-release indicators (including all common patterns)
    is_prerelease = bool(re.search(r"(a|b|dev)", clean_version))

    match = re.match(r"(\d+)\.(\d+)\.(\d+)", clean_version)
    if match:
        major, minor, patch = map(int, match.groups())
        if is_prerelease:
            patch = -1
        return [major, minor, patch]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")


def is_fbcode():
    return not hasattr(torch.version, "git_version")


def torch_version_at_least(min_version):
    if is_fbcode():
        return True

    # Parser for local identifiers
    return parse_version(torch.__version__) >= parse_version(min_version)


# Test cases
if __name__ == "__main__":
    test_cases = [
        ("2.5.0+cu126", [2, 5, 0]),
        ("2.5.0", [2, 5, 0]),
        ("2.5.0a0+git9f17037", [2, 5, -1]),
        ("2.5.0.dev20240708+cu121", [2, 5, -1]),
        ("2.4.0", [2, 4, 0]),
        ("2.2.0beta1", [2, 2, -1]),
    ]

    print("Testing parse_version:")
    for version_str, expected in test_cases:
        result = parse_version(version_str)
        status = "✓" if result == expected else "✗"
        print(f"{status} {version_str} -> {result} (expected: {expected})")
