# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import re
import sys
from typing import List

BASE_COPYRIGHT_TEXT = """Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree."""

EXTENSIONS = {".py", ".cu", ".h", ".cuh", ".sh", ".metal"}
PRIVACY_PATTERNS = [
    r"Meta Platforms, Inc\. and affiliates",
    r"Facebook, Inc(\.|,)? and its affiliates",
    r"[0-9]{4}-present(\.|,)? Facebook",
    r"[0-9]{4}(\.|,)? Facebook",
]


def get_copyright_header(file_ext: str) -> str:
    if file_ext in {".cu", ".h", ".cuh", ".cpp", ".metal"}:
        # C/C++ style files use // comments
        return "\n".join(
            "// " + line if line else "//" for line in BASE_COPYRIGHT_TEXT.split("\n")
        )
    else:
        # Python and shell scripts use # comments
        return "\n".join(
            "# " + line if line else "#" for line in BASE_COPYRIGHT_TEXT.split("\n")
        )


def has_copyright_header(content: str) -> bool:
    # Check first 16 lines for privacy policy
    first_16_lines = "\n".join(content.split("\n")[:16])
    return any(re.search(pattern, first_16_lines) for pattern in PRIVACY_PATTERNS)


def add_copyright_header(filename: str) -> None:
    with open(filename, "r") as f:
        content = f.read()

    if not has_copyright_header(content):
        ext = os.path.splitext(filename)[1]
        header = get_copyright_header(ext)
        with open(filename, "w") as f:
            f.write(header + "\n\n" + content)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    retval = 0

    for filename in args.filenames:
        # Skip __init__.py files
        if os.path.basename(filename) == "__init__.py":
            continue

        ext = os.path.splitext(filename)[1]
        if ext in EXTENSIONS:
            with open(filename, "r") as f:
                content = f.read()

            if not has_copyright_header(content):
                print(f"Adding copyright header to {filename}")
                add_copyright_header(filename)
                retval = 1

    return retval


if __name__ == "__main__":
    sys.exit(main())
