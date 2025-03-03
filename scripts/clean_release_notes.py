# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================
#  This script automatically cleans up the raw release notes
#  generated by github by doing an initial pass to sort the
#  commits. The output still requires manual reviewing.
#
#  This script uses PyGithub. If you don't have it yet, please
#  install it using:
#
#    pip install PyGithub
#
#  We expect the following format for the input release notes:
#
#    ## What's Changed
#    * commit1_title by @userX in https://github.com/pytorch/ao/pull/123
#    * commit2_title by @userY in https://github.com/pytorch/ao/pull/234
#    * commit3_title by @userZ in https://github.com/pytorch/ao/pull/345
#
#    ## New Contributors
#    * @userX made their first contribution in https://github.com/pytorch/ao/pull/123
#    * @userY made their first contribution in https://github.com/pytorch/ao/pull/234
#
#  Example output:
#
#    ## Highlights
#
#    We are excited to announce the X.Y.Z release of torchao! This release adds support for A, B, C, D!
#
#    ### Highlight Feature 1
#
#    ### Highlight Feature 2
#
#    ## BC-Breaking
#
#    ## Deprecation
#
#    ## New Features
#    * commit1_title (https://github.com/pytorch/ao/pull/123)
#
#    ## Improvement
#    * commit2_title (https://github.com/pytorch/ao/pull/234)
#
#    ## Bug Fixes
#    * commit3_title (https://github.com/pytorch/ao/pull/345)
#
#    ## Performance
#
#    ## Documentation
#
#    ## Developers
#
#    ## New Contributors
#    * @userX made their first contribution in https://github.com/pytorch/ao/pull/123
#    * @userY made their first contribution in https://github.com/pytorch/ao/pull/234
#
# =============================================================


import os
import re
import sys
from typing import Dict, List, Optional

try:
    from github import Github
except ImportError as err:
    raise ValueError(
        "PyGithub not installed, please run 'pip install PyGithub'"
    ) from err

if len(sys.argv) != 2:
    print("Usage: python clean_release_notes.py [raw_release_notes.txt]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file + ".out"
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
GITHUB_LABEL_TO_CATEGORY = {
    "topic: bc-breaking": "BC Breaking",
    "topic: deprecation": "Deprecations",
    "topic: new feature": "New Features",
    "topic: improvement": "Improvement",
    "topic: bug fix": "Bug Fixes",
    "topic: performance": "Performance",
    "topic: documentation": "Documentation",
    "topic: for developer": "Developers",
}


def clean_release_notes():
    """
    Main entry point for this script.

    This function pre-processes the raw release notes and produces a template
    with all the standard sections and pre-sorts the commits into different
    categories based on github labels and commit title keywords.
    """

    # Write the header section
    with open(output_file, "w") as out_f:
        out_f.write("## Highlights\n\n")
        out_f.write(
            "We are excited to announce the X.Y.Z release of torchao! This release adds support for A, B, C, D!\n\n"
        )
        out_f.write("### Highlight Feature 1\n\n")
        out_f.write("### Highlight Feature 2\n\n")

    # Sort commits into different categories and write them to output file
    # For lines after the commits, just copy them to the output file as is
    commit_lines = []
    commit_start = False
    commits_by_category = {
        "BC Breaking": [],
        "Deprecations": [],
        "New Features": [],
        "Improvement": [],
        "Bug Fixes": [],
        "Performance": [],
        "Documentation": [],
        "Developers": [],
    }
    with open(input_file, "r") as in_f, open(output_file, "a") as out_f:
        for line in in_f.readlines():
            if line.startswith("## What's Changed"):
                commit_start = True
            elif commit_start and line.startswith("*"):
                commit_lines.append(line)
            elif commit_start:
                # End of commits, fetch PR labels based on commits collected so far
                commit_start = False
                pr_number_to_label = fetch_pr_labels(commit_lines)
                # Assign each commit to a category
                for commit_line in commit_lines:
                    category = get_commit_category(commit_line, pr_number_to_label)
                    if category is not None:
                        commits_by_category[category].append(commit_line)
                # Write all commits to the output file by category
                for category, commits in commits_by_category.items():
                    out_f.write("## %s\n\n" % category)
                    for commit_line in commits:
                        out_f.write(format_commit(commit_line))
                    out_f.write("\n")
            else:
                # Not a commit, just copy to the output file
                out_f.write(line)
    print("Wrote to %s." % output_file)


def parse_pr_number(commit_line: str) -> int:
    """
    Helper function to parse PR number from commit line.
    """
    return int(re.match(".*pytorch/ao/pull/(.*)", commit_line).groups()[0])


def fetch_pr_labels(commit_lines: List[str]) -> Dict[int, str]:
    """
    Fetch the relevant github labels starting with "topic: " from all PRs.
    If such a label exists for a given PR, store the first one.
    """
    pr_number_to_label = {}
    all_pr_numbers = [parse_pr_number(line) for line in commit_lines]
    smallest_pr_number = min(all_pr_numbers)
    repo = Github().get_repo("pytorch/ao")

    # This call fetches 30 PRs at a time in descending order of when the PR was created
    pulls = repo.get_pulls(state="closed")
    for pr in pulls:
        if pr.number < smallest_pr_number:
            break
        labels = [l.name for l in pr.labels if l.name.startswith("topic: ")]
        if len(labels) > 0:
            if VERBOSE:
                print("Found label for PR %s: '%s'" % (pr.number, labels[0]))
            pr_number_to_label[pr.number] = labels[0]
    return pr_number_to_label


def get_commit_category(
    commit_line: str, pr_number_to_label: Dict[int, str]
) -> Optional[str]:
    """
    Assign the commit to a category based on:
      (1) The github label if it exists
      (2) Keywords in the PR title

    If the commit is not meant to be user facing, remove None.
    Otherwise, return "Improvement" by default.
    """
    pr_number = parse_pr_number(commit_line)
    if pr_number in pr_number_to_label:
        label = pr_number_to_label[pr_number]
        if label == "topic: not user facing":
            return None
        if label in GITHUB_LABEL_TO_CATEGORY:
            return GITHUB_LABEL_TO_CATEGORY[label]
    elif any(x in commit_line.lower() for x in ["revert", "version.txt"]):
        return None
    elif any(
        x in commit_line.lower()
        for x in ["doc", "readme", "tutorial", "typo", "example", "spelling"]
    ):
        return "Documentation"
    elif any(x in commit_line.lower() for x in ["test", "lint", " ci", "nightl"]):
        return "Developers"
    elif " fix" in commit_line.lower():
        return "Bug Fixes"
    elif " add" in commit_line.lower():
        return "New Features"
    else:
        return "Improvement"


def format_commit(commit_line: str) -> str:
    """
    Format the commit line as follows:
      Before: * commit title by @userX in https://github.com/pytorch/ao/pull/123
      After:  * Commit title (https://github.com/pytorch/ao/pull/123)
    """
    # Remove author, put PR link in parentheses
    commit_line = re.sub(" by @.* in (.*)", r" (\\g<1>)", commit_line)
    # Capitalize first letter
    commit_line = commit_line.lstrip("* ")
    commit_line = "* " + commit_line[0].upper() + commit_line[1:]
    return commit_line


if __name__ == "__main__":
    clean_release_notes()
