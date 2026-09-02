# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ruff_linter.yml"

# Python versions preinstalled in the ubuntu-latest (24.04) runner tool cache, see
# https://github.com/actions/runner-images/blob/main/images/ubuntu/toolsets/toolset-2404.json
# Asking setup-python for any other version makes the job download an interpreter
# through the GitHub API before it can lint, which is a network dependency the
# lint job does not need.
UBUNTU_LATEST_TOOLCACHE_PYTHONS = {"3.10", "3.11", "3.12", "3.13", "3.14"}


class TestRuffLinterWorkflow(unittest.TestCase):
    def test_python_version_is_preinstalled_on_runner(self):
        job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        for version in job["strategy"]["matrix"]["python-version"]:
            self.assertIn(version, UBUNTU_LATEST_TOOLCACHE_PYTHONS)


if __name__ == "__main__":
    unittest.main()
