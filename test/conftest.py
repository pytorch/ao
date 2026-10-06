# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


def pytest_sessionfinish(session, exitstatus):
    # Workaround for https://github.com/pytorch/pytorch/pull/180283, which made
    # the cuBLAS/cuBLASLt workspace maps thread_local (previously an intentionally
    # leaked static whose destructor never ran). As a result, the maps now get a
    # real destructor that frees CUDA workspace memory at process exit -- after the
    # CUDA context may already be torn down -- which aborts the interpreter with
    # SIGABRT (exit 134) once all tests have already passed. This first shipped in
    # the torch nightly dev20260722 and is what makes the CUDA regression legs die
    # at shutdown.
    #
    # torch.cuda._clear_cublas_workspaces() empties those maps. Calling it here, in
    # pytest's session-finish hook, runs while the CUDA context is still alive, so
    # the maps are empty by the time their thread_local destructors run and those
    # destructors become no-ops.
    #
    # This is scoped to test/ (rather than the repo root) because every pytest
    # invocation in CI targets a path under test/, so pytest loads this conftest for
    # all of them; the only pytest run outside test/ is the MPS metal job, which has
    # no CUDA and does not need this.
    import torch

    if torch.cuda.is_available() and hasattr(torch.cuda, "_clear_cublas_workspaces"):
        torch.cuda._clear_cublas_workspaces()
