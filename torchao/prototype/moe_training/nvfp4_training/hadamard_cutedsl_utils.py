# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime availability checks for the CuteDSL NVFP4 RHT kernels."""

import importlib.util

import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

# Import name -> pip package name for the required runtime packages.
_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list:
    """Return the list of missing CuteDSL runtime packages (empty if all present)."""
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            # ModuleNotFoundError: parent module absent (e.g. 'cuda' on CPU)
            # ValueError: malformed module name
            spec = None
        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """True iff the cuda-python + nvidia-cutlass-dsl + apache-tvm-ffi packages are importable."""
    return len(_missing_cutedsl_runtime_packages()) == 0


def cutedsl_nvfp4_kernels_available() -> bool:
    """True iff the CuteDSL NVFP4 RHT kernels can run here.

    Requires a Blackwell (SM 10.x) GPU, CUDA 12.8+, and the cuda-python +
    nvidia-cutlass-dsl + apache-tvm-ffi runtime packages.
    """
    return (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and is_cuda_version_at_least(12, 8)
        and _cutedsl_runtime_available()
    )


# Full requirement list, for user-facing error messages.
CUTEDSL_NVFP4_REQUIREMENTS = (
    "a Blackwell (SM100) GPU, CUDA 12.8+, and the cuda-python, nvidia-cutlass-dsl, "
    "and apache-tvm-ffi packages"
)


def cutedsl_nvfp4_unavailable_reason() -> str:
    """Reason the CuteDSL NVFP4 kernels are unavailable: the unmet hardware, CUDA, or
    package requirement(s). Returns ``""`` when everything is available."""
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("no CUDA device")
    elif not is_sm_at_least_100():
        reasons.append("requires a Blackwell (SM 10.x) GPU")
    if not is_cuda_version_at_least(12, 8):
        reasons.append("requires CUDA 12.8+")
    missing = _missing_cutedsl_runtime_packages()
    if missing:
        reasons.append("missing packages: " + ", ".join(sorted(missing)))
    return "; ".join(reasons)


def raise_if_cutedsl_nvfp4_unavailable(op_name: str) -> None:
    """Raise ``NotImplementedError`` when the CuteDSL NVFP4 kernels cannot run here."""
    if not cutedsl_nvfp4_kernels_available():
        raise NotImplementedError(
            f"{op_name} requires {CUTEDSL_NVFP4_REQUIREMENTS} "
            f"({cutedsl_nvfp4_unavailable_reason()})."
        )


def cutedsl_prepare_for_cuda_graph(device, *, sign_vectors=None) -> None:
    """Pre-allocate per-device CuteDSL state before ``torch.compile(mode="reduce-overhead")``.

    The CuteDSL ops cache small per-device buffers (the RHT / identity Hadamard operands and the
    stochastic-rounding RNG seed) and compile their kernels lazily on first use. Under CUDA-graph
    capture, anything first allocated *inside* the captured region lands in the cudagraph memory
    pool and is rejected as an untracked live allocation. Calling this once per device (passing the
    RHT sign vectors the graph will use) forces those allocations + compiles into the normal pool,
    before capture. Mirrors the Triton ``prepare_for_cuda_graph``; a no-op if CuteDSL is unavailable.
    """
    if not cutedsl_nvfp4_kernels_available():
        return
    from ._cutedsl_kernels_impl import (
        _compile_amax_tc_kernel,
        _compile_fused_kernel,
        _get_identity_buffer,
        _get_rht_buffer,
        _get_sr_rng_buffer,
    )

    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    _get_identity_buffer(idx)
    _get_sr_rng_buffer(idx)
    for sign_vector in sign_vectors or ():
        _get_rht_buffer(tuple(int(v) for v in sign_vector), idx)
    # Pre-compile the kernels so no lazy compile fires mid-capture: amax; the RHT fused RTNE +
    # SR variants (apply_rht=True); and the no-MMA weight-quantize variant (apply_rht=False).
    _compile_amax_tc_kernel(idx)
    for sr in (False, True):
        _compile_fused_kernel(idx, True, sr, apply_rht=True)
    _compile_fused_kernel(idx, True, False, apply_rht=False)
