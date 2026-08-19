"""Grouped per-expert global amax over dense NVFP4 weights.

Input-side twin of ``triton_group_weight_quantize_2d`` (group_quantize_2d_triton.py),
which consumes exactly the ``(E,)`` float32 amax this op produces. Unlike the activation
amax the weight stack is dense and uniform -- no ragged offsets, no RHT -- so this is a
flat 1D reduction per expert rather than a tiled one.

Replaces ``torch.linalg.vector_norm(W, ord=inf, dim=(1, 2))``, which computes the same
values but keeps too few loads in flight to saturate HBM (~2.5 TB/s vs ~3.9 TB/s here).
The unrolled ``U`` independent loads per program are what buy the difference.

Nothing here is expert-specific beyond the ``program_id(1)`` base, so ``nvfp4_linear``
uses it at ``E=1`` on ``W.unsqueeze(0)`` for the same reason.
"""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.10.0") and has_triton():
    import triton
    import triton.language as tl

    # BLOCK x U is the bytes-in-flight knob: more independent loads before the max
    # dependency chain is what lifts this above the TensorIterator reduce. Sweeping
    # both (rather than fixing BLOCK*U) matters because the winner moves with size --
    # 2048x4 leads at 117 MB, 4096x2 at 63 MB. Straight-line body, so num_stages has
    # nothing to pipeline and is left at the default.
    _GROUP_WEIGHT_AMAX_CONFIGS: list[triton.Config] = [
        triton.Config({"BLOCK": block, "U": u}, num_warps=nw)
        for block in (2048, 4096, 8192)
        for u in (1, 2, 4)
        for nw in (4, 8)
    ]

    # The output buffer starts at zero and repeated atomic_max launches are idempotent,
    # so autotuning does not need reset_to_zero.
    @triton.autotune(configs=_GROUP_WEIGHT_AMAX_CONFIGS, key=["numel_per_expert"])
    @triton.jit
    def _group_weight_amax_kernel(
        a_ptr,
        global_amax_ptr,
        numel_per_expert,
        BLOCK: tl.constexpr,
        U: tl.constexpr,
    ):
        """Per-expert amax -- one flat BLOCK*U chunk per CTA, one atomic per CTA."""
        # int64 expert base: E * M * N exceeds int32 at realistic expert counts,
        # silently wrapping the load to a bad address.
        expert = tl.program_id(1).to(tl.int64)
        base = expert * numel_per_expert
        start = tl.program_id(0).to(tl.int64) * (BLOCK * U)

        cumulative = tl.zeros((BLOCK,), dtype=tl.float32)
        for u in tl.static_range(U):
            offsets = start + u * BLOCK + tl.arange(0, BLOCK)
            a = tl.load(
                a_ptr + base + offsets,
                mask=offsets < numel_per_expert,
                other=0.0,
            )
            cumulative = tl.maximum(
                cumulative,
                tl.abs(a.to(tl.float32)),
                propagate_nan=tl.PropagateNan.ALL,
            )

        # tl.max does not propagate NaN, so re-inject it after the reduction.
        amax = tl.max(cumulative, axis=0)
        has_nan = tl.max((cumulative != cumulative).to(tl.int32), axis=0)
        amax = tl.where(has_nan != 0, float("nan"), amax)
        tl.atomic_max(global_amax_ptr + expert, amax)

    @torch.library.custom_op("torchao::triton_group_weight_amax", mutates_args=())
    def triton_group_weight_amax(A: torch.Tensor, num_tensors: int) -> torch.Tensor:
        """Per-expert global absolute maximum of a dense expert weight stack.

        Args:
            A: Dense ``(E, M, N)`` BF16 weights, contiguous. Each expert is a
                contiguous 2D matrix; no divisibility constraint on M or N.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            ``(E,)`` float32, where ``out[e] = A[e].float().abs().amax()``. NaN in an
            expert propagates to that expert's entry. Bit-exact with the PyTorch
            reduction, and directly consumable by ``triton_group_weight_quantize_2d``.
        """
        if A.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16, got {A.dtype}")
        if A.ndim != 3:
            raise ValueError("Tensor A must be 3-D")
        if not A.is_contiguous():
            raise ValueError("A must be contiguous")

        E, M, N = A.shape
        if E != num_tensors:
            raise ValueError(f"Expected {num_tensors} experts, got {E}")

        global_amax = torch.zeros((E,), dtype=torch.float32, device=A.device)
        numel_per_expert = M * N
        grid = lambda meta: (
            triton.cdiv(numel_per_expert, meta["BLOCK"] * meta["U"]),
            E,
        )
        _group_weight_amax_kernel[grid](A, global_amax, numel_per_expert)
        return global_amax

    @triton_group_weight_amax.register_fake
    def _(A, num_tensors):
        return A.new_empty((A.shape[0],), dtype=torch.float32)

else:

    def triton_group_weight_amax(A: torch.Tensor, num_tensors: int) -> torch.Tensor:
        raise NotImplementedError(
            "triton_group_weight_amax requires torch 2.10.0+ and Triton"
        )
