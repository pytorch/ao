import torch
import triton
import triton.language as tl


@triton.heuristics(
    values={
        "USE_MASK": lambda args: args["numels"] % args["BLOCK_SIZE"] != 0,
        "NUM_GROUPS": lambda args: triton.cdiv(args["numels"], args["BLOCK_SIZE"]),
    }
)
@triton.jit
def _quantize_blockwise_kernel(
    t_ptr,
    cutoffs_ptr,
    q_ptr,
    absmax_ptr,
    norm_ptr,
    numels,
    BLOCK_SIZE: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    USE_MASK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    RETURN_NORM: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = None
    absmax_mask = None
    if USE_MASK:
        mask = offsets < numels
        absmax_mask = pid < NUM_GROUPS
    t = tl.load(t_ptr + offsets, mask=mask)

    absmax = tl.max(tl.abs(t), axis=0)
    normalized = t / absmax

    # Load code buckets
    cutoffs = tl.load(cutoffs_ptr + tl.arange(0, NUM_BUCKETS))
    q = tl.reshape(normalized, (BLOCK_SIZE, 1)) > cutoffs

    # NOTE: explicit cast is needed, addition on tl.int1 (bool) does not work as per torch / numpy
    q = q.to(tl.uint8)
    q = tl.sum(q, axis=1)

    tl.store(q_ptr + offsets, q, mask=mask)
    # Each block processes one group_size number of elements, hence 1 absmax
    tl.store(absmax_ptr + pid, absmax, mask=absmax_mask)

    if RETURN_NORM:
        tl.store(norm_ptr + offsets, normalized, mask=mask)


# NOTE: Each block processes one group_size number of elements, hence BLOCK_SIZE = group_size
# where group_size corresponds to the groupwise quantization blocksize
def triton_quantize_blockwise(
    t: torch.Tensor, code, group_size=2048, return_normalized=False
):
    """
    Params:
        t: torch.Tensor, tensor to quantize
        code: torch.Tensor, quantization codebook for bitsandbytes, output of `bitsandbytes.functional.create_dynamic_map`
        # absmax: torch.Tensor, absolute max values for each block, if None, will be calculated from the input tensor
        group_size: int, groupwise quantization blocksize, default 2048, the hardcoded blocksize for bitsandbytes 8-bit optimizers
        return_normalized: bool, if True, will return the normalized tensor, primarily for debugging
    """
    numel = t.numel()
    q = torch.empty(numel, dtype=torch.uint8, device=t.device)
    normalized = torch.empty_like(t) if return_normalized else None
    num_groups = numel // group_size
    abs_max = torch.empty(num_groups, dtype=t.dtype, device="cuda")
    # Cutoffs for quantization
    # code corresponds to actual (normalized) quant codes
    # Cutoffs are used to calculate which of these codes a value belongs to
    # E.g., for consecutive codes C1 and C2, the corresponding cutoff is C1 + C2 / 2
    # Hence, if a value is greater is assigned C1 if it is less than all cutoffs up to this cutoff
    cutoffs = (code[:-1] + code[1:]) / 2

    # Need to make cutoffs multiple of 2 for triton reduce
    MAX_CUTOFF = torch.tensor(
        torch.finfo(cutoffs.dtype).max, dtype=cutoffs.dtype, device=cutoffs.device
    ).reshape(
        1,
    )
    cutoffs = torch.cat([cutoffs, MAX_CUTOFF], dim=-1)
    assert cutoffs.numel() % 2 == 0

    grid = lambda META: (triton.cdiv(t.numel(), META["BLOCK_SIZE"]),)
    # assert t.numel() % group_size == 0
    _quantize_blockwise_kernel[grid](
        t.view(-1),
        cutoffs,
        q,
        abs_max,
        normalized.view(-1) if return_normalized else None,
        numel,
        NUM_BUCKETS=len(cutoffs),
        BLOCK_SIZE=group_size,
        RETURN_NORM=return_normalized,
    )
    return (
        q.reshape(t.shape),
        normalized.reshape(t.shape) if return_normalized else None,
        abs_max,
    )


# Reference implementation
def _torch_quantize_blockwise(tensor: torch.Tensor, code, absmax=None, blocksize=2048):
    # Flatten values first

    # If not absmax, need to first normalize -> reshape to (-1, blocksize) -> max over the last dim

    # Quantize by flattening A to [numels, 1] > code[:, None], sum, then reshape back to original shape
    if absmax is None:
        absmax = tensor.reshape(-1, blocksize).abs().max(dim=-1).values

    normalized = tensor.reshape(-1, blocksize) / absmax[:, None]
    buckets = (code[:-1] + code[1:]) / 2
    q = normalized.reshape(normalized.numel(), 1) > buckets
    q = q.sum(dim=1).reshape(tensor.shape)
    return q.to(torch.uint8), normalized.reshape(tensor.shape), absmax
