# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor

from ..utils import channel_bucketize
from .quantizer import Quantizer


def binary_sign(input: Tensor) -> Tensor:
    """Same as `torch.sign(input)` but map 0 to 1."""
    return torch.where(input == 0, 1.0, input.sign())


def binary_quant_residue(u: Tensor, vs: Iterable[float]) -> Tensor:
    """Return residue for foldable binary quantization"""
    r = u.detach().clone()
    for v in vs:
        r.sub_(v * binary_sign(r))
    return r


def compute_v_per_channel(p: Tensor, dim: Optional[int] = None, ternary: bool = False):
    """Vectorized computation of optimal `v` for ternary/2-bit algorithm."""
    v_cands = p.abs().sort(dim=dim).values
    cumsum = v_cands.cumsum(dim=dim)
    cumsum, total_sum = cumsum[:, 1:-1], cumsum[:, -1:]

    # compute cumulative mean from right to left
    counts = torch.arange(1, p.size(dim=dim), device=p.device)
    counts_r2l = counts[:-1].flip((-1,))
    cmean_r2l = (total_sum - cumsum).div_(counts_r2l.mul_(2))
    v_cands, v_cands2 = v_cands[:, 1:-1], v_cands[:, 2:]

    # mask to estimate conditional expectation
    mask = (v_cands <= cmean_r2l).logical_and_(v_cands2 >= cmean_r2l)
    if ternary:
        # detect and fix any edge cases
        optimal_v = p.mean(dim=dim, keepdim=True).div_(2)
        row_invalid = optimal_v < p.min(dim=dim, keepdim=True).values
        if row_invalid.any():
            extra_col = row_invalid.to(p.dtype).mul(optimal_v)
            v_cands = torch.cat((v_cands, extra_col), -1)
            mask = torch.cat((mask, row_invalid), -1)
    else:
        # compute cumulative mean from left to right
        cmean_l2r = cumsum.div_(counts[1:].mul_(2)).add_(cmean_r2l)
        mask.logical_or_((v_cands <= cmean_l2r).logical_and_(v_cands2 >= cmean_l2r))

    # handle variable number of candidates per channel
    split_sizes = mask.sum(dim=dim).tolist()
    v_cands = v_cands[mask].split(split_sizes)
    v_cands = torch.nested.nested_tensor(list(v_cands))
    v_cands = torch.nested.to_padded_tensor(v_cands, 0.0)

    # update residual for each candidate `v`
    r = p.unsqueeze(dim - 1)
    v = v_cands.unsqueeze(-1)
    r = r.sub(v * binary_sign(r))
    if not ternary:
        v = v.mean(dim=dim, keepdim=True)
    r = r.sub(v * binary_sign(r))

    # compute least squares error, then select the `v` minimizes it
    costs = r.norm(dim=dim)
    indices = costs.argmin(dim=dim, keepdim=True)
    v_best = v_cands.gather(1, indices)
    return v_best


class LSBQuantizer(Quantizer):
    """Least-Square Binary Quantizer, using greedy algorithm by default.
    Optimal solution available for three cases: b=1, b=2 and ternary.
    """

    def __init__(
        self,
        center: bool = False,
        optimal: bool = False,
        ternary_multiplier: float = 1.1,
    ) -> None:
        """ternary_mult is factor multipled to 1-bit range as heuristic."""
        super().__init__(center)
        # optimal choice only meaningful for b=1, b=2 and ternary
        self.optimal = optimal
        self.ternary_multiplier = ternary_multiplier

    def get_quant_size(self, b: int) -> int:
        return 2**b if b > 0 else 3

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize(), with b=0 for ternary"""
        if b < 0:
            raise ValueError(f"Invalid {b=}; must be nonnegative")
        if self.optimal and b > 2:
            raise NotImplementedError(f"Unsupported {self.optimal=} for {b=}")

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        if self.optimal and b != 1:  # b == 1 optimal is the same as greedy
            if b == 0:
                q, Q = self.quantize_optimal_ternary(q, dim=dim)
            elif b == 2:
                q, Q = self.quantize_optimal_2bits(q, dim=dim)
        elif b == 0:
            q, Q = self.quantize_simple_ternary(q, self.ternary_multiplier, dim=dim)
        else:
            q, Q = self.quantize_greedy(q, b, dim=dim)

        # return quantized tensor and set of quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q

    @staticmethod
    def quantize_greedy(
        p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        r = p.detach().clone()
        vs = []
        keepdim = dim is not None
        for _ in range(b):
            v = r.abs().mean(dim=dim, keepdim=keepdim)
            r.sub_(binary_sign(r).mul_(v))
            vs.append(v)
        q = p - r

        # generate 2^b by b basis tensor B
        basis = list(itertools.product((-1, 1), repeat=b))
        B = torch.tensor(basis, dtype=p.dtype, device=p.device)
        if dim is not None:
            V = torch.concat(vs, dim=1)  # [dim0, b]
            Q = torch.sort(V @ B.T, dim=dim).values  # [dim0, 2^b]
        else:
            V = torch.tensor(vs, dtype=p.dtype, device=p.device)
            Q = torch.msort(B.matmul(V))  # [2^b]
        return q, Q

    @staticmethod
    def quantize_optimal_2bits(
        p: Tensor, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        # generate 4 x 2 basis tensor B, sorted lexicographically along dim 0
        basis = list(itertools.product((-1, 1), repeat=2))
        B = torch.tensor(basis, dtype=p.dtype, device=p.device)
        if dim is not None:
            v1 = compute_v_per_channel(p, dim=dim, ternary=False)
            s = binary_sign(p).mul_(v1)
            r = p.sub(s)
            v2 = r.abs().mean(dim=dim, keepdim=True)
            q = s.add_(binary_sign(r).mul_(v2))

            V = torch.cat((v1, v2), dim=-1)  # [dim0, b]
            Q = V @ B.T  # [dim0, 2^b]
            return q, Q

        # first form the cumulative sum of sorted absolute values of p
        p_abs_sorted = p.abs().flatten().sort().values
        cumsum = torch.cumsum(p_abs_sorted, dim=0)
        n = cumsum.numel()
        # find all solutions v1 to an inclusion problem (after sorting |p|)
        # |p|_{i} <= v1 < |p|_{i+1}
        # where v1 = (1/2) * (avg_{0:i}(|p|) + avg_{i+1:n-1}(|p|))
        V1V2 = []
        v1 = cumsum[-1] / n / 2
        v2 = torch.zeros_like(v1)
        q = torch.zeros_like(p)
        # check if v1 < |p|.min(), which means v1 == v2, effectively ternary
        if v1 < p_abs_sorted[0]:
            V1V2.append((v1, v1))
        for i in range(n - 1):
            E_above = (cumsum[-1] - cumsum[i]) / (n - (i + 1))
            E_below = cumsum[i] / (i + 1)
            # skip if E_above < E_below, implying v2 < 0, invalid solution
            if E_above < E_below:
                continue
            v1 = (E_above + E_below) / 2
            if v1 >= p_abs_sorted[i] and v1 < p_abs_sorted[i + 1]:
                v2 = (E_above - E_below) / 2
                assert v1 >= v2, "LSBQ 2-bit optimal: v1 should >= v2."
                V1V2.append((v1, v2))
        assert len(V1V2) > 0, "LSBQ 2-bit optimal: No solution found."
        # find the best solution with least-square quantization error
        min_error = p.norm()
        for v1v2 in V1V2:
            r = binary_quant_residue(p, v1v2)
            error = r.norm()
            if error < min_error:
                min_error = error
                q = p - r
                v1, v2 = v1v2

        V = torch.tensor((v1, v2), dtype=p.dtype, device=p.device)
        Q = B @ V
        return q, Q

    @staticmethod
    def quantize_optimal_ternary(
        p: Tensor, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Formula look reasonable, but derivation in reference incorrect?"""
        if dim is not None:
            v = compute_v_per_channel(p, dim=dim, ternary=True)
            p_sign = binary_sign(p)
            r = p.sub(p_sign.mul(v))

            # 0 if sign(p) != sign(r), else sign(p) * 2v
            q = p_sign.add_(binary_sign(r)).mul_(v)

            # each channel can take values [-2v, 0, 2v]
            v.mul_(2)
            Q = torch.cat((-v, torch.zeros_like(v), v), dim=-1)  # [dim0, 3]
            return q, Q

        # first form the cumulative sum of sorted absolute values of p
        p_abs_sorted = p.abs().flatten().sort().values
        cumsum = torch.cumsum(p_abs_sorted, dim=0)
        n = cumsum.numel()
        # find all solutions v1 to an inclusion problem (after sorting |p|)
        # |p|_{i} <= v < |p|_{i+1} where v = (1/2) * avg_{i+1:n-1}(|p|))
        v_feasible = []
        v = cumsum[-1] / n / 2
        # check if v < |p|.min(), which means v1 == v2, effectively ternary
        if v < p_abs_sorted[0]:
            v_feasible.append(v)
        for i in range(n - 1):
            v = ((cumsum[-1] - cumsum[i]) / (n - (i + 1))) / 2
            if v >= p_abs_sorted[i] and v < p_abs_sorted[i + 1]:
                v_feasible.append(v)
        assert len(v_feasible) > 0, "LSBQ ternary optimal: No solution found."
        # find the best solution with least-square quantization error
        min_error = p.norm()
        q_best = torch.zeros_like(p)
        v_best = torch.zeros_like(v)
        for v in v_feasible:
            Q = v * torch.tensor([-1.0, 0.0, 1.0], device=p.device)
            boundaries = v * torch.tensor([-0.5, 0.5], device=p.device)
            q = Q[torch.bucketize(p, boundaries)]
            error = torch.linalg.norm(p - q)
            if error < min_error:
                min_error = error
                q_best = q
                v_best = v
        Q = v_best * torch.tensor([-1, 0, 1], dtype=p.dtype, device=p.device)
        return q_best, Q

    @staticmethod
    def quantize_simple_ternary(
        p: Tensor, multiplier: float, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Heuristic by setting v as given multiple of |p|.abs().mean()"""
        v = multiplier * p.abs().mean(dim=dim, keepdim=dim is not None)
        Q = v * torch.tensor([-1.0, 0.0, 1.0], device=p.device)
        if dim is None:
            boundaries = v * torch.tensor([-0.5, 0.5], device=p.device)
            q = Q[torch.bucketize(p, boundaries)]
        else:
            # for each row, find the element-wise index of least upper bound
            boundaries = v * torch.tensor([[-0.5, 0.5]], device=p.device)
            q = Q.gather(1, channel_bucketize(p, boundaries))
        return q, Q
