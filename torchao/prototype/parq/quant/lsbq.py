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
        r -= v * binary_sign(r)
    return r


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

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize(), with b=0 for ternary"""
        assert b >= 0  # b==0 means ternary
        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        # b == 0 means ternary; b == 1 optimal same as greedy
        if b == 0:
            if self.optimal:
                q, Q = self.quantize_optimal_ternary(q)
            else:
                q, Q = self.quantize_simple_ternary(q, self.ternary_multiplier, dim=dim)
        elif b == 2 and self.optimal:
            q, Q = self.quantize_optimal_2bits(q)
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
            r -= v * binary_sign(r)
            vs.append(v)
        q = p - r

        # generate 2^b by b basis tensor B
        basis = list(itertools.product((-1, 1), repeat=b))
        B = torch.tensor(basis, dtype=p.dtype, device=p.device)
        if dim is not None:
            V = torch.concat(vs, dim=1)  # [dim0, b]
            Q = torch.sort(V @ B.T, dim=dim)[0]  # [dim0, 2^b]
        else:
            V = torch.tensor(vs, dtype=p.dtype, device=p.device)
            Q = torch.msort(B.matmul(V))  # [2^b]
        return q, Q

    @staticmethod
    def quantize_optimal_2bits(p: Tensor) -> tuple[Tensor, Tensor]:
        # first form the cumulative sum of sorted absolute values of p
        p_abs_sorted = torch.msort(torch.flatten(p.abs()))
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
        # generate 4 x 2 basis tensor B, sorted lexicographically along dim 0
        basis = list(itertools.product((-1, 1), repeat=2))
        B = torch.tensor(basis, dtype=p.dtype, device=p.device)
        # vmap workaround: calling torch.tensor on v1, v2 raises an error
        Q = v1 * B[:, 0] + v2 * B[:, 1]
        return q, Q

    @staticmethod
    def quantize_optimal_ternary(p: Tensor) -> tuple[Tensor, Tensor]:
        """Formula look reasonable, but derivation in reference incorrect?"""
        # first form the cumulative sum of sorted absolute values of p
        p_abs_sorted = torch.msort(torch.flatten(p.abs()))
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
