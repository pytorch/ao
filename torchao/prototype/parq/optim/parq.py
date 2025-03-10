import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from ..utils import channel_bucketize
from .proxmap import ProxMap


def amp_custom_fwd(cast_inputs: Optional[torch.types._dtype] = None):
    try:
        return partial(
            torch.amp.custom_fwd, device_type="cuda", cast_inputs=cast_inputs
        )
    except AttributeError:
        return partial(torch.cuda.amp.custom_fwd, cast_inputs=cast_inputs)


def normalized_mirror_sigmoid(t: float, t1: float, t2: float, s: float) -> float:
    """Sigmoid-like function decreasing from 1 to 0 over interval [t1, t2).
    s is steepness of the sigmoid-like function, almost linear for s < 1.
    'mirror' means decreasing instead of increasing as true sigmoid,
    'normalized' means value 1 at starting point t1 and 0 at end point t2."""
    assert t >= t1 and t < t2, "Normalized sigmoid: ensure t1 <= t < t2"
    ft = (t - t1) / (t2 - t1)  # fraction of progress from t1 to t2
    st = 1 / (1 + math.exp(s * (ft - 0.5)))  # scaled and shifted mirror sigmoid
    s1 = 1 / (1 + math.exp(-0.5 * s))  # st value when t = t1 -> ft = 0
    s2 = 1 / (1 + math.exp(0.5 * s))  # st value when t = t2 -> ft = 1
    return (st - s2) / (s1 - s2)  # shift and scale to range (0, 1]


class ProxPARQ(ProxMap):
    def __init__(
        self, anneal_start: int, anneal_end: int, steepness: float = 10
    ) -> None:
        assert anneal_start < anneal_end, "PARQ annealing: start before end."
        assert steepness > 0, "PARQ annealing steepness should be positive."
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.steepness = steepness

    @torch.no_grad()
    @amp_custom_fwd(cast_inputs=torch.float32)
    def apply_(
        self,
        p: Tensor,
        q: Tensor,
        Q: Tensor,
        step_count: int,
        dim: Optional[int] = None,
    ) -> float:
        """Prox-map of PARQ with gradual annealing to hard quantization."""

        if step_count < self.anneal_start:
            inv_slope = 1.0
        elif step_count >= self.anneal_end:
            inv_slope = 0.0
            if q is None:
                # hard quantization to the nearest point in Q
                Q_mid = (Q[..., :-1] + Q[..., 1:]) / 2
                if dim is None:
                    q = Q[torch.bucketize(p, Q_mid)]
                else:
                    q = Q.gather(1, channel_bucketize(p, Q_mid))
            p.copy_(q)
        else:
            inv_slope = normalized_mirror_sigmoid(
                step_count, self.anneal_start, self.anneal_end, self.steepness
            )
            # it is important to clamp idx-1 and then clamping idx itself
            # idx_1[k] == idx[k] iff p[k] > Q.max() or p[k] <= Q.min()
            if dim is None:
                idx = torch.bucketize(p, Q)  # locate quant interval
                idx_lower = (idx - 1).clamp_(min=0)  # index of lower bound
                idx_upper = idx.clamp(max=Q.numel() - 1)  # index of upper bound
                q_lower = Q[idx_lower]  # lower boundary of interval
                q_upper = Q[idx_upper]  # upper boundary of interval
                center = (q_lower + q_upper) / 2  # center of interval
                # concise implementation of piecewise-affine prox map
                q = (center + (p - center) / inv_slope).clamp_(min=q_lower, max=q_upper)
            else:
                idx = channel_bucketize(p, Q)
                idx_lower = (idx - 1).clamp_(min=0)
                idx_upper = idx.clamp(max=Q.size(1) - 1)
                q_lower = Q.gather(1, idx_lower)
                q_upper = Q.gather(1, idx_upper)
                center = (q_lower + q_upper) / 2
                q = torch.minimum(
                    torch.maximum(center + (p - center) / inv_slope, q_lower), q_upper
                )
            # in-place update of model parameters
            p.copy_(q)
        return inv_slope
