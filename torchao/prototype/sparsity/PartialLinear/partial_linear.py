import math

import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class PartialLinear(Module):
    r"""Applies a linear transformation where each output feature connects to only the top-k
    input features by weight magnitude: :math:`y = x * (M \odot W^T) + b`,
    where $\odot$ is the element-wise product and $M$ is a binary mask.
    This module implements a form of structured sparsity that can reduce computation
    and memory usage during inference.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        top_k: number of weights to retain per output feature
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        update_mask_every: update the mask every N forward passes during training (default: 50)
    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        mask: binary mask of shape :math:`(\text{out\_features}, \text{in\_features})`
            indicating which weights are retained (1) or pruned (0)
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.PartialLinear(20, 30, top_k=5)  # Each output connected to top 5 inputs
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features", "top_k", "update_mask_every", "is_sparse_forward"]
    in_features: int
    out_features: int
    top_k: int
    update_mask_every: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        top_k: int,
        bias: bool = True,
        update_mask_every: int = 50,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if top_k <= 0 or top_k > in_features:
            raise ValueError(f"top_k must be between 1 and {in_features}, got {top_k}")

        self.top_k = top_k
        self.update_mask_every = update_mask_every
        self._forward_counter = 0

        # Create a full weight matrix
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        # Create a binary mask for the weights
        mask = torch.ones((out_features, in_features), dtype=torch.bool, device=factory_kwargs.get('device'))
        self.register_buffer('mask', mask)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard initialization as in nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        # Reset the mask to all ones
        self.mask.fill_(1.0)

        # Initialize the mask to keep only top-k weights
        self._update_mask()

    def _update_mask(self):
        with torch.no_grad():
            # Compute the magnitude of weights
            weight_mag = self.weight.abs()

            # Create a new binary mask
            new_mask = torch.zeros_like(self.mask, dtype=torch.bool)

            # For each output feature, find the top-k input connections
            _, top_k_indices = weight_mag.topk(self.top_k, dim=1)

            # Set mask to True for top-k weights for each output
            for i in range(self.out_features):
                new_mask[i, top_k_indices[i]] = True

            # Update the mask
            self.mask.copy_(new_mask)

    def forward(self, input: Tensor) -> Tensor:
        # During training, periodically update the mask
        if self.training:
            self._forward_counter += 1
            if self._forward_counter >= self.update_mask_every:
                self._update_mask()
                self._forward_counter = 0

        # Apply the mask to the weights
        masked_weight = self.weight * self.mask

        # Use the masked weights for the linear transformation
        return F.linear(input, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"top_k={self.top_k}, bias={self.bias is not None}, "
                f"update_mask_every={self.update_mask_every}, "
                f"is_sparse_forward={self.is_sparse_forward}")
