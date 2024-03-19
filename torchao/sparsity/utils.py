import torch
from torch.ao.quantization.observer import UniformQuantizationObserverBase

__all__ = ["PerChannelNormObserver"]

# Observers
class PerChannelNormObserver(UniformQuantizationObserverBase):
    """
    A custom observer that computes the L2 norm of each channel and stores it in a buffer.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, **kwargs) -> None:
        # init with fixed qparams for quantization flow
        super().__init__(
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            reduce_range=False,
            quant_min=None,
            quant_max=None,
            eps=torch.finfo(torch.float32).eps,
            **kwargs
        )
        # set averaging constant so quantization flow knows observer is memoryless.
        self.averaging_constant = 1.0
        self.register_buffer("norm", torch.tensor([]))

    # pyre-fixme[14]: `forward` overrides method defined in `ObserverBase`
    #  inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape

        # channel_ax is always the last dimension
        new_axis_list = [i for i in range(x.dim())]  # noqa: C416
        new_axis_list[0], new_axis_list[-1] = new_axis_list[-1], new_axis_list[0]
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        norm = torch.norm(y, dim=1) ** 2

        if self.norm.numel() == 0:
            self.norm.resize_(norm.shape)
            self.norm.copy_(norm)
        else:
            self.norm += norm

        return x_orig

    # pyre-fixme[14]: `calculate_qparams` overrides method defined in `ObserverBase`
    #  inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def calculate_qparams(self):
        raise NotImplementedError("PerChannelNormObserver is designed to store activations only. ")
