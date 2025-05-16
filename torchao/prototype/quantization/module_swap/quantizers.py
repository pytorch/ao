from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function

__supported_group_size_strings__ = ["per_token", "per_channel", "per_tensor"]


class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        return output_grad


class IntQuantizer(nn.Module):
    def __init__(
        self,
        num_bits: int,
        group_size: Union[int, str],
        dynamic: bool,
        quantization_mode: str,
        range_learning: bool = False,
        scale_eps: float = 1e-9,
    ) -> None:
        super().__init__()

        self.num_bits = num_bits
        self.group_size = group_size
        self.dynamic = dynamic
        self.quantization_mode = quantization_mode
        self.scale_eps = scale_eps
        self.scale: Optional[torch.Tensor] = None
        self.offset: Optional[torch.Tensor] = None
        self._range_learning = range_learning
        self.quant_dtype: torch._C.dtype = torch.float32
        self.quantize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize:
            return x

        q_min, q_max = self.get_qmin_qmax(
            self.num_bits, self.quant_mode_to_signed(self.quantization_mode)
        )

        if self.dynamic:
            scale, offset = IntQuantizer.get_scale_offset(
                x,
                self.group_size,
                self.quantization_mode,
                q_min,
                q_max,
            )
        else:
            scale = self.scale
            offset = self.offset

        if scale is None:
            raise ValueError("Initialize scale before first forward pass")

        group_size = self.process_group_size_to_int(self.group_size, x)

        if self._range_learning:
            scale = torch.clamp(scale, min=self.scale_eps)
            if offset is not None:
                offset = RoundStraightThrough.apply(offset)
                offset = torch.clamp(offset, q_min, q_max)

        return self.quantize_forward(
            x,
            scale=scale,
            offset=offset,
            group_size=group_size,
            q_min=q_min,
            q_max=q_max,
        )

    @property
    def q_min(self) -> int:
        q_min, _ = self.get_qmin_qmax(
            self.num_bits, self.quant_mode_to_signed(self.quantization_mode)
        )
        return q_min

    @property
    def q_max(self) -> int:
        _, q_max = self.get_qmin_qmax(
            self.num_bits, self.quant_mode_to_signed(self.quantization_mode)
        )
        return q_max

    @staticmethod
    def quant_mode_to_signed(quant_mode: str) -> bool:
        if quant_mode == "symmetric":
            return True
        elif quant_mode == "asymmetric":
            return False
        else:
            raise NotImplementedError

    @staticmethod
    def get_scale_param_size(
        x: torch.Tensor, group_size: Union[int, str]
    ) -> torch.Size:
        int_group_size = IntQuantizer.process_group_size_to_int(group_size, x)
        if int_group_size is None:
            return torch.Size([1])
        else:
            if len(x.shape) == 1:
                return torch.Size([x.shape[0] // int_group_size])
            else:
                size_list = [x.shape[i] for i in range(len(x.shape) - 1)]
                size_list.append(x.shape[-1] // int_group_size)
                return torch.Size(size_list)

    @staticmethod
    def get_qmin_qmax(n_bits: int, signed: bool) -> Tuple[int, int]:
        if signed:
            qmin = -(2 ** (n_bits - 1))
            qmax = 2 ** (n_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**n_bits - 1
        return qmin, qmax

    @staticmethod
    def get_scale_offset(
        x: torch.Tensor,
        group_size: Union[int, str],
        quantization_mode: str,
        q_min: int,
        q_max: int,
        scale_eps: float = 1e-9,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(group_size, int) and group_size == "per_tensor":
            x_min = torch.min(x)
            x_max = torch.max(x)
        else:
            int_group_size = IntQuantizer.process_group_size_to_int(group_size, x)
            assert int_group_size is not None
            reshaped_x = x.reshape(
                [*x.shape[:-1], x.shape[-1] // int_group_size, int_group_size]
            )
            x_min = torch.min(reshaped_x, dim=-1)[0]
            x_max = torch.max(reshaped_x, dim=-1)[0]

        if quantization_mode == "symmetric":
            scale = IntQuantizer.get_scale_from_min_max(
                x_min, x_max, q_min, q_max, scale_eps
            )
            offset = None
        elif quantization_mode == "asymmetric":
            scale, offset = IntQuantizer.get_scale_offset_from_min_max(
                x_min, x_max, q_min, q_max, scale_eps
            )
        else:
            raise NotImplementedError

        return scale, offset

    @staticmethod
    def quantize_forward(
        x: torch.Tensor,
        scale: torch.Tensor,
        offset: Optional[torch.Tensor],
        q_min: int,
        q_max: int,
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        # if quantization is per_group, we need to reshape the tensor to apply it over
        reshaped = False
        orig_shape = x.shape
        if group_size is not None and group_size != x.shape[-1]:
            x = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
            scale = torch.unsqueeze(scale, -1)
            if offset is not None:
                offset = torch.unsqueeze(offset, -1)
            reshaped = True

        if scale.device != x.device:
            scale = scale.to(x.device)
            if offset is not None:
                offset = offset.to(x.device)

        input_dtype = x.dtype

        if offset is None or offset.numel() == 0:
            x = torch.clamp(RoundStraightThrough.apply(x / scale), q_min, q_max)
            x = x * scale
        else:
            x = torch.clamp(
                RoundStraightThrough.apply(x / scale) + offset,
                q_min,
                q_max,
            )
            x = (x - offset) * scale

        # reshape back to original shape if groups were used
        if reshaped:
            x = x.reshape(orig_shape)

        return x.to(input_dtype)

    @staticmethod
    def process_group_size_to_int(
        group_size: Union[int, str], x: torch.Tensor
    ) -> Optional[int]:
        if isinstance(group_size, int):
            return group_size
        elif group_size == "per_channel" or group_size == "per_token":
            return x.shape[-1]
        elif group_size == "per_tensor":
            return None
        else:
            raise NotImplementedError

    @staticmethod
    def get_scale_from_min_max(
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        q_min: int,
        q_max: int,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        smin = x_min / float(q_min)
        smax = x_max / float(q_max)
        mask = smin > smax
        scale = torch.where(mask, smin, smax)
        scale = torch.clamp(scale, min=eps)
        return scale

    @staticmethod
    def get_scale_offset_from_min_max(
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        q_min: int,
        q_max: int,
        eps: float = 1e-9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = ((x_max - x_min) / (abs(q_min) + abs(q_max))).detach()
        offset = (-x_min / scale).detach()

        offset = torch.round(offset)
        offset = torch.clamp(offset, q_min, q_max)

        scale = torch.clamp(scale, min=eps)
        return scale, offset

    def set_scale_offset_to_min_max(self, x: torch.Tensor) -> None:
        assert not self.dynamic

        x = x.detach()
        x = x.to(self.quant_dtype)

        signed = self.quant_mode_to_signed(self.quantization_mode)

        q_min, q_max = self.get_qmin_qmax(self.num_bits, signed=signed)
        scale, offset = self.get_scale_offset(
            x, self.group_size, self.quantization_mode, q_min, q_max
        )

        # with per-tensor we get empty tensors sometimes
        if scale.data.shape == torch.Size([]):
            scale_size_fix = torch.ones([1]).to(scale.device)
            scale_size_fix *= scale
            scale.data = scale_size_fix.data

        if self._range_learning:
            self.scale = torch.nn.Parameter(scale, requires_grad=True).to(
                self.quant_dtype
            )
            if offset is not None:
                self.offset = torch.nn.Parameter(offset, requires_grad=True).to(
                    self.quant_dtype
                )
        else:
            self.scale = scale
            self.offset = offset


class CodeBookQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        features: int,
        codebook_dim: int = 1,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.group_size = codebook_dim
        self.codebook_dim = codebook_dim
        self.dynamic = True
        self.scale_eps = 1e-9
        assert features % codebook_dim == 0
        torch.manual_seed(seed)

        N = 2 ** (n_bits * codebook_dim)
        codebook_shape = (N, codebook_dim)
        self.codebook = nn.Parameter(
            torch.Tensor(codebook_shape[0], codebook_shape[1]), requires_grad=False
        )

        self._codebook_shape: Tuple[int, int] = codebook_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling_factor = torch.mean(abs(x), dim=1, keepdim=True).detach()
        output = scaling_factor * VectorQuantizerFunction.apply(
            x / scaling_factor, self.codebook
        )
        return output


class VectorQuantizerFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[2, 14]
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.view(-1, codebook.shape[1])
        distances = torch.cdist(flat_inputs, codebook)
        indices = torch.argmin(distances, dim=1)
        sums = torch.zeros_like(codebook).float().cuda()
        counts = torch.zeros(codebook.size(0), dtype=torch.float).cuda()

        # Accumulate sums using index_add_
        sums.index_add_(0, indices, flat_inputs.float())

        # Accumulate counts using index_add_
        ones = torch.ones(flat_inputs.size(0), dtype=torch.float).cuda()
        counts.index_add_(0, indices, ones)

        # Avoid division by zero
        counts = counts.unsqueeze(1).clamp(min=1.0)

        # Update each centroid with the average of assigned inputs
        codebook.copy_(sums / counts)

        quantized = codebook[indices].view_as(inputs)
        return quantized

    @staticmethod
    # pyre-ignore[2, 14]
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None
