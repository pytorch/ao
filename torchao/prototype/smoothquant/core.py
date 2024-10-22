from typing import Optional
import torch
import torch.nn.functional as F
from torchao.quantization.quant_primitives import (
    MappingType,
)
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver, PerAxis
)


class SmoothQuantObserver(torch.nn.Module):
    def __init__(self,
        weight: torch.Tensor,
        alpha: Optional[float] = 0.5,
        quant_mode: str = "static", # or dynamic
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        """
        A custom observer for SmoothQuant

        Args:
            weight: The weight tensor to be observed.
            alpha: The alpha value to determine smoothing factor, normally between 0 and 1.
                   Fall back to conventional quantization if alpha is None.
            quant_mode: The mode of activation quantization, either static or dynamic
            quant_min: The minimum quantized value
            quant_max: The maximum quantized value
            eps: The minimum scale to avoid dividing by zero.
        """
        super().__init__()
        assert weight.ndim == 2
        self.weight = weight
        self.inputs = []
        self.device = self.weight.device
        self.alpha = alpha
        assert quant_mode in ["static", "dynamic"]
        self.quant_mode = quant_mode
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        # act.shape = [mb, ic] (reshape if needed), wei.shape = [oc, ic]
        # *_ic_obs are used to determine smoothing_factor
        # wei_oc_obs is used to find qparams for quantization
        self.act_ic_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            PerAxis(-1),
            eps=eps,
        )
        self.wei_ic_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            PerAxis(-1),
            eps=eps,
        )
        self.wei_oc_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            PerAxis(0),
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.wei_ic_obs(self.weight)

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        self.act_ic_obs(input.to("cpu"))
        return input

    def calculate_qparams(self):
        # 1 Get min/max per IC from observers
        wei_min_per_ic = self.wei_ic_obs.min_val
        wei_max_per_ic = self.wei_ic_obs.max_val
        act_min_per_ic = self.act_ic_obs.min_val
        act_max_per_ic = self.act_ic_obs.max_val
        x_abs_max_per_ic = (
            torch.max(torch.abs(act_min_per_ic), torch.abs(act_max_per_ic)) + self.eps
        )
        w_abs_max_per_ic = (
            torch.max(torch.abs(wei_min_per_ic), torch.abs(wei_max_per_ic)) + self.eps
        )
        # 2 calculate the smoothing factor
        if self.alpha is None:
            # fall back to conventional quantization if alpha is None
            smoothing_factor = torch.ones_like(
                x_abs_max_per_ic, dtype=x_abs_max_per_ic.dtype, device=x_abs_max_per_ic.device
            )
        else:
            smoothing_factor = torch.pow(x_abs_max_per_ic, self.alpha) / torch.pow(
                w_abs_max_per_ic.to(x_abs_max_per_ic.device), 1 - self.alpha
            )
        # 3 apply smoothing factor to activations and find scales for static quantization
        act_scales = None
        if self.quant_mode == "static":
            act_min_per_ic_new = act_min_per_ic / smoothing_factor.reshape(
                act_min_per_ic.shape
            )
            act_max_per_ic_new = act_max_per_ic / smoothing_factor.reshape(
                act_max_per_ic.shape
            )
            min_val_per_tensor = torch.min(act_min_per_ic_new)
            max_val_per_tensor = torch.max(act_max_per_ic_new)
            min_val_neg = torch.min(min_val_per_tensor, torch.zeros_like(min_val_per_tensor))
            max_val_pos = torch.max(max_val_per_tensor, torch.zeros_like(max_val_per_tensor))
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            act_scale = max_val_pos / (float(self.quant_max - self.quant_min) / 2)
            act_scales = act_scale.to(self.device)
        # 4 update weight and find scales
        self.wei_oc_obs(self.weight * smoothing_factor.to(self.device))
        wei_scales, _ = self.wei_oc_obs.calculate_qparams()
        # 5 return results
        return smoothing_factor.to(self.device), act_scales, wei_scales.to(self.device)


class SmoothQuantObservedLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            obs: SmoothQuantObserver,
            device=None,
            dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        assert isinstance(obs, SmoothQuantObserver)
        self.obs = obs

    def forward(self, input: torch.Tensor):
        input = self.obs(input)
        output = F.linear(input, self.weight, self.bias)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, obs: SmoothQuantObserver):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            obs,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
