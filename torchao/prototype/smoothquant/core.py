from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization import PerChannelMinMaxObserver, HistogramObserver
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.observer import (
    AffineQuantizedObserverBase, PerRow
)


class SmoothQuantObserver(AffineQuantizedObserverBase):
    def __init__(self,
        weight: torch.Tensor,
        alpha: float = 0.5,
        quant_mode: str = "static", # or dynamic
        n_calib_examples: int = 20,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        zero_point_domain = ZeroPointDomain.INT,
    ):
        """
        A custom observer for SmoothQuant

        Args:
            weight: The weight tensor to be observed.
            mapping_type: symmetric or asymmetric quantization of weight
            alpha: The alpha value to determine smoothing factor
            quant_mode: The mode of activation quantization, either static or dynamic
            n_calib_examples: Number of examples used to calibrate observer
            quant_min: The minimum quantized value
            quant_max: The maximum quantized value
            eps: The minimum scale to avoid dividing by zero.
            scale_dtype: The data type of the scale tensor.
            zero_point_dtype: The data type of the zero point tensor.
            zero_point_domain: The domain of the zero point.
        """
        super().__init__(
            MappingType.SYMMETRIC,
            torch.int8,
            PerRow(), 
            quant_min = quant_min,
            quant_max = quant_max,
            eps = eps,
            scale_dtype = scale_dtype,
            zero_point_dtype = zero_point_dtype,
            preserve_zero = True,
            zero_point_domain = zero_point_domain,
        )
        assert weight.ndim == 2
        self.weight = weight
        self.n_calib_examples = n_calib_examples
        self.inputs = []
        self.device = self.weight.device
        self.alpha = alpha
        assert quant_mode in ["static", "dynamic"]
        self.quant_mode = quant_mode
        # act.shape = [mb, ic] (reshape if needed), wei.shape = [oc, ic]
        # *_ic_obs are used to determine smoothing_factor
        # *_mb/oc_obs are used to find qparams for quantization
        self.act_ic_obs = PerChannelMinMaxObserver(
            ch_axis=-1,
            dtype=torch.int8,
            qscheme=torch.per_channel_affine,
            eps=eps,
        )
        self.act_obs = HistogramObserver(
            dtype=torch.int8,
            qscheme=torch.per_tensor_symmetric,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.wei_ic_obs = PerChannelMinMaxObserver(
            ch_axis=1,
            dtype=torch.int8,
            qscheme=torch.per_channel_affine,
            eps=eps,
        )
        self.wei_oc_obs = PerChannelMinMaxObserver(
            ch_axis=0,
            dtype=torch.int8,
            qscheme=torch.per_channel_symmetric,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.wei_ic_obs(self.weight)

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        if self.quant_mode == "static":
          # record inputs to find qparams for activation
          if len(self.inputs) < self.n_calib_examples:
              self.inputs.append(input.to("cpu").view(-1, input.size(-1)))
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
        smoothing_factor = torch.pow(x_abs_max_per_ic, self.alpha) / torch.pow(
            w_abs_max_per_ic, 1 - self.alpha
        )
        # 3 apply smoothing factor to activations and find scales for static quantization
        act_scales = None
        if self.quant_mode == "static":
            inv_smoothing_factor = 1 / smoothing_factor
            for act in self.inputs:
                act_new = act * inv_smoothing_factor
                self.act_obs(act_new)
            act_scale, _ = self.act_obs.calculate_qparams()
            act_scales = torch.Tensor([act_scale]).to(self.device)
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
