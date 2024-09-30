from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.ao.quantization import PerChannelMinMaxObserver, HistogramObserver
from torchao.dtypes.uintx.uintx import to_uintx
from torchao.dtypes.affine_quantized_tensor import (
    to_affine_quantized_intx,
    LayoutType,
    register_layout_cls,
    AQTLayout,
    register_aqt_quantized_linear_dispatch

) 
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.observer import (
    AffineQuantizedObserverBase, PerRow
)
from torchao.quantization.utils import (
    dynamically_quantize_per_channel,
    quant_int8_per_token_matmul,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_2, TORCH_VERSION_AT_LEAST_2_4

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
        reduce_range: Optional[bool] = False,
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
            reduce_range: Quantize act/wei to less than 8 bits on old platforms
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
            reduce_range=False,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.act_obs = HistogramObserver(
            dtype=torch.int8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.wei_ic_obs = PerChannelMinMaxObserver(
            ch_axis=1,
            dtype=torch.int8,
            qscheme=torch.per_channel_affine,
            reduce_range=False,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        self.wei_oc_obs = PerChannelMinMaxObserver(
            ch_axis=0,
            dtype=torch.int8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=reduce_range,
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
        self.act_ic_obs(input)
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
            act_scales = torch.Tensor([act_scale])
        # 4 update weight and find scales
        self.wei_oc_obs(self.weight * smoothing_factor)
        wei_scales, _ = self.wei_oc_obs.calculate_qparams()
        # 5 return results
        return smoothing_factor, act_scales, wei_scales


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


@dataclass(frozen=True)
class SmoothQuantLayoutType(LayoutType):
    inv_smoothing_factor: torch.Tensor
    act_scales: torch.Tensor
    wei_scales: torch.Tensor


def _quantized_linear_impl(input_tensor, weight_tensor, bias):
    inv_smoothing_factor = weight_tensor.layout_tensor.layout_type.inv_smoothing_factor
    act_scales = weight_tensor.layout_tensor.layout_type.act_scales
    wei_scales = weight_tensor.layout_tensor.layout_type.wei_scales
    input = input_tensor * inv_smoothing_factor
    if (weight_tensor.device.type == "cpu" and not TORCH_VERSION_AT_LEAST_2_4) or \
        not TORCH_VERSION_AT_LEAST_2_2:
        # _int_mm is not available on CUDA before PyTorch 2.2
        # _int_mm is not available on CPU before PyTorch 2.4
        # So compute in float here
        y = F.linear(input, weight_tensor.dequantize(), bias)
    else:
        target_dtype = torch.int8
        quant_min = _DTYPE_TO_QVALUE_BOUNDS[target_dtype][0]
        quant_max = _DTYPE_TO_QVALUE_BOUNDS[target_dtype][1]
        if act_scales is not None:
            # dynamic quant
            act_zero_points = torch.zeros_like(act_scales, dtype=torch.int64)
            qx = torch.ops.quantized_decomposed.quantize_per_tensor(
                input,
                act_scales,
                act_zero_points,
                quant_min,
                quant_max,
                dtype=target_dtype,
            )
            act_scales = act_scales * torch.ones(input.size(0), dtype=act_scales.dtype)
        else:
            # static quant
            qx, act_scales, _ = dynamically_quantize_per_channel(input, quant_min, quant_max, target_dtype)
        y = quant_int8_per_token_matmul(
            qx, act_scales, weight_tensor.layout_tensor.int_data, wei_scales
        )
        if bias is not None:
            y += bias
    return y.to(input_tensor.dtype)


def _linear_sq_check(input_tensor, weight_tensor, bias):
    return isinstance(weight_tensor.layout_tensor, SmoothQuantAQTLayout)


register_aqt_quantized_linear_dispatch(_linear_sq_check, _quantized_linear_impl)


@register_layout_cls(SmoothQuantLayoutType)
class SmoothQuantAQTLayout(AQTLayout):
    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.layout_type = layout_type     

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale, self.zero_point

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point"], [self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        layout_type, = tensor_attributes
        return cls(int_data, scale, zero_point, layout_type)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"SmoothQuantAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, SmoothQuantLayoutType)
        return cls(int_data, scale, zero_point, layout_type)

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        self.zero_point = fn(self.zero_point)
        return self
    
to_smooth_quant = SmoothQuantAQTLayout.from_plain