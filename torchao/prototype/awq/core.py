from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing
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
)
from torchao.quantization.observer import (
    AffineQuantizedObserverBase, PerGroup
)


class AWQObserver(AffineQuantizedObserverBase):
    def __init__(self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        block_size: Tuple,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        n_validation_examples: int,
        validation_sequence_len: int,
        scale_search_space_size: int = 20,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: Optional[bool] = True,
        zero_point_domain = ZeroPointDomain.INT,
    ):
        """
        A custom observer for Activation aware Weight Quantization (AWQ)

        Args:
            weight: The weight tensor to be observed.
            bias: The bias tensor to be observed.
            block_size: The weight tensor shape after being reshaped to support per group quantization
            input_dtype: The data type of the input tensor.
            mapping_type: Always set to asymmetric 
            target_dtype: The target data type of the quantized tensor
            n_validation_examples: Number of examples used to calibrate observer
            validation_sequence_len: Number of tokens in each example
            scale_search_space_size: The number of scales to search for.
            quant_min: The minimum quantized value
            quant_max: The maximum quantized value
            eps: The minimum scale.
            scale_dtype: The data type of the scale tensor.
            zero_point_dtype: The data type of the zero point tensor.
            preserve_zero: A flag to indicate whether we need zero to be exactly
                representable or not.
            zero_point_domain: The domain of the zero point.
        """
        super().__init__(
            mapping_type,
            target_dtype,
            PerGroup(block_size[-1]), 
            quant_min = quant_min,
            quant_max = quant_max,
            eps = eps,
            scale_dtype = scale_dtype,
            zero_point_dtype = zero_point_dtype,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
        )
        self.block_size = block_size
        self.weight = weight
        self.bias = bias
        self.n_validation_examples = n_validation_examples
        self.validation_sequence_len = validation_sequence_len
        self.calibration_token_count = 0
        self.inputs = []
        self.outputs = []
        self.scale_options = scale_search_space_size
        self.device = self.weight.device
        self.average =  torch.zeros((1,weight.shape[1]), device= self.device)
        if self.bias is not None:
            self.bias.to(self.device)
    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        # import pdb
        # pdb.set_trace()
        # print(input.shape, input.abs().sum(1).shape, self.average.shape)
        if len(self.inputs) < self.n_validation_examples:
            self.inputs.append(input.to("cpu"))
            self.outputs.append(output.to("cpu"))
        self.calibration_token_count += input.shape[-2]
        self.average += input.abs().sum(-2)
        
        

    def calculate_qparams(self):
        # import pdb
        # pdb.set_trace()
        assert self.outputs != None, "calibrate observer first by running model on exemplar data"
        self.average /= (self.calibration_token_count)
        for i in range(self.n_validation_examples):
            self.inputs[i] = self.inputs[i].to(self.device)
            self.outputs[i] = self.outputs[i].to(self.device)

        best_loss = float('inf')
        best_scales = None
        for i in range(self.scale_options):
            ratio = i * 1 / self.scale_options
            scales = self.average.pow(ratio).to(self.weight.dtype)
            scales = scales / (scales.max() * scales.min()).sqrt()
            layout = AwqLayoutType(self.target_dtype, scales)
            # regardless of weight dtype, we have to store as packed uint8 tensors
            tensor_dtype = torch.uint8
            w = to_affine_quantized_intx(
                self.weight*scales,
                self.mapping_type,
                self.block_size,
                tensor_dtype,
                quant_min = self.quant_min,
                quant_max = self.quant_max,
                eps = self.eps,
                scale_dtype = self.scale_dtype,
                zero_point_dtype = self.zero_point_dtype,
                preserve_zero = self.preserve_zero,
                zero_point_domain = self.zero_point_domain,
                layout_type = layout
            )
            loss = 0
            for i in range(self.n_validation_examples):
                q_out = F.linear(self.inputs[i]/scales, w, self.bias)
                loss += (self.outputs[i] - q_out).pow(2).mean().item()
            if loss < best_loss:
                best_scales = scales
                best_loss = loss
            for i in range(self.n_validation_examples):
                self.inputs[i].to("cpu")
                self.outputs[i].to("cpu")
        return best_scales.detach()

class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs
        self.equalization_scale = None

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserver):
        observed_linear = cls(float_linear.in_features, float_linear.out_features, act_obs, False, device=float_linear.weight.device, dtype=float_linear.weight.dtype)
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
    

@dataclass(frozen=True)
class AwqLayoutType(LayoutType):
    dtype: torch.dtype
    equalization_scale: torch.Tensor

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        # pack weights for sub dtype bit size
        if self.dtype != torch.uint8:
            return to_uintx(input, self.dtype)
        return input
    
def _quantized_linear_impl(input_tensor, weight_tensor, bias):
    # divide activations by awq scales
    return F.linear(input_tensor / weight_tensor.layout_tensor.equalization_scale, weight_tensor.dequantize(), bias)

def _linear_awq_check(input_tensor, weight_tensor, bias):
    return isinstance(weight_tensor.layout_tensor, AwqAQTLayout)

register_aqt_quantized_linear_dispatch(_linear_awq_check, _quantized_linear_impl)

@register_layout_cls(AwqLayoutType)
class AwqAQTLayout(AQTLayout):
    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        equalization_scale: torch.Tensor,
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
        equalization_scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.equalization_scale = equalization_scale
        self.layout_type = layout_type     
        
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # unpack if needed
        w = self.int_data if self.layout_type.dtype == torch.uint8 else self.int_data.get_plain()
        return w, self.scale, self.zero_point
    
    def __tensor_flatten__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ["int_data", "scale", "zero_point", "equalization_scale"], [self.layout_type]
    
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        equalization_scale = tensor_data_dict["equalization_scale"]
        layout_type, = tensor_attributes
        return cls(int_data, scale, zero_point, equalization_scale, layout_type)
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"AwqAQTLayout dispatch: attempting to run {func}, this is not supported"
        )
        
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, AwqLayoutType)
        return cls(int_data, scale, zero_point, layout_type.equalization_scale, layout_type)
    
    def get_layout_type(self) -> LayoutType:
        return self.layout_type
    
    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        self.zero_point = fn(self.zero_point)
        self.equalization_scale = fn(self.equalization_scale)
        return self
    
to_awq = AwqAQTLayout.from_plain