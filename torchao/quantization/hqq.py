import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
aten = torch.ops.aten

__all__ = [
    "HQQ4Mix16LinearWeight",
]

class HQQ4Mix16LinearWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, aqt_tensor, mixed_tensor, permutation, shape, *args, **kwargs):
        print(type(aqt_tensor))
        if "Core" in str(type(aqt_tensor)):
            breakpoint()
        kwargs["device"] = aqt_tensor.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else aqt_tensor.layout
        )
        assert "dtype" in kwargs
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        aqt_tensor,
        mixed_tensor,
        permutation,
        shape,
        *args,
        **kwargs,
    ):
        self.aqt_tensor = aqt_tensor
        self.mixed_tensor = mixed_tensor
        self.permutation = permutation

    def __tensor_flatten__(self):
        return ["aqt_tensor", "mixed_tensor"], [self.permutation, self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):
        permutation, shape, dtype = tensor_attributes
        return cls(
            tensor_data_dict["aqt_tensor"],
            tensor_data_dict["mixed_tensor"],
            permutation,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides = outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_float,
        num_saved_features=32,
        group_size=64,
        inner_k_tiles=8,
        mixed_dtype=torch.bfloat16
    ):
        def pick_saved_indices(input_float, num_saved_features):
            # random implementation for perf benchmarking
            features = list(range(input_float.shape[0]))
            import random
            return sorted(random.sample(features, num_saved_features))

        # get mixed_tensor
        mixed_indices = pick_saved_indices(input_float, num_saved_features)
        # input float is a linear weight with shape out_feat x in_feat
        # we extract the out_features denoted by the indices
        mixed_tensor = input_float[mixed_indices, :].to(dtype=mixed_dtype)

        # get aqt_tensor
        aqt_indices = [x for x in range(input_float.shape[0]) if x not in mixed_indices]
        aqt_float = input_float[aqt_indices, :]
        from torchao.quantization.quant_api import int4_weight_only
        q_func = int4_weight_only(group_size, inner_k_tiles)
        aqt_tensor = q_func(aqt_float)

        # get permutation list, i.e. for each original index, which index did it end up at when ordered [aqt_indices, mixed_indices]
        perm = torch.argsort(torch.tensor([x for x in range(input_float.shape[0]) if x not in mixed_indices] + mixed_indices)).tolist()

        return cls(
            aqt_tensor,
            mixed_tensor,
            perm,
            input_float.shape,
            dtype=mixed_dtype,
        )

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        breakpoint()
        aqt_out = torch.nn.functional.linear(act_mat, w_qtensor.aqt_tensor)
        mixed_out = torch.nn.functional.linear(act_mat, w_qtensor.mixed_tensor)
        return torch.cat((aqt_out, mixed_out), 1)[...,w_qtensor.permutation]+bias
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_qtensor, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            return cls._quantized_op(mat1, w_qtensor, bias)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.aqt_tensor),
            fn(self.mixed_tensor),
            self.permutation,
            self.shape,
            dtype=self.dtype,
        )

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        return self.__class__(
            self.aqt_tensor.to(device),
            self.mixed_tensor.to(device),
            self.permutation,
            self.shape,
            dtype=self.dtype
        )

    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(func)
        kwargs = {} if kwargs is None else kwargs
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten._to_copy.default:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
            )

        # if func is aten.t.default:
        #     return return_and_correct_aliasing(
        #         func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
        #     )
        breakpoint()
        raise NotImplementedError(f"No specialized dispatch found for quantized linear op {func}")
