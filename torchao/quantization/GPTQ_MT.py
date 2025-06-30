import torch
from collections import defaultdict
from typing import Callable, Any, Union, List, Tuple, Dict, Optional, Sequence, OrderedDict
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
import gc
from torchao.quantization.unified import Quantizer
from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    groupwise_affine_dequantize_tensor_from_qparams
)

def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = None):
    k_divisible_by_groupsize = k % groupsize == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_groupsize

class MultiTensor(torch.Tensor):
    get_qparams_func = None
    quantize_func = None
    dequantize_func = None
    combine_qparams_list_func = None
    make_names_and_values_dict_func = None
    skip_layer_func = None
    act_fake_quant_func = None
    percdamp = 0.01
    blocksize = 128
    groupsize = -1

    @staticmethod
    def __new__(cls, input: Union[torch.Tensor, Sequence[torch.Tensor]], **kwargs: Any) -> "MultiTensor":
        if isinstance(input, (list, tuple)):
            input = input[0]
        kwargs["dtype"] = kwargs.get("dtype", input.dtype)
        shape = kwargs.pop("shape", input.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, input: Union[torch.Tensor, Sequence[torch.Tensor]],**kwargs: Any) -> None:
        self.values: List[torch.Tensor] = []
        self.state_dict_manager = StateDictManager.get_instance()
        self.count: int = 0
        self.add_tensors(input)
        self.debug: bool = True
        self.gptq_done = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.values})"

    def add_tensors(self, input: Union[torch.Tensor, Sequence[torch.Tensor]]) -> "MultiTensor":
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(input, torch.Tensor), f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            self.count += 1
            self.values.append(input)
        return self

    def pad_to_length(self, length: int) -> "MultiTensor":
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]]*(length-self.count))
        return self

    @classmethod
    def configure_quantization_mode(
        cls,
        get_qparams_func,
        quantize_func,
        dequantize_func,
        combine_qparams_list_func,
        make_names_and_values_dict_func,
        skip_layer_func,
        act_fake_quant_func = None,
        percdamp = 0.01,
        blocksize = 128,
        groupsize = -1
    ):
        cls.get_qparams_func = get_qparams_func
        cls.quantize_func = quantize_func
        cls.dequantize_func = dequantize_func
        cls.combine_qparams_list_func = combine_qparams_list_func
        cls.make_names_and_values_dict_func = make_names_and_values_dict_func
        cls.skip_layer_func = skip_layer_func
        cls.act_fake_quant_func = act_fake_quant_func if act_fake_quant_func is not None else lambda x: x
        cls.percdamp = percdamp
        cls.blocksize = blocksize
        cls.groupsize = groupsize

    @classmethod
    def __torch_function__(
        cls, 
        func: Callable, 
        types: Tuple[type, ...], 
        args: Tuple[Any, ...]=(), 
        kwargs: Optional[Dict[str, Any]]=None, 
        skip_gptq:bool=False
    ) -> Any:
        def flat_to_grouped(flat: List[Any]) -> List[Tuple[Any, ...]]:
            multi_tensor_size = max([x.count if isinstance(x, MultiTensor) else 1 for x in flat])
            grouped = list(zip(*[x.pad_to_length(multi_tensor_size).values if isinstance(x, MultiTensor) else [x] * multi_tensor_size for x in flat]))
            return grouped

        def grouped_to_flat(grouped: List[Tuple[Any, ...]]) -> Tuple[List[Any], bool]:
            flat_tups = list(zip(*grouped))
            flattened = [cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0] for tup in flat_tups]
            non_tensors_equal = all(all(x == tup[0] for x in tup) for tup in flat_tups if not isinstance(tup[0], torch.Tensor))
            return flattened, non_tensors_equal

        kwargs = {} if kwargs is None else kwargs
        
        flat_args, spec = tree_flatten((args, kwargs))
        grouped_args = flat_to_grouped(flat_args)
        
        quantize_linear = (
            not skip_gptq
            and cls.is_linear_layer(func)
        )

        if quantize_linear:
            H = 0
            total_batches = 0
            
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            for inp in grouped_args:
                cur_args, cur_kwargs = tree_unflatten(inp, spec)
                # Check if we're in a linear layer
                if quantize_linear:
                    #Construct Hessian matrix for quantization
                    x = cur_args[0].float()
                    #x = self.act_fake_quant_func(x)
                    shape = x.shape
                    n = 1 if len(shape) == 2 else shape[0]
                    H *= total_batches / (total_batches + n)
                    total_batches += n
                    x = ((2 / total_batches) ** (1 / 2)) * x.reshape(
                        -1, shape[-1]
                    ).t().float()
                    H += x.matmul(x.t())    
                else:
                    out = func(*cur_args, **cur_kwargs)
                    outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)

            if quantize_linear:
                weight_tensor = cur_args[1]
                if isinstance(weight_tensor, MultiTensor):
                    weight_tensor = weight_tensor.values[0]
                weight_tensor.to("cuda")
                # Get the original parameter name
                state_dict_manager = StateDictManager.get_instance()
                original_param_name = state_dict_manager.get_name_for_param(weight_tensor)
                Q, DQ, all_qparams = cls.faster_quant(H, weight_tensor)
                # Replace the original weight with the quantized weight
                
                state_dict_manager.update_param(original_param_name, Q)
                # Run the function again with updated weights and skip_gptq=True
                return cls.__torch_function__(func, types, cur_args, kwargs, skip_gptq=True)
            else:
                grouped_outputs = [tree_flatten(x)[0] for x in outputs]
                out_spec = tree_flatten(outputs[0])[1]
                # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
                flat_outputs, non_tensors_equal = grouped_to_flat(grouped_outputs)
                assert non_tensors_equal, (
                    f"ERR: found a function in model: {func} which "
                    +"caused an error in GPTQ MultiTensor, the function dispatch only works for functions"
                    +" with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
                )
                return tree_unflatten(flat_outputs, out_spec)

                
    @classmethod
    def faster_quant(cls, H, W):
        percdamp = cls.percdamp
        blocksize = cls.blocksize
        groupsize = cls.groupsize
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if groupsize == -1:
            cur_qparams = cls.get_qparams_func(W)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        all_qparams = []
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:  # start of new group
                    cur_qparams = cls.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + groupsize)]
                    )
                    all_qparams.append(cur_qparams)

                q = cls.quantize_func(w.unsqueeze(1), cur_qparams).flatten()
                dq = cls.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()

                DQ1[:, i] = dq
                Losses1[:, i] = (w - dq) ** 2 / d**2

                err1 = (w - dq) / d
                W1[:, i:] -= (
                    err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1

            DQ[:, i1:i2] = DQ1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if all_qparams == []:
            all_qparams.append(cur_qparams)

        all_qparams = cls.combine_qparams_list_func(all_qparams)
        Q = cls.quantize_func(DQ, all_qparams)
        return Q, DQ.to(orig_dtype), all_qparams

    @classmethod
    def __torch_dispatch__(cls, func: Callable, types: Tuple[type, ...], args: Tuple[Any, ...]=(), kwargs: Dict[str, Any]={}, skip_gptq: bool=False) -> Any:
        pass

    def __tensor_flatten__(self) -> Tuple[List[str], Optional[Any]]:
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict: Dict[str, Any],
        tensor_attributes: Optional[Any],
        outer_size: torch.Size,
        outer_stride: Tuple[int, ...]
    ) -> 'MultiTensor':
        return cls(tensor_data_dict["values"])
    
    @classmethod
    def is_linear_layer(cls, func: Callable) -> bool:
        return func == torch.nn.functional.linear

class StateDictManager:
    _instance = None

    @staticmethod
    def get_instance():
        if StateDictManager._instance is None:
            StateDictManager._instance = StateDictManager()
        return StateDictManager._instance

    def __init__(self):
        self.state_dict = {}
        self.param_ptr_to_name = {}

    def set_state_dict(self, model):
        self.state_dict = model.state_dict()
        self.param_ptr_to_name = {v.data_ptr(): k for k, v in model.named_parameters()}

    def get_name_for_param(self, param):
        return self.param_ptr_to_name.get(param.data_ptr(), None)

    def update_param(self, name, new_value):
        if name in self.state_dict:
            if isinstance(new_value, MultiTensor):
                self.state_dict[name] = new_value.values[0]  # Convert MultiTensor to regular tensor
            else:
                self.state_dict[name] = new_value
        else:
            raise KeyError(f"Parameter {name} not found in state_dict")

    def get_state_dict(self):
        return self.state_dict

class GPTQQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.state_dict_manager = StateDictManager.get_instance()
        self.get_qparams_func = None
        self.quantize_func = None
        self.dequantize_func = None
        self.combine_qparams_list_func = None
        self.make_names_and_values_dict_func = None
        self.skip_layer_func = None
        self.act_fake_quant_func = None

    def _check_functions(self):
        assert self.get_qparams_func is not None, "get_qparams_func must be set"
        assert self.quantize_func is not None, "quantize_func must be set"
        assert self.dequantize_func is not None, "dequantize_func must be set"
        assert self.combine_qparams_list_func is not None, "combine_qparams_list_func must be set"
        assert self.make_names_and_values_dict_func is not None, "make_names_and_values_dict_func must be set"
        assert self.skip_layer_func is not None, "skip_layer_func must be set"

    def _replace_parameters_with_multitensor(self, model):
        for name, param in model.named_parameters():
            setattr(model, name.split('.')[-1], MultiTensor(param))
    
    def covert_multi_tensors_to_tensors(self, state_dict):
        for key, value in state_dict.items():
            if isinstance(value, MultiTensor):
                state_dict[key] = value.values[0]
        return state_dict
        
    
    @torch.no_grad()
    def _create_quantized_state_dict(
        self,
        model,
        inputs,
        blocksize=128,
        percdamp=0.01,
        groupsize=64,
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Dict:
        MultiTensor.configure_quantization_mode(
            get_qparams_func=self.get_qparams_func,
            quantize_func=self.quantize_func,
            dequantize_func=self.dequantize_func,
            combine_qparams_list_func=self.combine_qparams_list_func,
            make_names_and_values_dict_func=self.make_names_and_values_dict_func,
            skip_layer_func=self.skip_layer_func,
            percdamp=percdamp,
            blocksize=blocksize,
            groupsize=groupsize
        )
        # Set the state dict for the original model
        self.state_dict_manager.set_state_dict(model)
        # Replace parameters with MultiTensor
        self._replace_parameters_with_multitensor(model)
        # Replace buffers and parameters with MultiTensor
        with torch.no_grad():
            _replace_with_custom_fn_if_matches_filter(
                model=model,
                replacement_fn=replace_buffers_and_params,
                filter_fn=lambda x, y: True
            )
         # Run the model
        with torch.no_grad():
            out = model(*inputs)
        state_dict = self.state_dict_manager.get_state_dict()
        del_list = []
        for param_fqn in state_dict:
            if "kv_cache" in param_fqn:
                del_list.append(param_fqn)
        for param_fqn in del_list:
            state_dict.pop(param_fqn)
        return state_dict

class Int4WeightOnlyGPTQQuantizer(GPTQQuantizer):
    def __init__(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=64,
        inner_k_tiles=8,
        padding_allowed=True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.device = device
        self.act_fake_quant_func = None
        n_bit = 4
        self.get_qparams_func = lambda w: get_groupwise_affine_qparams(
            w, n_bit, groupsize
        )
        self.quantize_func = lambda w, qparams: groupwise_affine_quantize_tensor_from_qparams(
            w, qparams[0], qparams[1], n_bit, groupsize
        )
        self.dequantize_func = lambda q, qparams: groupwise_affine_dequantize_tensor_from_qparams(
            q,
            qparams[0],
            qparams[1],
            n_bit,
            groupsize,
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize) or padding_allowed
        )
        self.make_names_and_values_dict_func = lambda Q, qparams: {
            "Q": Q,
            "qparams": qparams,
        }
        self._check_functions()

    def quantize(self, model: torch.nn.Module, inputs: List[MultiTensor], **kwargs: Any) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.groupsize,
        )

        model.load_state_dict(state_dict, strict=False)
        
        
        
        return model

def replace_buffers_and_params(model: nn.Module) -> nn.Module:
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, nn.Parameter(MultiTensor([param]), param.requires_grad))
    return model

def _replace_with_custom_fn_if_matches_filter(
    model: nn.Module,
    replacement_fn: Callable[[nn.Module], nn.Module],
    filter_fn: Callable[[nn.Module, str], bool],
    cur_fqn: str ="",
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
    for name, child in model.named_children():
        new_child = _replace_with_custom_fn_if_matches_filter(
            child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
        )
        if new_child is not child:
            setattr(model, name, new_child)
    return model
