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
from torchao.quantization.utils import compute_error as SQNR

from torchao.dtypes import (
    to_affine_quantized_intx,
    TensorCoreTiledLayoutType
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)

def _check_linear_int4_k(k, group_size = 1, inner_k_tiles = None):
    k_divisible_by_group_size = k % group_size == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_group_size and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_group_size

class MultiTensor(torch.Tensor):
    get_qparams_func = None
    quantize_func = None
    dequantize_func = None
    combine_qparams_list_func = None
    make_qtensor = None
    skip_layer_func = None
    act_fake_quant_func = None
    percdamp = 0.01
    blocksize = 128
    group_size = -1

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

    def pad_to_length(self, length):
        if self.count < length:
            for _ in range(length-self.count):
                # we need to handle in place ops where we do want the model's value to stay changed.
                # e.g. if someone does z[1,:]=x, if z were a size 1 multiTensor and x were size 3,
                # we want z to become a multi tensor of size 3. Thus we pad the MultiTensor to the correct
                # size by adding new tensor instances (and not just instances of the pointers to the same original tensor)
                self.add_tensors(self.values[-1].detach())
        return self

    def unpad(self):
        if min([(self.values[0] == x).min() for x in self.values]):
            self.values = [self.values[0]]
            self.count = 1
        else:
            return self     


    @classmethod
    def configure_quantization_mode(
        cls,
        get_qparams_func,
        quantize_func,
        dequantize_func,
        combine_qparams_list_func,
        make_qtensor,
        skip_layer_func,
        act_fake_quant_func = None,
        percdamp = 0.01,
        blocksize = 128,
        group_size = -1
    ):
        cls.get_qparams_func = get_qparams_func
        cls.quantize_func = quantize_func
        cls.dequantize_func = dequantize_func
        cls.combine_qparams_list_func = combine_qparams_list_func
        cls.make_qtensor = make_qtensor
        cls.skip_layer_func = skip_layer_func
        cls.act_fake_quant_func = act_fake_quant_func if act_fake_quant_func is not None else lambda x: x
        cls.percdamp = percdamp
        cls.blocksize = blocksize
        cls.group_size = group_size

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
            # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
            multi_tensor_size = max([x.count if isinstance(x, MultiTensor) else 1 for x in flat])
            grouped = list(zip(*[x.pad_to_length(multi_tensor_size).values if isinstance(x, MultiTensor) else [x] * multi_tensor_size for x in flat]))
            return grouped

        def grouped_to_flat(grouped: List[Tuple[Any, ...]]) -> Tuple[List[Any], bool]:
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [(A,A,A), (b1,b2,b3), (c1,c2,c3)]
            flat_tups = list(zip(*grouped))
            # convert [(A,A,A), (b1,b2,b3), (c1,c2,c3)] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flattened = [cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0] for tup in flat_tups]
            non_tensors_equal = all(all(x == tup[0] for x in tup) for tup in flat_tups if not isinstance(tup[0], torch.Tensor))
            return flattened, non_tensors_equal
        
        def tensors_to_cuda(args):
            # this is needed because we want to execute the actual ops in cuda so they don't take forever
            new_args = []
            for x in args:
                new_args.append(x.cuda() if isinstance(x, torch.Tensor) and not isinstance(x, MultiTensor) else x)
            return new_args

        def copy_new_values(orig_inp, new_inp):
            for x, new_inp in zip(orig_inp, new_inp):
                if isinstance(x, torch.Tensor):
                    new_inp = new_inp.to(x.device)
                    if (x != new_inp).max():
                        x.copy_(new_inp)

        def unpad(args):
            for arg in args:
                if isinstance(arg, MultiTensor):
                    arg.unpad()

        quantize_linear = (
            not skip_gptq
            and cls.is_linear_layer(func)
        )

        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = flat_to_grouped(flat_args)
            
        with torch._C.DisableTorchFunctionSubclass():
            if quantize_linear:
                H = 0
                total_batches = 0

            outputs = []
            for inp in grouped_args:
                cuda_inp = tensors_to_cuda(inp)
                cur_args, cur_kwargs = tree_unflatten(cuda_inp, spec)

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
                    try:
                        out = func(*cur_args, **cur_kwargs)
                    except:
                        breakpoint()

                    outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)

                    # handling for in place ops: since we did that actual function in
                    # cuda, any in place operations changed the value of the cuda tensor
                    # not the original tensor passed into the function. So we check
                    # if anything changed and if so, copy the new value into the 
                    # original tensor
                    # TODO detect non in-place ops and skip this
                    copy_new_values(inp, cuda_inp)
                    


            if quantize_linear:

                W = args[1]
                if isinstance(W, MultiTensor):
                    W = W.values[0]
                W=W.to(H.device)
                
                
                
                Q, DQ, all_qparams = cls.faster_quant(H, W.detach())

                # Replace the original weight with the quantized tensor subclass
                qtensor  = cls.make_qtensor(Q, all_qparams)

                # Get the original parameter name
                state_dict_manager = StateDictManager.get_instance()
                original_param_name = state_dict_manager.get_name_for_param(args[1])
                state_dict_manager.update_param(original_param_name, qtensor)
                print(original_param_name)
                
                # Run the function again with updated weights and skip_gptq=True
                new_out = cls.__torch_function__(func, types, (args[0], DQ.cpu(), *args[2:]), kwargs, skip_gptq=True)

                if args[0].debug:
                    old_out = cls.__torch_function__(func, types, args, kwargs, skip_gptq=True)

                    DQ_after = cls.dequantize_func(Q, all_qparams).to(W.dtype)

                    print(
                        "SQNR for QDQ (this should be inf)", SQNR(DQ, DQ_after)
                    )  # matches
                    print(
                        "SQNR for weight (can be low)", SQNR(W, DQ.cuda())
                    )  # fine to not match
                    print(
                        "SQNR for output with GPTQ (hopefully 35+)",
                        SQNR(old_out.values[0], new_out.values[0])
                    )

                    qparams2 = cls.get_qparams_func(W)
                    Q2 = cls.quantize_func(W, qparams2)
                    DQ2 = cls.dequantize_func(Q2, qparams2).to(W.dtype)
                    old_q_out = cls.__torch_function__(func, types, (args[0], DQ2, *args[2:]), kwargs, skip_gptq=True)
                    print(
                        "SQNR for output without GPTQ (should be less than above)",
                        SQNR(old_out.values[0], old_q_out.values[0])
                    )
                return new_out
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

                # we padded each of the MultiTensors to match the largest multitensor so that if we had in place ops, we would be able
                # to store the many changed value and have those updates be reflected in the model. However if there are no in place ops, then
                # we just increased the size of all parameters/buffers by n times for no reason. To avoid issues, go back and unpad
                # everything where possible. i.e. all the multi tensor values are the same.
                unpad(flat_args)
                return tree_unflatten(flat_outputs, out_spec)

                
    @classmethod
    def faster_quant(cls, H, W):
        percdamp = cls.percdamp
        blocksize = cls.blocksize
        group_size = cls.group_size
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if group_size == -1:
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

                if group_size != -1 and (i1 + i) % group_size == 0:  # start of new group
                    cur_qparams = cls.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + group_size)]
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
        self.id_to_name = {}

    def set_state_dict(self, model):
        self.state_dict = model.state_dict()
        self.id_to_name = {id(v): k for k, v in model.named_parameters()}

    def update_id_to_name(self, model):
        self.id_to_name = {id(v): k for k, v in model.named_parameters()}

    def get_name_for_param(self, param):
        return self.id_to_name.get(id(param), None)

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
        self.make_qtensor = None
        self.skip_layer_func = None
        self.act_fake_quant_func = None

    def _check_functions(self):
        assert self.get_qparams_func is not None, "get_qparams_func must be set"
        assert self.quantize_func is not None, "quantize_func must be set"
        assert self.dequantize_func is not None, "dequantize_func must be set"
        assert self.combine_qparams_list_func is not None, "combine_qparams_list_func must be set"
        assert self.make_qtensor is not None, "make_qtensor must be set"
        assert self.skip_layer_func is not None, "skip_layer_func must be set"

    # this doesn't work
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
        group_size=64,
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Dict:
        MultiTensor.configure_quantization_mode(
            get_qparams_func=self.get_qparams_func,
            quantize_func=self.quantize_func,
            dequantize_func=self.dequantize_func,
            combine_qparams_list_func=self.combine_qparams_list_func,
            make_qtensor=self.make_qtensor,
            skip_layer_func=self.skip_layer_func,
            percdamp=percdamp,
            blocksize=blocksize,
            group_size=group_size
        )
        # Set the state dict for the original model
        self.state_dict_manager.set_state_dict(model)
        # Replace parameters with MultiTensor
        # self._replace_parameters_with_multitensor(model)
        # Replace buffers and parameters with MultiTensor
        with torch.no_grad():
            _replace_with_custom_fn_if_matches_filter(
                model=model,
                replacement_fn=replace_buffers_and_params_with_multitensors,
                filter_fn=lambda x, y: True
            )
        self.state_dict_manager.update_id_to_name(model)
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
        group_size=64,
        inner_k_tiles=8,
        padding_allowed=True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.device = device
        self.act_fake_quant_func = None
        n_bit = 4
        self.get_qparams_func = lambda w: get_groupwise_affine_qparams(
            w, n_bit, group_size
        )
        self.quantize_func = lambda w, qparams: groupwise_affine_quantize_tensor_from_qparams(
            w, qparams[0], qparams[1], n_bit, group_size
        )
        self.dequantize_func = lambda q, qparams: groupwise_affine_dequantize_tensor_from_qparams(
            q,
            qparams[0],
            qparams[1],
            n_bit,
            group_size,
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], group_size) or padding_allowed
        )

        def make_qtensor(q, qparams):
            # this should be setup to just use the quantized tensor and qparams directly to make
            # the aqt int4 tensor but i don't think we have that functionality atm so just dequant
            # then requant
            weight = self.dequantize_func(q, qparams)

            # copied from quant_api apply_int4_weight_only_quant (this should probably be made into a utility fn at some point)
            mapping_type = MappingType.ASYMMETRIC
            block_size = (1, group_size)
            target_dtype = torch.int32
            quant_min = 0
            quant_max = 15
            eps = 1e-6
            preserve_zero = False
            zero_point_dtype = torch.bfloat16
            zero_point_domain = ZeroPointDomain.FLOAT
            layout_type = TensorCoreTiledLayoutType(inner_k_tiles=8)
            # at least the big up to here should be a util

            quantized_tensor = to_affine_quantized_intx(
                weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, 
                zero_point_dtype=zero_point_dtype, 
                preserve_zero=preserve_zero, 
                zero_point_domain=zero_point_domain, 
                layout_type=layout_type, 
            )
            return quantized_tensor
        self.make_qtensor = make_qtensor

        self._check_functions()

    def quantize(self, model: torch.nn.Module, inputs: List[MultiTensor], **kwargs: Any) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.group_size,
        )
        # this is hacky and potentially wrong, better to just make the flow return a state dict and let user
        # do with it what they will
        _replace_with_custom_fn_if_matches_filter(
            model=model,
            replacement_fn=remove_multitensors_from_buffers_and_params,
            filter_fn=lambda x, y: True
        )
        model.load_state_dict(state_dict, assign=True, strict=False)
        
        return model

# this should probably be a multitensor method that can be applied and we just traverse
# and look for multitensors and unpack them
def remove_multitensors_from_buffers_and_params(model: nn.Module) -> nn.Module:
    for name, buf in model.named_buffers(recurse=False):
        if isinstance(buf, MultiTensor):
            setattr(model, name, buf.values[0])
    for name, param in model.named_parameters(recurse=False):
        if isinstance(param, MultiTensor):
            setattr(model, name, nn.Parameter(param.values[0], param.values[0].requires_grad))
    return model

def replace_buffers_and_params_with_multitensors(model: nn.Module) -> nn.Module:
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
