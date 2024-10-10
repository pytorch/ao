import torch
from typing import Callable, Any, Union, List, Tuple, Dict, Optional, Sequence
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchao.quantization.unified import Quantizer
from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    groupwise_affine_dequantize_tensor_from_qparams
)
from torchao.quantization.utils import compute_error as SQNR

from torchao.dtypes import (
    to_affine_quantized_intx_static,
    TensorCoreTiledLayout
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

NON_IN_PLACE_OPS = {}

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
    in_place_threshold = 5 # Number of times to see a function before assuming it's not in-place

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

    def pad_to_length(self, length, pad_in_place=True):
        if self.count < length:
            if pad_in_place:
                for _ in range(length-self.count):
                    # we need to handle in place ops where we do want the model's value to stay changed.
                    # e.g. if someone does z[1,:]=x, if z were a size 1 multiTensor and x were size 3,
                    # we want z to become a multi tensor of size 3. Thus we pad the MultiTensor to the correct
                    # size by adding new tensor instances (and not just instances of the pointers to the same original tensor..
                    # otherwise changes to one would change all of them)
                    self.add_tensors(self.values[-1].clone())
            else:
                # for non in place ops, no need to bloat memory, can just pad with same tensor instance
                return self.__class__(self.values).add_tensors([self.values[-1]]*(length-self.count))
        return self
        

    def unpad(self, count=1, force=False):
        count = min(count, self.count)
        if force or all((self.values[0] == x).all().item() for x in self.values):
            self.values = self.values[:count]
            self.count = count
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
        def flat_to_grouped(flat: List[Any], pad_in_place=True) -> List[Tuple[Any, ...]]:
            # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
            multi_tensor_size = max([x.count if isinstance(x, MultiTensor) else 1 for x in flat])
            grouped = list(zip(*[x.pad_to_length(multi_tensor_size, pad_in_place=pad_in_place).values if isinstance(x, MultiTensor) else [x] * multi_tensor_size for x in flat]))
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
                if isinstance(x, MultiTensor) and x.count == 1:
                    new_args.append(x.__class__(x.values[0].cuda()))
                else:
                    new_args.append(x.cuda() if isinstance(x, torch.Tensor) and not isinstance(x, MultiTensor) else x)
            return new_args
        def maybe_copy_new_values(orig_inp, new_inp):
            detected_difference = False
            for x, new_x in zip(orig_inp, new_inp):
                if isinstance(x, torch.Tensor):
                    new_x = new_x.to(x.device)
                    if (x != new_x).any():
                        x.copy_(new_x)
                        detected_difference = True
            return detected_difference
        def unpad(args, orig_counts, force=False):
            for arg, count in zip(args, orig_counts):
                if isinstance(arg, MultiTensor) and arg.count > count:
                    arg.unpad(count, force)

        # The way MultiTensor handles various functions is as follows. Normally when you apply a function on a MultiTensor that has n Tensors inside, we want
        # the function handling here to run that function once for each of the MultiTensor inputs. We also want it to happen in the same way as if you ran the function
        # first input and the second independently, i.e. if you ran model(input1)=out1 and model(input2), vs model(MultiTensor(input1, input2)) the output for the
        # second case should be MultiTensor(out1, out2) and all the activations along the way should be the same. Normally it is easy enough to handle MultiTensors
        # running the func for each set of inputs and then combine the outputs back into MultiTensors at the end if applicable. Since we can have a lot of tensors in a MultiTensor but we
        # normally want to execute the function on cuda, we can move them over to cuda before we evaluate the function.
        # this scheme works pretty well but has a few issues
        # 1) We end up moving the same tensors to cuda over and over again which can make things slow. if we have a function input like (MultiTensor(a), MultiTensor(b1, b2, ...b_n) 
        # when we pad and group MultiTensors into Tensor-only function inputs i.e. (a, b1), (a, b2), ..., (a, b_n), since each Tensor-only input uses a common tensor a but a unique tensor b,
        # we would be moving the same tensor a to cuda n times for no reason. This is easy to fix normally, just move all singular MultiTensors to cuda before padding/grouping but...
        # 2) In place ops are tricky to handle (funcs that modify the inputs). We want in_place operations like k_cache[:, indices] = k_val (where k_cache, k_val are MultiTensor inputs)
        # to be supported but if we move tensors to cuda then any modifications to the inputs will be applied to the cuda tensors rather than the originals. Additionally the originals are often
        # singular MultiTensors i.e. MultiTensor(a) that only has a single value, we don't want each in place op to overwrite the value of a, we need each change to the inputs to be recorded. So
        # we can modify MultiTensor(a) -> MultiTensor(a1, a2, ...) at the start so that each change of a can be recorded and won't overwrite the value of a for other ops, but then problem 1 comes back
        # since we have to move a1, a2, a3...etc to cuda over and over again. despite them all being the same value initially. And if we move them to cuda then the in place value changes will
        # be applied to a1_cuda not a1 and a1_cuda isn't in the MultiTensor. So we have to manually copy those values back a1.copy_(a1_cuda).
        # There's not really a great way to resolve the 2 issues, when there's an in place op you have to do the slow thing with cuda and checking for modified values....etc, when
        # there's not an in place op you can throw all singular tensors onto cuda at the start and go much faster.
        # This brings up the final issue, how do we know if we have an in place op? In general we don't so I added handling to MultiTensor to resolve that as well as can be hoped.
        # we have a dict that contains all the funcs we see NON_IN_PLACE_OPS, we initially treat ops as in place and see if any of the inputs got modified if they do then it gets
        # set to always be handled as an in place op. If nothing changes then once we've seen the op enough times that we're confident its not an in place op (cls.in_place_threshold)
        # then we can do the fast thing.
    

        quantize_linear = (
            not skip_gptq
            and cls.is_linear_layer(func)
        )
        # Determine if function is in-place

        #initialize function tracking
        if func not in NON_IN_PLACE_OPS:
            NON_IN_PLACE_OPS[func] = {'count': 0, 'is_in_place': None}
        NON_IN_PLACE_OPS[func]['count'] += 1

        
        if NON_IN_PLACE_OPS[func]['is_in_place'] is not None:
            is_in_place = NON_IN_PLACE_OPS[func]['is_in_place']
        elif NON_IN_PLACE_OPS[func]['count'] >= cls.in_place_threshold or quantize_linear:
            is_in_place = False  # Assume not in-place after threshold
        else:
            is_in_place = True

        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs into a single tuple
        # flat_args holds all the actual inputs, spec stores the original structure
        flat_args, spec = tree_flatten((args, kwargs))

        orig_counts = [x.count if isinstance(x, MultiTensor) else 1 for x in flat_args]
        
        # if we're not doing an in place op, move singular tensors to cuda now
        if not is_in_place:
            flat_args = tensors_to_cuda(flat_args)

        # convert [A, MultiTensor(b), MultiTensor(c1,c2,c3)] => [[A,b,c1], [A,b,c2] [A,b,c3]]
        # if its in place then instead we first convert MultiTensor(b) => MultiTensor(b1, b2, b3)
        # then proceed as normal.
        grouped_args = flat_to_grouped(flat_args, is_in_place)
            
        with torch._C.DisableTorchFunctionSubclass():
            if quantize_linear:
                H = 0
                total_batches = 0

            outputs = []
            for inp in grouped_args:
                # we move all remaining cpu tensors to cuda
                cuda_inp = tensors_to_cuda(inp)

                # return input to original structure
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

                    # if we're doing an in place op, here is where we copy modifications
                    # back to the original tensors, if we saw any differences, immediately 
                    # categortize func as in place, otherwise we can treat as not in
                    # place (especially for the upcoming unpad step)
                    if is_in_place:
                        detected_difference = maybe_copy_new_values(inp, cuda_inp)
                        #if detected_difference and not isinstance(NON_IN_PLACE_OPS[func], bool):
                            #print("THIS OP IS IN PLACE", func)
                            #NON_IN_PLACE_OPS[func] = False
                        if detected_difference:
                            NON_IN_PLACE_OPS[func]['is_in_place'] = True
                            print(f"Function {func} is in-place")

                        elif NON_IN_PLACE_OPS[func]['count'] >= cls.in_place_threshold:
                            NON_IN_PLACE_OPS[func]['is_in_place'] = False


            if quantize_linear:
                # turn weight MultiTensor into single cuda tensor
                W = args[1]
                if isinstance(W, MultiTensor):
                    W = W.values[0]
                W=W.to(H.device)
                
                Q, DQ, all_qparams = cls.faster_quant(H, W.detach())

                # make quantized tensor subclass
                qtensor  = cls.make_qtensor(Q, all_qparams)

                # Get the original parameter name
                state_dict_manager = StateDictManager.get_instance()
                original_param_name = state_dict_manager.get_name_for_param(args[1])
                state_dict_manager.update_param(original_param_name, qtensor)
                print(original_param_name)
                
                # Run the function again with updated weights and skip_gptq=True
                out = cls.__torch_function__(func, types, (args[0], DQ.cpu(), *args[2:]), kwargs, skip_gptq=True)
                print(args[0].debug)
                if args[0].debug:
                    act = args[0].values[0].to("cuda")
                    bias = args[2].values[0].to("cuda") if args[2] is not None else args[2]

                    new_out = out.values[0].cpu()
                    old_out = cls.__torch_function__(func, types, (act, args[1].values[0], bias), kwargs, skip_gptq=True).values[0].cpu()

                    DQ_after = cls.dequantize_func(Q, all_qparams).to(W.dtype)
                    print(
                        "SQNR for QDQ (this should be inf)", SQNR(DQ, DQ_after)
                    )  # matches
                    print(
                        "SQNR for weight (can be low)", SQNR(W, DQ.cuda())
                    )  # fine to not match
                    print(
                        "SQNR for output with GPTQ (hopefully 35+)",
                        SQNR(old_out, new_out)
                    )
                    
                    DQ_from_qtensor = qtensor.dequantize()
                    qtensor_out = torch.nn.functional.linear(act, qtensor, bias).cpu()
                    print("SQNR for output from qtensor vs output from DQ (should be high)", SQNR(qtensor_out, new_out))
                    print("SQNR for DQ vs DQ from qtensor (should be inf)", SQNR(DQ, DQ_from_qtensor))

                    qparams2 = cls.get_qparams_func(W)
                    Q2 = cls.quantize_func(W, qparams2)
                    DQ2 = cls.dequantize_func(Q2, qparams2).to(W.dtype)
                    old_q_out = cls.__torch_function__(func, types, (act, DQ2, bias), kwargs, skip_gptq=True).values[0].cpu()
                    
                    print(
                        "SQNR for output without GPTQ (should be less than above)",
                        SQNR(old_out, old_q_out)
                    )
                unpad(flat_args, orig_counts=orig_counts, force=True)
                return out
            else:
                # we padded each of the MultiTensors to match the largest multitensor so that if we had in place ops, we would be able
                # to store the many changed value and have those updates be reflected in the model. However if there are no in place ops, then
                # we just increased the size of all parameters/buffers by n times for no reason. To avoid issues, go back and unpad
                # everything where possible. i.e. all the multi tensor values are the same. We already checked for mutations and 
                # if we detected them, we updated NON_IN_PLACE_OPS to be False, so we can just check that see if we need
                # to be careful during unpadding.
                unpad(flat_args, orig_counts=orig_counts, force=(not isinstance(NON_IN_PLACE_OPS[func], bool)))

                grouped_outputs = [tree_flatten(x)[0] for x in outputs]
                out_spec = tree_flatten(outputs[0])[1]
                # conslidate out into MultiTensors [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
                flat_outputs, non_tensors_equal = grouped_to_flat(grouped_outputs)
                assert non_tensors_equal, (
                    f"ERR: found a function in model: {func} which "
                    +"caused an error in GPTQ MultiTensor, the function dispatch only works for functions"
                    +" with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
                )
                final_out = tree_unflatten(flat_outputs, out_spec)
                return final_out

                
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
            scale = qparams[0]
            zero_point = qparams[1]

            # copied from quant_api apply_int4_weight_only_quant (this should probably be made into a utility fn at some point)
            # mapping_type = MappingType.ASYMMETRIC
            block_size = (1, group_size)
            target_dtype = torch.int32
            quant_min = 0
            quant_max = 15
            zero_point_domain = ZeroPointDomain.FLOAT
            _layout = TensorCoreTiledLayout(inner_k_tiles=8)
            # at least the big up to here should be a util

            quantized_tensor = to_affine_quantized_intx_static(
                weight,
                scale=scale,
                zero_point=zero_point,
                block_size=block_size, 
                target_dtype=target_dtype, 
                quant_min=quant_min, 
                quant_max=quant_max, 
                zero_point_domain=zero_point_domain, 
                _layout=_layout, 
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

        model = _replace_with_custom_fn_if_matches_filter(
            model=model,
            replacement_fn=remove_multitensors_from_buffers_and_params,
            filter_fn=lambda x, y: True
        )
        remove = [k for k in state_dict if "kv_cache" in k]
        for k in remove:
            del state_dict[k]

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
