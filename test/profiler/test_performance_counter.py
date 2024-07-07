import pytest

# Skip if transformers is not installed
transformers = pytest.importorskip("transformers")
LlamaConfig = transformers.models.llama.modeling_llama.LlamaConfig
LlamaForCausalLM = transformers.models.llama.modeling_llama.LlamaForCausalLM
import torch

from torchao.profiler.performance_counter import PerformanceCounterMode
from torchao.utils import TORCH_VERSION_AFTER_2_5


def get_leaf_nodes(count_keys, module_name):
    return [k for k in count_keys if k.endswith(module_name)]

def attn_proj_io_check(model_config, batch_size, seqlen, element_size):
    input_size = batch_size * seqlen * model_config.hidden_size * element_size 
    weight_size = model_config.hidden_size * model_config.hidden_size * element_size 
    output_size = batch_size * seqlen * model_config.hidden_size * element_size
    return input_size + weight_size + output_size
def attn_io_check(model_config, batch_size, seqlen, element_size):
    # queries, keys, values -> factor of 3
    input_size = (batch_size * seqlen * model_config.hidden_size * 3) * element_size
    output_size = (batch_size * seqlen * model_config.hidden_size) * element_size
    return input_size + output_size 
    
def ffn_io_check(model_config, batch_size, seqlen, element_size, module_name):
    assert module_name in ["up_proj", "gate_proj", "down_proj"]

    if module_name == "down_proj":
        input_size = batch_size * seqlen * model_config.intermediate_size * element_size
    else:
        input_size = batch_size * seqlen * model_config.hidden_size * element_size
    weight_size = model_config.hidden_size * model_config.intermediate_size * element_size 
    if module_name == "down_proj":
        output_size = batch_size * seqlen * model_config.hidden_size * element_size
    else:
        output_size = batch_size * seqlen * model_config.intermediate_size * element_size 

    return input_size + weight_size + output_size


CONFIG_7B = (32, 4096, 11008, 32, 32000)
MEDIUM_CONFIG = [p // 2 for p in CONFIG_7B]
SMALL_CONFIG = [p // 4 for p in CONFIG_7B]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not TORCH_VERSION_AFTER_2_5, reason="requires torch >= 2.5")
@pytest.mark.parametrize("num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size", [MEDIUM_CONFIG, SMALL_CONFIG])
@pytest.mark.parametrize("batch_size, seqlen", [(1, 128),])
@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda p: str(p))
def test_performance_counter(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size, batch_size, seqlen, dtype):

    cfg = LlamaConfig(num_hidden_layers=num_hidden_layers, 
                      hidden_size=hidden_size,
                      intermediate_size=intermediate_size, 
                      num_attention_heads=num_attention_heads, 
                      vocab_size=vocab_size)

    # Note we set some options manually since the model doesn't seem to be initialized correctly
    # when these options are set in LlamaConfig
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).to(dtype).to("cuda")
    model_config = model.config
    element_size = dtype.itemsize
    
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seqlen), device="cuda")
    with torch.no_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            with PerformanceCounterMode() as perf_counter:
                _ = model(input_ids)
    
    summary_flops = perf_counter.get_summary_flop_counts()
    summary_io = perf_counter.get_summary_io_counts()
    flops_by_op = perf_counter.get_flop_counts()
    io_by_op = perf_counter.get_io_counts()
    assert len(summary_flops) == len(summary_io)
    assert summary_flops.keys() == summary_io.keys()

    # Attn Projections
    for k in ["q_proj", "k_proj", "v_proj"]:
        # Flops check
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.hidden_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io movement check
        expected_size = attn_proj_io_check(model_config, batch_size, seqlen, element_size)
        assert expected_size == summary_io[proj_keys[0]]

    # Attention
    attention_keys = get_leaf_nodes(summary_flops.keys(), "self_attn")
    for k in attention_keys:
        flops = flops_by_op[k]
        io_movement = io_by_op[k]
        for op, count in flops.items():
            if "attention" in op.__name__: 
                expected_flops = 2 * 2 * batch_size * seqlen * seqlen * model_config.hidden_size
                assert expected_flops == count
        for op, count in io_movement.items():
            if "attention" in op.__name__:
                # Check approx equal due to other small artifacts returned by sdpa.mem_efficient_attention
                # See #https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention.cu#L867
                # Check within 100 bytes
                expected_size = attn_io_check(model_config, batch_size, seqlen, element_size)
                assert abs(expected_size - count) < 100
    # FFN
    for k in ["up_proj", "gate_proj", "down_proj"]:
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.intermediate_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io movement check
        expected_size = ffn_io_check(model_config, batch_size, seqlen, element_size, k)
        assert expected_size == summary_io[proj_keys[0]]
