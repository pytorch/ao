from transformers import AutoConfig
from diffusers import DiffusionPipeline
import warnings
import torch

def get_llm_mm_shapes(model_name, seq_len=512):
    """Extracts matrix shapes for matrix multiplications in attention and feed-forward layers for an LLM model."""
    config = AutoConfig.from_pretrained(model_name)

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)  # Typically 4x hidden size

    d_head = hidden_size // num_attention_heads

    matrix_shapes = {
        "Attention mm": (seq_len, seq_len, d_head),  # Attention score matrix per head
        "Input -> Intermediate": (seq_len, hidden_size, intermediate_size),  # Feed-forward layer matrix multiplication shapes
        "Intermediate -> Output": (seq_len, intermediate_size, hidden_size),  # Feed-forward layer matrix multiplication shapes
    }

    return matrix_shapes.items()

def get_diffusion_mm_shapes(model_name, spatial_dim=64):
    """Extracts matrix shapes for attention and convolution operations in a diffusion model."""
    try:
        model = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    except Exception as e:
        warnings.warn(f"Could not load {model_name}: {e}")
        return None
    print(f"Loaded {model_name}, model: {model}")
    # Example for attention layers (Q, K, V)
    hidden_size = model.config.block_out_channels[-1]  # Highest channel dimension in the U-Net
    num_attention_heads = getattr(model.config, "num_heads", 8)  # Typically defined or defaulted
    d_head = hidden_size // num_attention_heads

    # Attention shapes
    attention_shapes = {
        "QK^T": (spatial_dim * spatial_dim, spatial_dim * spatial_dim),  # Flattened spatial dimensions
        "Attention * V": (spatial_dim * spatial_dim, d_head)
    }

    # Convolution shapes
    conv_shapes = [
        (in_ch, out_ch, kernel, kernel)
        for in_ch, out_ch in zip(model.config.in_channels, model.config.block_out_channels)
        for kernel in [3]  # Assuming 3x3 convolutions commonly used in diffusion models
    ]
    
    return {
        "attention_shapes": attention_shapes,
        "conv_shapes": conv_shapes
    }

# Example usage for extracting and printing matrix shapes
llm_model_names = ["bert-base-uncased", "gpt2", "t5-small", "meta-llama/Llama-3.2-3B", "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"]
diffusion_model_names = ["stabilityai/stable-diffusion-3.5-large"]  # Example diffusion model

# Get LLM matrix shapes
for model_name in llm_model_names:
    print(f"\nMatrix shapes for LLM '{model_name}':")
    shapes = get_llm_mm_shapes(model_name)
    print("shapes:", shapes)

# Get diffusion model matrix shapes
for model_name in diffusion_model_names:
    print(f"\nMatrix shapes for Diffusion model '{model_name}':")
    shapes = get_diffusion_mm_shapes(model_name)
    print("Attention Shapes:", shapes["attention_shapes"])
    print("Convolution Shapes:", shapes["conv_shapes"])
# diffusion_shapes = get_diffusion_mm_shapes(diffusion_model_name)
# if diffusion_shapes:
#     print("Attention Shapes:", diffusion_shapes["attention_shapes"])
#     print("Convolution Shapes:", diffusion_shapes["conv_shapes"])
