def quantized_model_size_in_bytes(num_elements, group_size, bit_zeropoint, bit_scale, x, y, A, B):
    # Size for A-bit quantization layers
    size_A_bit = x * (num_elements * A + num_elements // group_size * (bit_zeropoint + bit_scale))
    
    # Size for B-bit quantization layers
    size_B_bit = y * (num_elements * B + num_elements // group_size * (bit_zeropoint + bit_scale))
    
    # Total quantized model size in bits
    total_size_bits = size_A_bit + size_B_bit
    
    # Convert to bytes
    total_size_bytes = total_size_bits / 8
    
    # Convert to gigabytes
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    return total_size_gb

# Example usage
num_elements = 250945664 #number of elements per Llama3 linear layer
group_size = 32  # Example value, please adjust as needed
bit_zeropoint = 2  # Example value, please adjust as needed
bit_scale = 2  # Example value, please adjust as needed
x = 32  # Example number of layers for A-bit quantization, adjust as needed
y = 0  # Example number of layers for B-bit quantization, adjust as needed
#A = 4  # Example bit width for A-bit quantization, adjust as needed
#B = 0  # Example bit width for B-bit quantization, adjust as needed

#for b in [8]:
#    model_size_bytes = quantized_model_size_in_bytes(num_elements, group_size, bit_zeropoint, bit_scale, 32, 0, b, 0)
#    print(f"The quantized model size for {b} bits is {model_size_bytes} GB")

for (x,y) in [(16,8),(16,6),(16,5),(16,4),(16,3),(16,2),(8,6),(8,5),(8,4),(8,3),(8,2),(6,5),(6,4),(6,3),(6,2),(5,4),(5,3),(5,2), (4,3),(4,2),(3,2)]:
    model_size_bytes = quantized_model_size_in_bytes(num_elements, group_size, bit_zeropoint, bit_scale, 5, 27, x, y)
    print(f"The quantized model size for {b} bits is {model_size_bytes} GB")
