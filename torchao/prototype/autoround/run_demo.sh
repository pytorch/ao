# If we only have cpu
python autoround_demo.py -m /models//models/Llama-2-7b-chat-hf/  --device cpu
# If we have gpu
python autoround_demo.py -m /models//models/Llama-2-7b-chat-hf/  --device cuda
python autoround_demo.py -m /models/Meta-Llama-3.1-8B-Instruct/  --device cpu --speedup_optimization
