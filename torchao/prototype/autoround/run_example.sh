# Run examples
python autoround_llm.py -m meta-llama/Llama-2-7b-chat-hf
python autoround_llm.py -m meta-llama/Llama-2-7b-chat-hf --quant_lm_head
python autoround_llm.py -m meta-llama/Meta-Llama-3-8B-Instruct --model_device cpu
python autoround_llm.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --model_device cpu

# Evaluate with lm-eval
# Auto-round
python eval_autoround.py -m meta-llama/Llama-2-7b-chat-hf --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu
python eval_autoround.py -m meta-llama/Meta-Llama-3-8B-Instruct --model_device cpu --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu
python eval_autoround.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --model_device cpu --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu
# wo_int4
python eval_autoround.py -m meta-llama/Llama-2-7b-chat-hf --woq_int4 --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu
python eval_autoround.py -m meta-llama/Meta-Llama-3-8B-Instruct --woq_int4 --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu
python eval_autoround.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --woq_int4 --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu 
# uintx
python eval_autoround.py -m /models/Meta-Llama-3.1-8B-Instruct/ --uintx --bits 2 --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu