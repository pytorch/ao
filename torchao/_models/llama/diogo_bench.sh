export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS

# OTHER BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization gptq-marlin --precision float16 --write_result benchmark_results.txt
