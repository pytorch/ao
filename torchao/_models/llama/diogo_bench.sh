export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS

# OTHER BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization gptq-marlin-128 --precision float16 --calibration_limit 1 --limit 1
# python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-128-gptq --calibration_limit 1 --limit 1