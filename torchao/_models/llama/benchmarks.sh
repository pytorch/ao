export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

# README BENCHMARKS
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt


# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 8
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 8


# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 32
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 32


# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt


# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 8

# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 32

# export MODEL_REPO=meta-llama/Meta-Llama-3-8B
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt


# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 8
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 8

# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 32
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 32



# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt


# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 8
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 8
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 8

# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --write_result benchmark_results.txt --precision float16 --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemsub-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-64  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-4-None  --write_result benchmark_results.txt --batch_size 32
# python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo  --write_result benchmark_results.txt --batch_size 32
# # python generate.py --compile --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --precision float16 --quantization gemlite-8-None  --write_result benchmark_results.txt --batch_size 32
