export MODEL_REPO=mistralai/Mixtral-8x7B-Instruct-v0.1
export CHECKPOINT_PATH=checkpoints/

######### GROUPED_MM #######

# noquant
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant --compile --compile_mode "max-autotune"

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant --compile --compile_mode "max-autotune"

# scaled_grouped_mm
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq-base --compile --compile_mode "max-autotune"

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq-base --compile --compile_mode "max-autotune"

# ######### MULTI TOKEN #######

# noquant
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant --compile --decompose_grouped_mm

# int8wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8wo-base --compile --decompose_grouped_mm

# needs balanced tokens due to minimum matmul sizes
# int8dq-base
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8dq-base --compile --decompose_grouped_mm

# int4wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base --compile --decompose_grouped_mm

# fp8wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8wo-base --compile --decompose_grouped_mm

# fp8dq-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq-base --compile --decompose_grouped_mm


######### SINGLE TOKEN #######

# einsum
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --compile

#noquant
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant --compile --decompose_grouped_mm

# int8wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8wo-base --compile --decompose_grouped_mm

# int4wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int4wo-base --compile --decompose_grouped_mm

# fp8wo-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8wo-base --compile --decompose_grouped_mm

# fp8dq-base
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq-base --compile --decompose_grouped_mm

########## ARM ##########
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --device cpu
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --compile --device cpu
