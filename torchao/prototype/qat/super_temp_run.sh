MAX_STEPS=100 QAT_IMPL="tensor_subclass" bash torchao/prototype/qat/temp_run.sh
MAX_STEPS=100 QAT_IMPL="module_swap" bash torchao/prototype/qat/temp_run.sh
MAX_STEPS=200 QAT_IMPL="tensor_subclass" bash torchao/prototype/qat/temp_run.sh
MAX_STEPS=200 QAT_IMPL="module_swap" bash torchao/prototype/qat/temp_run.sh
