# For backward compatibility only
# These will be removed in the future

from .linear import (
    Int8DynActInt4WeightQATQuantizer as Int8DynActInt4WeightQATQuantizerModuleSwap,
    Int4WeightOnlyQATQuantizer as Int4WeightOnlyQATQuantizerModuleSwap,
    enable_8da4w_fake_quant as enable_8da4w_fake_quant_module_swap,
    disable_8da4w_fake_quant as disable_8da4w_fake_quant_module_swap,
    enable_4w_fake_quant as enable_4w_fake_quant_module_swap,
    disable_4w_fake_quant as disable_4w_fake_quant_module_swap,
)
