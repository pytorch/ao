from torchao.quantization.qat.linear import (
    disable_4w_fake_quant,
    disable_8da4w_fake_quant,
    enable_4w_fake_quant,
    enable_8da4w_fake_quant,
    FakeQuantizedLinear,
    Int4WeightOnlyQATLinear,
    Int4WeightOnlyQATQuantizer,
    Int8DynActInt4WeightQATLinear,
    Int8DynActInt4WeightQATQuantizer,
)
