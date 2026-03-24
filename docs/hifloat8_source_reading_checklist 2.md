# HiFloat8 源码阅读 Checklist

本清单适合首次阅读 TorchAO 中 HiFloat8 适配的同学，按从“能运行”到“能改动”的顺序梳理。

## 1. 基础入口（能跑起来）

- [X] 阅读 `/Users/wangzhenwu/Desktop/code/GitHub/ao/docs/hifloat8_usage.md`
- [X] 在 NPU 上运行 `test_hifloat8_npu.py` 的基础用例

## 2. dtype 识别与边界（不踩坑）

- [X] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/hifloat8_utils.py`
  - [X] `is_hifloat8_dtype`
  - [X] `_looks_like_hifloat8_dtype`
  - [X] `to_hifloat8`
- [X] 理解为何 `torch_npu.hifloat8` 不是 `torch.dtype`
- [X] 确认 hifloat8 不应直接走 `torch.finfo`

## 3. scale 计算（核心计算路径）

- [X] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/quant_primitives.py`
  - [X] `_choose_scale_float8_impl`
  - [X] `_get_float8_max`
  - [X] `_quantize_affine_float8` / `_dequantize_affine_float8`
- [X] 理解为何 hifloat8 必须走 Python impl，避开 torch.library schema

## 4. 训练路径（Float8TrainingTensor）

- [X] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_scaling_utils.py`
  - [ ] `hp_tensor_to_float8_dynamic`
- [X] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_training_tensor.py`
  - [X] `Float8TrainingTensor.__torch_dispatch__`
  - [X] `to_fp8_saturated` 调用链

## 5. 算子路径（NPU 内核）

- [X] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_ops.py`
  - [X] `_hif8_npu_matmul`
  - [ ] `float8_mm / float8_addmm`
- [ ] 理解 `torch_npu.npu_quant_matmul` 的输入格式

## 6. 推理/量化路径

- [ ] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/quantize_/workflows/float8/float8_tensor.py`
  - [ ] `Float8Tensor.from_hp`
- [ ] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/qat/fake_quantizer.py`
  - [ ] `Float8FakeQuantizer.forward`

## 7. 编译/图模式路径

- [ ] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/pt2e/_affine_quantization.py`
- [ ] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/utils.py` 中 `_register_custom_op`

## 8. 测试与验证

- [ ] 看 `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/tests/test_hifloat8_npu.py`
  - [ ] 实测 `npu_quant_matmul` 误差评估
  - [ ] 覆盖 `Float8Tensor` / `FakeQuantizer`

---

阅读建议：每完成一个模块，尝试在 NPU 环境里复现该模块对应的最小例子。完成本 checklist 后，你应能定位 HiFloat8 的关键入口、dtype 陷阱以及 NPU 内核调用路径。
