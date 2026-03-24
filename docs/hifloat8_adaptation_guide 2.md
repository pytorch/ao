# HiFloat8 (TorchNPU) 适配指南

本文基于近期多次改动总结 HiFloat8 在 torchao 中的适配要点，强调 TorchNPU 与 PyTorch dtype 体系差异带来的坑点、推荐实现方式、以及对应的 NPU 实测用例。

**背景与核心问题**
- `torch_npu.hifloat8` 不是 `torch.dtype`，而是 TorchNPU 自定义的 dtype 枚举。
- 当该 dtype 通过 `torch.library` 自定义算子 schema 传递时，会被 PyTorch 强制解析为某个 `ScalarType`，出现错误映射（例如 `torch.uint5`），从而触发 `torch.finfo()` 报错。
- 解决原则是：**不要让 `torch_npu.hifloat8` 走 `torch.library` schema 的 dtype 参数转换**，改用 Python 实现的逻辑路径。

**关键适配点**
- dtype 识别统一通过 `torchao/float8/hifloat8_utils.py`。
- 量化尺度计算绕过自定义 op，使用 `_choose_scale_float8_impl`。
- 量化/反量化使用 HiFloat8 的最大绝对值与上下界。
- matmul 走 `torch_npu.npu_quant_matmul`，并以误差测试验证数值合理性。

**实现落点（模块级）**
- `torchao/float8/hifloat8_utils.py`
  - `is_hifloat8_dtype`：不依赖 `torch.dtype`，兼容 TorchNPU 自定义 dtype。
  - `_looks_like_hifloat8_dtype`：通过字符串判断兜底识别。
  - `to_hifloat8`：调用 TorchNPU HiFloat8 内部 cast。
- `torchao/quantization/quant_primitives.py`
  - `_get_float8_max`：对 hifloat8 走 `hifloat8_max_abs()`。
  - `_choose_scale_float8_impl`：Python 侧实现，避免 torch.library dtype 转换。
  - `_choose_scale_float8`：仍保留自定义 op，但调用 impl。
- `torchao/quantization/quantize_/workflows/float8/float8_tensor.py`
  - `Float8Tensor.from_hp`：hifloat8 走 `_choose_scale_float8_impl`，其他 fp8 走 `_choose_scale_float8`。
- `torchao/quantization/qat/fake_quantizer.py`
  - `Float8FakeQuantizer`：hifloat8 走 `_choose_scale_float8_impl`。
- `torchao/float8/float8_ops.py`
  - `_hif8_npu_matmul`：真实调用 `torch_npu.npu_quant_matmul`。

**开发与适配建议**
- 任何 **通过 torch.library 暴露的自定义 op** 都不要直接接收 `torch_npu.hifloat8` 作为 `torch.dtype` 参数。
- 对于 hifloat8 的 scale 计算，统一调用 `_choose_scale_float8_impl`。
- 若新增 float8 量化逻辑，请优先复用：
  - `hifloat8_max_abs()` / `hifloat8_min_max()`
  - `to_hifloat8()` / `is_hifloat8_dtype()`
- 误差评估建议：对比 “反量化后浮点值” 的 matmul 结果，而不是原始高精度输入。

**NPU 实测用例**
测试文件：`./torchao/tests/test_hifloat8_npu.py`

覆盖点：
- `_choose_scale_float8_impl` 在 NPU + hifloat8 下能正确运行。
- `Float8Tensor.from_hp` 可生成 HiFloat8 张量并可反量化。
- `Float8FakeQuantizer` 在 NPU + hifloat8 下可执行。
- `torch.mm(a, b)` 实测 npu_quant_matmul 数值误差。

运行方式：
```bash
python -m pytest ./torchao/tests/test_hifloat8_npu.py -k hifloat8
```

**数值误差检查建议**
示例测试在 `test_hifloat8_npu_matmul_numerics` 中：
- 输入范围缩小到 `0.1` 量级，减少溢出影响。
- 参考输出使用 `a._data.float()/a._scale` 与 `b._data.float()/b._scale` 的浮点乘积。
- 参考阈值：
  - 相对误差 `< 0.15`
  - 最大绝对误差 `< 0.5`

如果未来硬件或底层内核升级导致误差分布变化，可基于统计分位数重新调整阈值。
