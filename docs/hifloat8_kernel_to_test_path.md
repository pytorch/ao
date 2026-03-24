# 从 add kernel 到 test 的开发路径（HiFloat8）

本路径面向要新增/修改 HiFloat8 相关内核或算子链路的开发者，给出端到端的可执行步骤。目标是确保新增内核能正确接入 TorchAO，并通过 NPU 实测验证数值合理性。

## 0) 需求确认
- 明确目标算子：`mm` / `addmm` / `linear` / `conv` 等
- 明确数据类型：`torch_npu.hifloat8` 是否参与
- 明确输出精度：fp16 / bf16 / fp32

## 1) 内核接入点
**HiFloat8 训练/推理统一入口：**
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_ops.py`
  - `_hif8_npu_matmul`：当前 HiFloat8 的 NPU 内核调用位置
  - `float8_mm / float8_addmm`：dispatch 到 NPU 或常规路径

如果是新增内核：
- 在 `_hif8_npu_matmul` 中扩展参数或改调用
- 或新增类似 `_hif8_npu_addmm` 并在 `float8_addmm` 分发

## 2) dtype/scale 适配
**避免 torch.library schema 转换导致 dtype 失真：**
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/hifloat8_utils.py`
  - `is_hifloat8_dtype`
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/quant_primitives.py`
  - `_choose_scale_float8_impl`
  - `_get_float8_max`

如果新增算子涉及 scale 计算或量化：
- hifloat8 必须走 `_choose_scale_float8_impl`
- 不要直接把 `torch_npu.hifloat8` 传入 `torch.library` dtype 参数

## 3) 接入训练路径
**Float8TrainingTensor 路由**
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_training_tensor.py`
  - `__torch_dispatch__` 触发 `FLOAT8_OPS_TABLE`
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/float8/float8_ops.py`
  - `FLOAT8_OPS_TABLE` 注册新 op

步骤：
1. 在 `float8_ops.py` 添加新 op handler
2. 将 handler 注册到 `FLOAT8_OPS_TABLE`
3. 确保 `Float8TrainingTensor.__torch_dispatch__` 可命中该 handler

## 4) 推理/量化路径接入（如有）
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/quantize_/workflows/float8/float8_tensor.py`
  - `Float8Tensor.from_hp`
- `/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/quantization/qat/fake_quantizer.py`
  - `Float8FakeQuantizer`

若算子影响推理或 QAT：
- 确保 hifloat8 路径不经过 torch.library dtype 解析

## 5) 写 NPU 实测用例
**必备测试：**
文件：`/Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/tests/test_hifloat8_npu.py`

建议新增：
- 功能性测试（能跑通）
- 数值误差测试（相对误差/最大误差阈值）

示例误差测试步骤：
1. 生成 `Float8TrainingTensor`（hifloat8）
2. 走真实 `torch.mm` -> NPU 内核
3. 用反量化后浮点值计算 reference
4. 对比误差，设置可接受阈值

## 6) 本地构建与 CI
- 构建脚本：`/Users/wangzhenwu/Desktop/code/GitHub/ao/ci/build.sh`
  - `bash ci/build.sh --python=3.10`
- 运行测试：
```bash
python -m pytest /Users/wangzhenwu/Desktop/code/GitHub/ao/torchao/tests/test_hifloat8_npu.py -k hifloat8
```

## 7) 常见问题
- **dtype 变成 `torch.uint5`**  
  说明你把 `torch_npu.hifloat8` 传入了 torch.library dtype 参数，需改走 Python impl。
- **误差过大**  
  检查 scale 计算、输入范围、以及 reference 计算是否使用反量化浮点值。

---

按此路径走，你可以做到“新增/修改 NPU 内核”并完整验证集成效果。如果你希望我提供具体的代码模板（例如新增 `float8_bmm` 的处理），告诉我目标算子即可。
