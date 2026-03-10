# HiFloat8 使用说明

本文介绍如何在 TorchAO 中使用 TorchNPU 的 HiFloat8（`torch_npu.hifloat8`），包含环境检查、API 用法、常见坑点与测试建议。

## 前置条件

- 已安装并可用的 `torch_npu`，且 `torch_npu.npu.is_available()` 为 True。
- 当前 PyTorch 运行环境仅启用 NPU（避免 GPU/其他加速器冲突）。

快速检查：
```python
import torch_npu  # noqa: F401
print(torch_npu.npu.is_available())
```

## 基础用法

### 1) Float8Tensor.from_hp（权重/激活量化）

```python
import torch
import torch_npu
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
from torchao.quantization.granularity import PerRow

x = torch.randn(4, 8, device="npu", dtype=torch.bfloat16)
qt = Float8Tensor.from_hp(
    x,
    float8_dtype=torch_npu.hifloat8,
    granularity=PerRow(),
)

print(qt.qdata.dtype)  # HiFloat8 tensor subclass
dq = qt.dequantize()
print(dq.shape, dq.dtype)
```

### 2) Float8TrainingTensor（训练时动态缩放）

```python
import torch
import torch_npu
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_training_tensor import GemmInputRole, LinearMMConfig
from torchao.float8.config import ScalingGranularity

x = torch.randn(4, 8, device="npu", dtype=torch.float16)
w = torch.randn(8, 16, device="npu", dtype=torch.float16)
mm_cfg = LinearMMConfig()

a = hp_tensor_to_float8_dynamic(
    x,
    torch_npu.hifloat8,
    mm_cfg,
    gemm_input_role=GemmInputRole.INPUT,
    scaling_granularity=ScalingGranularity.TENSORWISE,
)
b = hp_tensor_to_float8_dynamic(
    w,
    torch_npu.hifloat8,
    mm_cfg,
    gemm_input_role=GemmInputRole.WEIGHT,
    scaling_granularity=ScalingGranularity.TENSORWISE,
)

out = torch.mm(a, b)
print(out.shape, out.dtype, out.device)
```

### 3) QAT FakeQuantizer

```python
import torch
import torch_npu
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.qat.fake_quantizer import Float8FakeQuantizer
from torchao.quantization.granularity import PerRow

x = torch.randn(4, 8, device="npu", dtype=torch.float16)
cfg = Float8FakeQuantizeConfig(dtype=torch_npu.hifloat8, granularity=PerRow())
fq = Float8FakeQuantizer(cfg)
y = fq(x)
print(y.shape, y.dtype)
```

## 常见问题与注意事项

1. **不要把 `torch_npu.hifloat8` 直接传入 `torch.library` 自定义算子 dtype 参数**  
   它不是 `torch.dtype`，可能被错误解析为 `torch.uint5` 等，导致 `torch.finfo()` 失败。

2. **使用 `_choose_scale_float8_impl` 走 Python 侧逻辑**  
   在 hifloat8 情况下，scale 计算应走 `_choose_scale_float8_impl`，避免 schema 转换。

3. **数值误差评估要基于“反量化后的浮点值”**  
   比较 `a._data.float()/a._scale` 与 `b._data.float()/b._scale` 的乘积更合理。

## 推荐测试

测试文件：
`./torchao/tests/test_hifloat8_npu.py`

运行：
```bash
python -m pytest ./torchao/tests/test_hifloat8_npu.py -k hifloat8
```

其中包含：
- `_choose_scale_float8_impl` 的 NPU 实测
- `Float8Tensor.from_hp` 的 hifloat8 实测
- `Float8FakeQuantizer` 的 hifloat8 实测
- `torch.mm` 路径下 `npu_quant_matmul` 的数值误差检查
