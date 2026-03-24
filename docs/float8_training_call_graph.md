# Float8 训练与推理示例调用关系图

本文基于当前仓库里的实现，对下面这段代码做“按源码推演”的模拟运行说明，不依赖真实 CUDA 执行结果：

```python
import torch
from torch import nn
import torch.nn.functional as F

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8 import convert_to_float8_training

m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
    nn.Linear(128, 1),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

def module_filter_fn(mod: torch.nn.Module, fqn: str):
    if fqn == "1":
        return False
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

convert_to_float8_training(m, module_filter_fn=module_filter_fn)
```

## 1. 先给结论：这段代码只会把第 0 层换成 `Float8Linear`

| FQN | 原始模块 | 过滤结果 | 转换后 |
| --- | --- | --- | --- |
| `0` | `Linear(2048, 4096)` | 通过，输入/输出维度都能被 16 整除 | `Float8Linear` |
| `1` | `Linear(4096, 128)` | 不通过，因为 `fqn == "1"` | `nn.Linear` |
| `2` | `Linear(128, 1)` | 不通过，因为 `out_features == 1` 不能被 16 整除 | `nn.Linear` |

注意：注释 `# don't convert the last module` 和实际代码不一致。  
在 `nn.Sequential` 里，最后一层的 FQN 是 `2`，不是 `1`。所以这里跳过的是中间层，不是最后一层。

因此，转换后的模型等价于：

```python
m = nn.Sequential(
    Float8Linear(2048, 4096),
    nn.Linear(4096, 128),
    nn.Linear(128, 1),
)
```

补充两点：

- 两个 `convert_to_float8_training` import 最终指向同一个实现，后面的 `from torchao.float8 import convert_to_float8_training` 会覆盖前一个绑定。
- `optimizer` 虽然在模块替换前创建，但仍然有效，因为 `Float8Linear.from_float(...)` 复用了原始 `weight` / `bias` 参数对象。

## 2. 模块替换阶段调用图

```mermaid
flowchart TD
    A["convert_to_float8_training(m, module_filter_fn=...)"] --> B["swap_linear_layers(...)"]
    B --> C["post_order_traversal(root=Sequential)"]
    C --> D["访问子模块 FQN=0: Linear(2048, 4096)"]
    C --> E["访问子模块 FQN=1: Linear(4096, 128)"]
    C --> F["访问子模块 FQN=2: Linear(128, 1)"]

    D --> D1["module_filter_fn -> True"]
    D1 --> D2["Float8Linear.from_float(mod, config=Float8LinearConfig())"]
    D2 --> D3["复用原始 weight / bias"]

    E --> E1["module_filter_fn -> False"]
    E1 --> E2["保留为 nn.Linear"]

    F --> F1["module_filter_fn -> False"]
    F1 --> F2["保留为 nn.Linear"]
```

`Float8LinearConfig()` 默认等价于 tensorwise recipe，关键点是：

- `input` / `weight` 动态转成 float8，默认目标 dtype 是 `e4m3`
- `grad_output` 动态转成 float8，默认目标 dtype 是 `e5m2`
- forward 对应的 output GEMM 开启 `use_fast_accum=True`
- `pad_inner_dim=False`
- `emulate=False`

## 3. 一次训练迭代的模拟执行

### 3.1 前向阶段

```text
optimizer.zero_grad()

output = m(x)
  -> nn.Sequential.forward
    -> layer[0]: Float8Linear.forward(x)
      -> matmul_with_hp_or_float8_args.apply(
           input=x,
           weight_t=self.weight.t(),
           linear_mm_config=self.linear_mm_config,
           config=self.config,
         )
        -> matmul_with_hp_or_float8_args.forward
          -> hp_tensor_to_float8_dynamic(input_hp, role=INPUT)
            -> tensor_to_scale(...)
            -> hp_tensor_and_scale_to_float8(...)
              -> _ToFloat8ConstrFunc.forward(...)
                -> Float8TrainingTensor(data, scale, orig_dtype=bf16, role=INPUT)
          -> hp_tensor_to_float8_dynamic(weight_hp_t, role=WEIGHT)
            -> tensor_to_scale(...)
            -> hp_tensor_and_scale_to_float8(...)
              -> _ToFloat8ConstrFunc.forward(...)
                -> Float8TrainingTensor(data, scale, orig_dtype=bf16, role=WEIGHT)
          -> torch.mm(fp8_input_2d, fp8_weight_t)
            -> Float8TrainingTensor.__torch_dispatch__
              -> FLOAT8_OPS_TABLE[aten.mm.default]
                -> float8_mm
                  -> preprocess_addmm
                    -> choose_scaled_mm_config(INPUT, WEIGHT)
                    -> padding / layout / granularity 对齐
                  -> addmm_float8_unwrapped
                    -> torch._scaled_mm(...)
          -> reshape 回原始 batch 形状
      -> output + bias
    -> layer[1]: nn.Linear.forward(...)
    -> layer[2]: nn.Linear.forward(...)

fake_labels = torch.ones_like(output)
loss = F.mse_loss(output, fake_labels)
```

这一段里，第 0 层会走 Float8 路径，第 1、2 层仍然是普通 PyTorch `Linear` 路径。

### 3.2 反向阶段

```text
loss.backward()
  -> layer[2] 普通 nn.Linear backward
  -> layer[1] 普通 nn.Linear backward
  -> layer[0] matmul_with_hp_or_float8_args.backward
    -> 计算 grad_input
      -> hp_tensor_to_float8_dynamic(grad_output, role=GRAD_OUTPUT)
      -> hp_tensor_to_float8_dynamic(weight_t, role=WEIGHT)
      -> weight_t_fp8.t()
        -> Float8TrainingTensor.__torch_dispatch__
          -> float8_transpose
      -> torch.mm(fp8_grad_output, fp8_weight)
        -> Float8TrainingTensor.__torch_dispatch__
          -> float8_mm
            -> choose_scaled_mm_config(GRAD_OUTPUT, WEIGHT)
            -> preprocess_addmm
            -> torch._scaled_mm(...)
    -> 计算 grad_weight
      -> hp_tensor_to_float8_dynamic(grad_output, role=GRAD_OUTPUT)
      -> hp_tensor_to_float8_dynamic(input, role=INPUT)
      -> grad_output_fp8.t()
        -> Float8TrainingTensor.__torch_dispatch__
          -> float8_transpose
      -> torch.mm(fp8_grad_output_t, fp8_input)
        -> Float8TrainingTensor.__torch_dispatch__
          -> float8_mm
            -> choose_scaled_mm_config(GRAD_OUTPUT, INPUT)
            -> preprocess_addmm
            -> torch._scaled_mm(...)

optimizer.step()
```

## 4. 合并后的主调用关系图

```mermaid
flowchart TD
    A["训练循环"] --> B["optimizer.zero_grad()"]
    B --> C["output = m(x)"]

    C --> D["Sequential[0] = Float8Linear.forward"]
    D --> E["matmul_with_hp_or_float8_args.forward"]
    E --> F["hp_tensor_to_float8_dynamic(input, INPUT)"]
    E --> G["hp_tensor_to_float8_dynamic(weight.t, WEIGHT)"]
    F --> H["Float8TrainingTensor(INPUT)"]
    G --> I["Float8TrainingTensor(WEIGHT)"]
    H --> J["torch.mm(...)"]
    I --> J
    J --> K["Float8TrainingTensor.__torch_dispatch__"]
    K --> L["float8_mm"]
    L --> M["preprocess_addmm"]
    M --> N["choose_scaled_mm_config(INPUT, WEIGHT)"]
    M --> O["padding / layout / granularity 对齐"]
    L --> P["addmm_float8_unwrapped"]
    P --> Q["torch._scaled_mm(...)"]
    Q --> R["+ bias"]

    C --> S["Sequential[1] = nn.Linear.forward"]
    C --> T["Sequential[2] = nn.Linear.forward"]
    R --> U["F.mse_loss(...)"]
    S --> U
    T --> U

    U --> V["loss.backward()"]
    V --> W["matmul_with_hp_or_float8_args.backward"]

    W --> X["grad_input 路径"]
    X --> X1["hp_tensor_to_float8_dynamic(grad_output, GRAD_OUTPUT)"]
    X --> X2["hp_tensor_to_float8_dynamic(weight_t, WEIGHT)"]
    X --> X3["torch.mm(...)"]
    X3 --> X4["__torch_dispatch__ -> float8_mm"]
    X4 --> X5["choose_scaled_mm_config(GRAD_OUTPUT, WEIGHT)"]
    X4 --> X6["torch._scaled_mm(...)"]

    W --> Y["grad_weight 路径"]
    Y --> Y1["hp_tensor_to_float8_dynamic(grad_output, GRAD_OUTPUT)"]
    Y --> Y2["hp_tensor_to_float8_dynamic(input, INPUT)"]
    Y --> Y3["grad_output_fp8.t()"]
    Y3 --> Y4["__torch_dispatch__ -> float8_transpose"]
    Y --> Y5["torch.mm(...)"]
    Y5 --> Y6["__torch_dispatch__ -> float8_mm"]
    Y6 --> Y7["choose_scaled_mm_config(GRAD_OUTPUT, INPUT)"]
    Y6 --> Y8["torch._scaled_mm(...)"]

    V --> Z["optimizer.step()"]
    Z --> AA["更新三层原始 Parameter"]
    AA --> AB["torch.save(..., 'checkpoint.pth')"]
```

## 5. 和示例图相比，哪些点需要特别说明

### 5.1 这段代码里的 Float8 主链命中的是 `float8_mm`，不是 `float8_addmm`

`Float8Linear.forward()` 当前实现是：

1. 先调用 `matmul_with_hp_or_float8_args.apply(...)`
2. 在里面执行 `torch.mm(...)`
3. 回到 `Float8Linear.forward()` 后再单独做 `output + bias`

所以这段示例代码里，第 0 层的 Float8 主链实际是：

```text
Float8Linear.forward
  -> matmul_with_hp_or_float8_args.forward
    -> torch.mm(...)
      -> __torch_dispatch__
        -> float8_mm
          -> preprocess_addmm
          -> addmm_float8_unwrapped
            -> torch._scaled_mm(...)
  -> + bias
```

`float8_addmm` 是同级分支，只有当进入 `aten.addmm.default` 且输入里包含 `Float8TrainingTensor` 时才会命中。  
因此它属于同一套 dispatch 体系的一部分，但不是这段示例代码里第 0 层 forward 的主路径。

### 5.2 NPU fastpath 只在条件满足时才会命中

`float8_mm(...)` / `float8_addmm(...)` 开头都会检查：

```text
_should_use_npu_fp8_matmul(...)
```

如果返回 `True`，会改走：

```text
_npu_fp8_matmul(...)
  -> _fp8_npu_matmul(...)
```

否则才走当前更常见的：

```text
preprocess_addmm(...)
  -> addmm_float8_unwrapped(...)
    -> torch._scaled_mm(...)
```

由于示例代码使用的是 `.cuda()`，按常见场景推演，主路径通常是 `torch._scaled_mm(...)` 这一支。

## 6. 最后可直接记住的精简版调用树

```text
convert_to_float8_training
  -> swap_linear_layers
    -> Float8Linear.from_float

训练时：
Sequential.forward
  -> Float8Linear.forward
    -> matmul_with_hp_or_float8_args.forward
      -> hp_tensor_to_float8_dynamic
        -> tensor_to_scale
        -> hp_tensor_and_scale_to_float8
          -> Float8TrainingTensor
      -> torch.mm
        -> Float8TrainingTensor.__torch_dispatch__
          -> float8_mm
            -> preprocess_addmm
              -> choose_scaled_mm_config
            -> addmm_float8_unwrapped
              -> torch._scaled_mm
  -> nn.Linear.forward
  -> nn.Linear.forward

loss.backward
  -> matmul_with_hp_or_float8_args.backward
    -> torch.mm
      -> Float8TrainingTensor.__torch_dispatch__
        -> float8_mm
          -> choose_scaled_mm_config
          -> torch._scaled_mm

optimizer.step
torch.save
```

## 7. 推理示例先给结论

下面这段 inference 代码，走的不是训练态 `Float8Linear.forward()` 那条路，而是推理态 `quantize_` 路线：

```python
import torch

from torchao.float8.float8_linear import Float8Linear
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import quantize_
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)

checkpoint = torch.load("checkpoint.pth", weights_only=False)
model = checkpoint["model"]
model.load_state_dict(checkpoint["model_state_dict"])

quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    out = model(x)
    print(out)
```

对当前这个 checkpoint 按源码推演后，模块状态会变成：

| 层 | `torch.load` 后 | `quantize_` 后 | 推理时实际路径 |
| --- | --- | --- | --- |
| `0` | `Float8Linear(2048, 4096)` | `nn.Linear(weight=Float8Tensor)` | float8 inference 路径 |
| `1` | `nn.Linear(4096, 128)` | `nn.Linear(weight=Float8Tensor)` | float8 inference 路径 |
| `2` | `nn.Linear(128, 1)` | `nn.Linear(weight=bf16 Tensor)` | 普通 `nn.Linear` 路径 |

原因分别是：

- 第 0 层虽然在 checkpoint 里是训练态 `Float8Linear`，但 `quantize_` 会先把它还原成普通 `nn.Linear`，再做推理态 float8 量化。
- 第 1 层权重形状是 `(128, 4096)`，输入/输出维度都能被 16 整除，满足 `_fp8_mm_compat(...)`，所以会被量化。
- 第 2 层权重形状是 `(1, 128)`，`out_dim == 1` 不满足 float8 mm 兼容性检查，所以不会进入 float8 kernel 路线。

补充：

- `from torchao.float8.float8_linear import Float8Linear` 在这里主要是为了让 `weights_only=False` 反序列化 checkpoint 时能找到类定义。
- 下面描述的是 `Float8DynamicActivationFloat8WeightConfig(version=2)` 的默认路径，不是旧版 `AffineQuantizedTensor + Float8Layout` 路线。

## 8. `quantize_` 阶段调用图

```mermaid
flowchart TD
    A["torch.load('checkpoint.pth', weights_only=False)"] --> B["model = checkpoint['model']"]
    B --> C["model.load_state_dict(...)"]
    C --> D["quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))"]
    D --> E["_replace_with_custom_fn_if_matches_filter(...)"]

    E --> F["子模块 0: Float8Linear"]
    F --> F1["先还原成 nn.Linear<br/>复用原 weight / bias"]
    F1 --> F2["_float8_dynamic_activation_float8_weight_transform"]
    F2 --> F3["_float8_dynamic_activation_float8_weight_quantize_tensor"]
    F3 --> F4["_check_hardware_support((PerTensor, PerTensor))"]
    F4 --> F5["_fp8_mm_compat(weight) == True"]
    F5 --> F6["构造 QuantizeTensorToFloat8Kwargs(act)"]
    F6 --> F7["Float8Tensor.from_hp(weight, float8_dtype=e4m3, granularity=PerTensor, mm_config=Float8MMConfig(use_fast_accum=True))"]
    F7 --> F8["_choose_scale_float8(...)"]
    F8 --> F9["_quantize_affine_float8(...)"]
    F9 --> F10["module.weight = Parameter(Float8Tensor, requires_grad=False)"]

    E --> G["子模块 1: nn.Linear"]
    G --> G1["_float8_dynamic_activation_float8_weight_transform"]
    G1 --> G2["_fp8_mm_compat(weight) == True"]
    G2 --> G3["Float8Tensor.from_hp(weight, ...)"]
    G3 --> G4["module.weight = Parameter(Float8Tensor, requires_grad=False)"]

    E --> H["子模块 2: nn.Linear"]
    H --> H1["_float8_dynamic_activation_float8_weight_transform"]
    H1 --> H2["_fp8_mm_compat(weight) == False"]
    H2 --> H3["保留原始 bf16 weight"]
```

这一步之后，模型更接近下面这个状态：

```python
model = nn.Sequential(
    nn.Linear(2048, 4096),  # weight: Float8Tensor
    nn.Linear(4096, 128),   # weight: Float8Tensor
    nn.Linear(128, 1),      # weight: 普通 Tensor
)
```

## 9. inference forward 的主调用链

### 9.1 一次 `model(x)` 的模拟执行

```text
with torch.inference_mode():
    out = model(x)
      -> nn.Sequential.forward
        -> layer[0]: nn.Linear.forward
          -> F.linear(input, weight=Float8Tensor, bias)
            -> Float8Tensor 对 F.linear / aten.linear.default 的 override
              -> 读取 weight.act_quant_kwargs
              -> _choose_quant_func_and_quantize_tensor(input, act_quant_kwargs)
                -> Float8Tensor.from_hp(input, float8_dtype=e4m3, granularity=PerTensor)
                  -> _choose_scale_float8(...)
                  -> _quantize_affine_float8(...)
              -> 此时 input 也变成 Float8Tensor
              -> 选择 kernel
                -> 若 CUDA + SM90+ + fbgemm_gpu_genai 可用:
                  -> torch.ops.fbgemm.f8f8bf16(...)
                -> 否则:
                  -> preprocess_scale(input_scale, input_shape)
                  -> preprocess_data(input_qdata, weight_qdata.T, mm_config)
                  -> addmm_float8_unwrapped_inference(...)
                    -> 若 _should_use_npu_fp8_matmul(...):
                      -> _fp8_npu_matmul(...)
                    -> 否则:
                      -> torch._scaled_mm(...)
        -> layer[1]: nn.Linear.forward
          -> 与 layer[0] 相同的 float8 inference 路径
        -> layer[2]: nn.Linear.forward
          -> 普通 F.linear(...)
```

这段代码的 shape 演化大致是：

- 输入 `x`: `[1, 4096, 2048]`
- 第 0 层输出: `[1, 4096, 4096]`
- 第 1 层输出: `[1, 4096, 128]`
- 第 2 层输出: `[1, 4096, 1]`

### 9.2 合并后的 inference 调用关系图

```mermaid
flowchart TD
    A["with torch.inference_mode(): out = model(x)"] --> B["Sequential.forward"]

    B --> C["layer[0] = nn.Linear.forward"]
    C --> D["F.linear(x, weight=Float8Tensor, bias)"]
    D --> E["Float8Tensor override for F.linear / aten.linear.default"]
    E --> F["_choose_quant_func_and_quantize_tensor(x, act_quant_kwargs)"]
    F --> G["Float8Tensor.from_hp(x, e4m3, PerTensor)"]
    G --> H["_choose_scale_float8(...)"]
    H --> I["_quantize_affine_float8(...)"]
    I --> J["input 变成 Float8Tensor"]

    J --> K["kernel 分发"]
    K --> L["CUDA SM90+ 且 fbgemm 可用"]
    L --> M["torch.ops.fbgemm.f8f8bf16(...)"]
    K --> N["否则走 torch kernel"]
    N --> O["preprocess_scale(...)"]
    O --> P["preprocess_data(...)"]
    P --> Q["addmm_float8_unwrapped_inference(...)"]
    Q --> R["NPU fastpath?"]
    R --> S["_fp8_npu_matmul(...)"]
    R --> T["torch._scaled_mm(...)"]

    B --> U["layer[1] = nn.Linear.forward"]
    U --> V["与 layer[0] 相同"]

    B --> W["layer[2] = nn.Linear.forward"]
    W --> X["普通 F.linear(...)"]

    M --> Y["得到 layer[0]/layer[1] 输出"]
    T --> Y
    S --> Y
    Y --> Z["print(out)"]
    X --> Z
```

## 10. 训练态和推理态调用链的关键区别

```text
训练态:
Float8Linear.forward
  -> matmul_with_hp_or_float8_args.forward/backward
  -> Float8TrainingTensor.__torch_dispatch__
  -> float8_mm / float8_addmm
  -> torch._scaled_mm

推理态:
nn.Linear.forward
  -> F.linear(input, weight=Float8Tensor, bias)
  -> Float8Tensor 对 F.linear / aten.linear.default 的 override
  -> 动态把 activation 量化成 Float8Tensor
  -> fbgemm.f8f8bf16 或 torch._scaled_mm / NPU matmul
```

一句话记忆：

- 训练态是“模块替换成 `Float8Linear`，前后向都由自定义 autograd 接管”。
- 推理态是“保留 `nn.Linear`，把权重换成 `Float8Tensor`，在 `F.linear` 里动态量化 activation 并分发到 float8 kernel”。

## 11. 训练态抽象 vs 推理态抽象

下面这张小图可以直接回答“为什么推理前要先还原成普通 `nn.Linear`”：

```mermaid
flowchart LR
    subgraph T["训练态抽象"]
        T1["模块形态: Float8Linear"]
        T2["forward: matmul_with_hp_or_float8_args"]
        T3["backward: 自定义三次 GEMM"]
        T4["核心 tensor: Float8TrainingTensor"]
        T5["设计目标: 训练数值路径 + autograd 可控"]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph I["推理态抽象"]
        I1["模块形态: nn.Linear"]
        I2["只替换 weight -> Float8Tensor"]
        I3["在 F.linear 中动态量化 activation"]
        I4["分发到 fbgemm / torch._scaled_mm / NPU"]
        I5["设计目标: 统一量化框架 + kernel dispatch + 部署"]
        I1 --> I2 --> I3 --> I4 --> I5
    end

    T1 -. "checkpoint 加载后，先去掉训练壳" .-> I1
    T4 -. "训练专用 tensor，不直接复用到推理接口" .-> I2
```

可以把这个设计理解成两层分工：

- 训练态解决的是“怎么安全地把 `Linear` 的前向和反向都改成 float8 训练路径”。
- 推理态解决的是“怎么在尽量保留 `nn.Linear` 形态的前提下，把权重和激活接到最合适的推理 kernel 上”。

所以推理前先把 `Float8Linear` 还原成 `nn.Linear`，本质上是在去掉训练专用的模块壳，回到推理量化框架的统一入口。
