# TorchAO Float8 Training 入门

这篇文档面向第一次接触 TorchAO 的读者，目标不是先讲完所有量化理论，而是先帮你读懂下面这段代码在做什么，以及它为什么能跑起来。

先说明一个很实际的前提：这篇文档讲的两条主线都是 `float8`。

- 训练示例直接用了 `.cuda()`，默认面向 CUDA/ROCm 等加速器环境
- 推理里的 `Float8DynamicActivationFloat8WeightConfig` 也有明确硬件要求

所以如果你的机器是 Mac，尤其是只有 Apple MPS、没有 CUDA/ROCm/XPU/NPU，那么这两段示例通常不适合作为“本机可运行教程”。

这不影响你用这篇文档理解源码和调用链，但会影响你直接复现运行结果。对 Mac 用户，更合理的目标通常是：

- 先读懂 TorchAO 的设计和代码路径
- 真要跑 float8，再切到支持的 Linux/GPU 环境

```python
import torch
from torch import nn
import torch.nn.functional as F

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8 import convert_to_float8_training

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
    nn.Linear(128, 1),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, module_filter_fn=module_filter_fn)

# enable torch.compile for competitive performance
# m = torch.compile(m)

# toy training loop
for _ in range(10):
    optimizer.zero_grad()
    output = m(x)
    # use fake labels for demonstration purposes
    fake_labels = torch.ones_like(output)
    loss = F.mse_loss(output, fake_labels)
    loss.backward()
    optimizer.step()

# save the model
torch.save({
    'model': m,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

补充两点小细节：

- `Float8Linear` 在这段示例里只是用来观察替换结果，训练主流程本身并不直接依赖你手动调用它。
- `convert_to_float8_training` 被导入了两次，实际保留 `from torchao.float8 import convert_to_float8_training` 这一行就够了。

## 1. 先用一句话理解 TorchAO

`torchao` 是一个基于原生 PyTorch 的模型优化库。它主要做两类事情：

- 让模型在推理时更小、更快，比如 int4/int8/float8 量化。
- 让模型在训练时更快，比如 float8 training、稀疏训练。

你这段代码使用的是第二类能力：`torchao.float8`。

它的核心目标不是“把整个模型永久压成 8 bit 文件”，而是：

- 让 `Linear` 里的矩阵乘法尽量用 float8 kernel 跑起来。
- 同时尽量保留 PyTorch 原本的训练体验，包括 autograd、optimizer、`torch.compile`、分布式等。

## 2. 这段代码到底是不是“量化”

是，但它和很多人第一次接触到的“推理量化”不是一回事。

最容易混淆的区别是：

- 推理量化通常会把权重长期保存成 int8/int4/float8，目标是省显存、省带宽、提速推理。
- 这里的 float8 training 更像是“训练时低精度计算路径”。

在这套实现里，模型参数并不会简单粗暴地永久改成 float8 再训练。更准确地说：

- 模型参数通常仍然以较高精度保存和更新，比如你的例子里是 `bfloat16`。
- 真正做 GEMM 时，输入、权重、梯度会按配置动态 cast 成 float8。
- GEMM 做完后，结果再回到期望的输出精度。

所以，理解这段代码最好的方式不是“权重量化了”，而是：

“训练时把 `Linear` 的前向和反向矩阵乘法切到 float8 计算路径。”

## 3. 先看你的示例最终会转换哪些层

你的模型是：

1. `0`: `Linear(2048, 4096)`
2. `1`: `Linear(4096, 128)`
3. `2`: `Linear(128, 1)`

再叠加你的 `module_filter_fn`：

- `fqn == "1"` 时返回 `False`，所以第二层不会被转换。
- 维度不是 16 的倍数时也不会转换。

因此最终结果是：

- 第 0 层会被转换为 `Float8Linear`
- 第 1 层不会转换，因为被 `fqn == "1"` 排除了
- 第 2 层不会转换，因为 `out_features = 1`，不是 16 的倍数

这里有一个很容易忽略的小坑：注释写的是 “don't convert the last module”，但在 `nn.Sequential` 里：

- `"0"` 是第一层
- `"1"` 是第二层
- `"2"` 才是最后一层

也就是说，这段代码实际跳过的是中间层，不是最后一层。如果你的真实意图是跳过最后一层，条件应该写成 `fqn == "2"`。

## 4. 从入口 API 开始：`convert_to_float8_training`

你调用的是：

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m, module_filter_fn=module_filter_fn)
```

这个符号是从 [torchao/float8/__init__.py](../../torchao/float8/__init__.py) 导出的，真正实现位于 [torchao/float8/float8_linear_utils.py](../../torchao/float8/float8_linear_utils.py)。

它做的事情并不神秘，本质上只有两步：

1. 遍历整个模型，找到所有 `nn.Linear`
2. 把符合条件的 `nn.Linear` 原地替换成 `Float8Linear`

对应源码逻辑可以概括成下面这样：

```python
def convert_to_float8_training(module, module_filter_fn=None, config=None):
    if config is None:
        config = Float8LinearConfig()

    from_float = lambda m: Float8Linear.from_float(m, config=config)
    return swap_linear_layers(module, from_float, module_filter_fn=module_filter_fn)
```

也就是说，`convert_to_float8_training` 本身不负责真正的 float8 GEMM，它只是“把模块换掉”。

## 5. 为什么先建 optimizer，再做模块替换，代码还能工作

很多人第一次看到这里会困惑：

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
convert_to_float8_training(m, ...)
```

通常来说，如果你替换了模块，优化器不是会丢失参数引用吗？

这里之所以还能工作，是因为 `Float8Linear.from_float(...)` 并没有重新随机初始化一套新参数，而是复用了原来的 `weight` 和 `bias` 对象。

在 [torchao/float8/float8_linear.py](../../torchao/float8/float8_linear.py) 里，核心逻辑是：

```python
with torch.device("meta"):
    new_mod = cls(mod.in_features, mod.out_features, bias=False, config=config)
new_mod.weight = mod.weight
new_mod.bias = mod.bias
```

这意味着：

- 模块类型变了，从 `nn.Linear` 变成 `Float8Linear`
- 但底层参数对象还是原来那两个 `Parameter`

所以：

- 已经创建好的 optimizer 仍然能更新这些参数
- 9 `state_dict()` 也仍然是熟悉的参数保存方式
- 这也是 TorchAO 尽量保持“PyTorch 原生训练体验”的一个关键设计

## 6. `Float8Linear` 到底和 `nn.Linear` 有什么不同

最关键的区别在 `forward()`。

`nn.Linear` 大致做的是：

```python
output = input @ weight.T + bias
```

而 `Float8Linear.forward()` 做的是：

1. 先看当前是否 autocast
2. 把输入和权重按配置决定是否 cast 成 float8
3. 调用 float8 matmul
4. 再把 bias 加回去

源码入口在 [torchao/float8/float8_linear.py](../../torchao/float8/float8_linear.py)：

```python
output = matmul_with_hp_or_float8_args.apply(
    input,
    self.weight.t(),
    self.linear_mm_config,
    self.config,
)
```

这里不是直接调普通函数，而是调了一个自定义 `autograd.Function`。这么做的原因是：

- 前向要控制 cast 和 mm
- 反向也要控制 `grad_input`、`grad_weight` 的 float8 路径

换句话说，TorchAO 不是只改了前向，它是把 `Linear` 的前向和反向 GEMM 都重新接管了。

## 7. 前向到底怎么走

下面按你一次 `output = m(x)` 来展开。

### 第一步：进入 `Float8Linear.forward`

如果某一层已经被替换成 `Float8Linear`，调用这层时会进入它自己的 `forward`。

这时会把：

- `input`
- `self.weight.t()`
- `linear_mm_config`
- `config`

一起交给 `matmul_with_hp_or_float8_args.apply(...)`。

### 第二步：决定要不要 cast 成 float8

在 `matmul_with_hp_or_float8_args.forward(...)` 中，TorchAO 会分别处理：

- 输入 `input_hp`
- 转置后的权重 `weight_hp_t`

判断逻辑大致是：

- 如果它已经是 fp8 tensor，就直接用
- 如果这一项配置成了 `ScalingType.DISABLED`，就保留高精度
- 否则就调用 `hp_tensor_to_float8_dynamic(...)`

这一步的目的很直接：float8 数值范围很小，不能直接把 bf16 生硬地 cast 下去，必须先根据当前 tensor 的幅值算一个 scale，再做转换。

### 第三步：动态计算 scale

`hp_tensor_to_float8_dynamic(...)` 位于 [torchao/float8/float8_scaling_utils.py](../../torchao/float8/float8_scaling_utils.py)。

它会先调：

```python
scale = tensor_to_scale(...)
```

再调：

```python
hp_tensor_and_scale_to_float8(...)
```

其中：

- `tensor_to_scale(...)` 会先统计 `amax = max(abs(x))`
- 再根据目标 float8 dtype 的可表示范围，算出合适的缩放因子

直觉上可以理解为：

- 原 tensor 太大，直接 cast 到 float8 会溢出
- 原 tensor 太小，直接 cast 到 float8 会大量下溢成 0
- 所以先乘一个 scale，把数值尽量搬到 float8 更容易表达的范围里

这就是 dynamic scaling 的核心原因。

### 第四步：把高精度 tensor 包装成 `Float8TrainingTensor`

`hp_tensor_and_scale_to_float8(...)` 并不是简单返回一个普通 `torch.Tensor`。

它返回的是 `Float8TrainingTensor`，这是一个 tensor subclass，定义在 [torchao/float8/float8_training_tensor.py](../../torchao/float8/float8_training_tensor.py)。

它内部同时保存了：

- `_data`: 真正的 float8 数据
- `_scale`: 对应的 scale
- `_orig_dtype`: 原始精度，比如 bf16
- `_linear_mm_config`
- `_gemm_input_role`

这样设计的原因是：后面做矩阵乘法时，TorchAO 需要同时知道“float8 位模式”和“如何把它解释回原始精度”。

### 第五步：`torch.mm(...)` 被重定向到 float8 kernel

在 `matmul_with_hp_or_float8_args.forward(...)` 里，真正的乘法是：

```python
res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
```

看起来像普通 `torch.mm`，但这里的输入往往已经是 `Float8TrainingTensor`。

这时 PyTorch 会进入 `Float8TrainingTensor.__torch_dispatch__`，然后路由到 [torchao/float8/float8_ops.py](../../torchao/float8/float8_ops.py) 里的 `float8_mm(...)`。

在 `float8_mm(...)` 中，TorchAO 最终会调用：

```python
torch._scaled_mm(...)
```

这才是底层真正吃 float8 数据和 scale 的矩阵乘法入口。

所以，前向主链可以记成：

```text
Float8Linear.forward
  -> matmul_with_hp_or_float8_args.forward
    -> hp_tensor_to_float8_dynamic
      -> tensor_to_scale
      -> hp_tensor_and_scale_to_float8
        -> Float8TrainingTensor
    -> torch.mm
      -> Float8TrainingTensor.__torch_dispatch__
      -> float8_mm
      -> torch._scaled_mm
```

最后 `Float8Linear.forward` 再把 `bias` 加回去。

## 8. 反向是怎么走的

这部分非常值得看，因为它解释了为什么 TorchAO 需要自定义 `autograd.Function`。

对于一个 `Linear`，反向至少有三类核心计算：

1. 前向：`input @ weight_t -> output`
2. 反向求输入梯度：`grad_output @ weight -> grad_input`
3. 反向求权重梯度：`grad_output_t @ input -> grad_weight`

在 [torchao/float8/float8_training_tensor.py](../../torchao/float8/float8_training_tensor.py) 文件顶部就专门写了这三个 GEMM。

`matmul_with_hp_or_float8_args.backward(...)` 会：

- 读取前向保存下来的 `input_hp` 和 `weight_hp_t`
- 根据配置决定如何处理 `grad_output`
- 分别计算 `grad_input`
- 再计算 `grad_weight`

而且这两次乘法也会复用和前向相同的 float8 dispatch 机制。

也就是说，TorchAO 并不是只在 forward 时“偷偷换了个低精度 matmul”，它是系统性地为 `Linear` 的前向、后向三次 GEMM 都建立了可配置的 float8 路径。

## 9. `Float8LinearConfig` 在控制什么

如果你不传 `config`：

```python
convert_to_float8_training(m)
```

默认会创建一个 `Float8LinearConfig()`，定义在 [torchao/float8/config.py](../../torchao/float8/config.py)。

默认配置基本等价于 `tensorwise` recipe，也就是：

- `input` 用动态 tensorwise scaling cast 到 float8
- `weight` 用动态 tensorwise scaling cast 到 float8
- `grad_output` 也会参与动态 scaling
- 默认 dtype 选择上，`input/weight` 主要用 `e4m3`，`grad_output` 默认会落到 `e5m2`

它主要控制三件事：

1. 哪些张量要 cast
2. 用什么粒度做 scaling
3. 三次 GEMM 各自的配置

### 9.1 哪些张量要 cast

配置并不是只有“开 float8 / 关 float8”这么简单。

TorchAO 会分别配置：

- `input`
- `weight`
- `grad_output`

并且还能分别控制它们在不同反向 GEMM 中的行为。

### 9.2 scaling 粒度

最常见的两种粒度是：

- `tensorwise`: 整个 tensor 共用一个 scale
- `axiswise` 或 rowwise: 沿某个轴分别算 scale

直觉上：

- `tensorwise` 更简单，通常更快
- `rowwise` 更细粒度，往往数值更稳，但实现和开销更复杂

### 9.3 recipe

TorchAO 给你预设了三种 recipe：

- `tensorwise`
- `rowwise`
- `rowwise_with_gw_hp`

在 [torchao/float8/config.py](../../torchao/float8/config.py) 里可以看到 `Float8LinearConfig.from_recipe_name(...)` 如何构造它们。

一个够用的经验是：

- 刚上手先用默认配置或 `tensorwise`
- 如果更关心收敛和数值稳定，再去看 `rowwise`

## 10. 为什么很多地方都在强调“维度必须是 16 的倍数”

因为底层 float8 kernel 对矩阵形状有硬件和实现约束。

在这套实现里，很多路径最终会走到 `torch._scaled_mm(...)`，而这类 kernel 通常要求相关维度满足对齐条件。仓库里多处都明确写了这一点，所以示例代码和自动过滤逻辑都会检查：

```python
mod.in_features % 16 == 0 and mod.out_features % 16 == 0
```

这也是为什么：

- 某些层明明是 `nn.Linear`
- 但仍然不建议强行转成 `Float8Linear`

否则你可能得到：

- 性能不升反降
- 或者直接命中 kernel 约束

## 10.1 这套 float8 路径对硬件也有明确要求

除了 shape 约束，这两条 float8 路径还有硬件约束。

对于推理里的 dynamic float8 quantization，源码在 [torchao/float8/inference.py](../../torchao/float8/inference.py) 里明确检查：

- CUDA 且 compute capability >= 8.9
- 或 MI300+
- 或 XPU
- 或 NPU

如果不满足，就会直接报：

```text
Float8 dynamic quantization requires CUDA compute capability ≥8.9 or MI300+ or XPU or NPU.
```

这意味着：

- Apple Silicon 的 MPS 不在这条支持列表里
- 所以你在 Mac 上通常不能真正跑通这里的 float8 推理量化流程

训练那条 `Float8Linear` 路线虽然更像 Python 级模块替换，但示例本身仍然是围绕 CUDA 工作流设计的，而且性能收益也建立在对应硬件 kernel 存在的前提上。

## 11. 为什么推荐配合 `torch.compile`

示例里写了：

```python
# m = torch.compile(m)
```

这不是装饰项，而是非常关键的性能建议。

原因是 TorchAO 的 float8 路径里有不少 Python 层逻辑，例如：

- 动态计算 scale
- tensor subclass dispatch
- 自定义 autograd.Function

如果完全 eager 执行，框架开销会比较明显。`torch.compile` 更有机会把这些逻辑和周边算子一起优化掉，所以官方 README 才会写 “for competitive performance”。

换句话说：

- 不开 `torch.compile`，功能通常仍然能跑
- 但如果你想真的看到比较像样的训练吞吐收益，通常应该开

## 12. 保存 checkpoint 时，保存下来的到底是什么

你最后调用了：

```python
torch.save({
    "model": m,
    "model_state_dict": m.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pth")
```

这里建议你这样理解：

- `model` 会把当前模块结构一起存下去，也就是里面已经包含 `Float8Linear`
- `model_state_dict()` 主要还是参数状态
- `optimizer_state_dict()` 是 AdamW 的状态

因为 `Float8Linear.from_float(...)` 复用了原始参数对象，所以很多时候 state dict 仍然保持“高精度参数 + float8 训练逻辑由模块负责”的风格。

工程实践里通常更推荐：

1. 重新构造模型
2. 再次调用 `convert_to_float8_training(...)`
3. 然后加载 `state_dict`

原因不是这段代码不能保存整个模型，而是 `state_dict` 的兼容性通常更稳。

## 13. 把这段代码翻译成一句“人话”

如果把你的示例压缩成一句话，就是：

“先建一个 bf16 的普通 PyTorch MLP，再用 TorchAO 把一部分 `Linear` 替换成 `Float8Linear`，让这些层在训练时的前向和反向 GEMM 动态走 float8 kernel，同时保持 optimizer、autograd、checkpoint 用法基本不变。”

## 14. 推荐的最小上手顺序

如果你是零基础，我建议按这个顺序理解，而不是一上来就读完整个仓库：

1. 先读 [torchao/float8/README.md](../../torchao/float8/README.md)，建立整体印象。
2. 再看 [torchao/float8/float8_linear_utils.py](../../torchao/float8/float8_linear_utils.py)，理解“模块替换”。
3. 再看 [torchao/float8/float8_linear.py](../../torchao/float8/float8_linear.py)，理解 `forward/backward` 主逻辑。
4. 然后看 [torchao/float8/float8_scaling_utils.py](../../torchao/float8/float8_scaling_utils.py)，理解 dynamic scaling。
5. 最后看 [torchao/float8/float8_training_tensor.py](../../torchao/float8/float8_training_tensor.py) 和 [torchao/float8/float8_ops.py](../../torchao/float8/float8_ops.py)，理解 tensor subclass dispatch 到 `_scaled_mm`。

## 15. 建议你立刻动手做的三个小实验

### 实验 1：确认哪些层真的被替换了

```python
print(m)
for name, mod in m.named_modules():
    if isinstance(mod, Float8Linear):
        print("float8 layer:", name, mod)
```

### 实验 2：确认权重参数本身仍然是高精度

```python
print(type(m[0]))
print(m[0].weight.dtype)
print(m[0].bias.dtype)
```

如果第一层被替换成功，你会看到：

- 模块类型是 `Float8Linear`
- 但 `weight.dtype` 仍然很可能是 `torch.bfloat16`

### 实验 3：修正过滤条件，观察替换结果变化

把：

```python
if fqn == "1":
    return False
```

改成：

```python
if fqn == "2":
    return False
```

再重新打印模型，你会更直观地理解 FQN 是怎么工作的。

## 16. 一张总图

```text
原始模型 nn.Sequential
  -> convert_to_float8_training(...)
    -> 遍历子模块
    -> 命中 nn.Linear 且通过 filter
    -> Float8Linear.from_float(...)
      -> 复用原 weight/bias
      -> 模块替换完成

训练时 forward
  -> Float8Linear.forward(...)
    -> matmul_with_hp_or_float8_args.apply(...)
      -> 对 input/weight 做 dynamic scaling
      -> 包装成 Float8TrainingTensor
      -> torch.mm(...)
        -> __torch_dispatch__
        -> float8_mm(...)
        -> torch._scaled_mm(...)
    -> 加 bias

训练时 backward
  -> matmul_with_hp_or_float8_args.backward(...)
    -> 计算 grad_input
    -> 计算 grad_weight
    -> 同样复用 float8 mm 路径

optimizer.step()
  -> 更新原始高精度参数
```

## 17. 你现在应该记住的四句话

- `convert_to_float8_training` 做的是模块替换，不是直接执行 GEMM。
- 真正的 float8 计算发生在 `Float8Linear.forward/backward` 里。
- 参数通常仍然以高精度保存和更新，float8 主要是计算路径。
- `torch.compile`、层过滤和 16 对齐约束，都是为了让这条路径“既能跑，又值得跑”。

## 18. 第二段代码在做什么

现在来看你补充的第二段代码：

```python
import torch

from torchao.float8.float8_linear import Float8Linear
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import quantize_
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)

# load checkpoint
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

# optional: apply dynamic float8 quantization on both activations and weights for inference
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

# run inference
x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    out = model(x)
    print(out)
```

这段代码的目标已经和前面的训练代码不一样了。

前一段代码的目标是：

- 训练时让 `Linear` 的前向和反向 GEMM 用 float8 路径跑

这一段代码的目标是：

- 加载训练好的模型
- 把它改造成适合推理的 float8 量化模型
- 在推理时对 activation 做动态 float8 量化，对 weight 使用 float8 权重

所以它属于 float8 inference quantization，而不是 float8 training。

## 19. 这段代码和前一段代码的关系

如果 checkpoint 是由前一段代码保存出来的，那么这里的 `model` 往往有两个特征：

- 模块结构里可能已经包含 `Float8Linear`
- 参数值已经是训练完成后的值

这也是为什么这段推理代码看起来有点“叠层”：

1. 先从 checkpoint 里拿到一个已经训练过的模型对象
2. 再在这个模型上调用 `quantize_`

但这里并不是“再走一次训练 float8 转换”，而是切换到另一套面向推理的量化机制。

## 20. `torch.load(...)` 和 `load_state_dict(...)` 这两步各自做了什么

你写的是：

```python
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])
```

可以这样理解：

- `checkpoint['model']` 会把整个模块对象反序列化出来
- `checkpoint['model_state_dict']` 是一份显式保存的参数状态

因为你在保存时已经把 `model` 整个存进去了，所以严格来说：

```python
model = checkpoint['model']
```

这一步通常已经把参数带回来了。

因此这里再执行：

```python
model.load_state_dict(checkpoint['model_state_dict'])
```

很多时候是重复恢复一次状态。

它不是错，只是语义上有些重复。更常见、更稳妥的工程写法通常是：

1. 重新构造模型结构
2. `load_state_dict(...)`

但既然你这里已经把整个模型对象一起保存了，这样写也能工作。

## 21. 这时模型里到底还是不是 `Float8Linear`

很可能是。

如果这个 checkpoint 来自前面的训练脚本，那么 `checkpoint['model']` 里被训练时转换过的层，通常还是 `Float8Linear`。

这点很重要，因为很多人会自然地以为：

- 训练阶段用了 `Float8Linear`
- 推理时就继续沿用 `Float8Linear`

但 TorchAO 在推理量化路径里，实际上并不想继续沿用训练专用的 `Float8Linear` 模块逻辑。

在 [torchao/quantization/quant_api.py](../../torchao/quantization/quant_api.py) 里，`quantize_` 调用的递归替换函数开头就专门做了这个处理：

```python
if isinstance(model, Float8Linear):
    with torch.device("meta"):
        new_module = nn.Linear(model.in_features, model.out_features)
    new_module.weight = model.weight
    new_module.bias = model.bias
    model = new_module
```

也就是说，进入推理量化前：

- 如果发现某层是 `Float8Linear`
- TorchAO 会先把它“还原成普通 `nn.Linear`”
- 同时继续复用原来的 `weight` 和 `bias`

这一步的目的非常明确：

- 训练 float8 用模块替换
- 推理 float8 量化用权重 tensor subclass

这两条路线不要混在一起。

## 22. `quantize_` 这次做的不是“替换模块”，而是“替换权重”

前面训练流程里你已经见过：

- `convert_to_float8_training(...)` 是替换 `nn.Linear -> Float8Linear`

这次推理量化里，`quantize_(...)` 的核心动作不同。

它本质上是：

1. 遍历模型里的线性层
2. 不去换模块类型
3. 只把 `module.weight` 替换成量化后的 tensor subclass

入口在 [torchao/quantization/quant_api.py](../../torchao/quantization/quant_api.py)，`quantize_` 默认会对符合条件的 `nn.Linear` 应用配置对应的 handler。

对于 `Float8DynamicActivationFloat8WeightConfig`，真正注册的 handler 是：

- `_float8_dynamic_activation_float8_weight_transform(...)`

它的核心逻辑是：

```python
quantized_weight = _float8_dynamic_activation_float8_weight_quantize_tensor(
    module.weight, config
)
module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
```

所以这一阶段的模型更像：

- 模块仍然是 `nn.Linear`
- 但 `weight` 已经不再是普通 bf16 tensor，而是 `Float8Tensor`

## 23. `Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())` 在表达什么

这个配置名比较长，但拆开看并不复杂：

- `Float8DynamicActivation`: activation 在运行时动态量化成 float8
- `Float8Weight`: 权重也量化成 float8

你的写法是：

```python
Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
```

这里的 `PerTensor()` 表示：

- activation 使用 per-tensor scale
- weight 也使用 per-tensor scale

也就是：

- 一个 activation tensor 共用一个 scale
- 一个 weight tensor 共用一个 scale

这是更容易理解、也更简单的一种量化粒度。

直觉上：

- `PerTensor` 更简单
- `PerRow` 更细粒度，数值表现通常更好，但逻辑更复杂

## 24. `Float8DynamicActivationFloat8WeightConfig` 是怎么初始化的

在 [torchao/quantization/quant_api.py](../../torchao/quantization/quant_api.py) 里，这个 config 的 `__post_init__` 会做几件事：

1. 如果你没传 `mm_config`，默认用 `Float8MMConfig(use_fast_accum=True)`
2. 把 `granularity` 规范化成 `[activation_granularity, weight_granularity]`

所以你的这句：

```python
Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
```

最终可以理解成：

- activation granularity = `PerTensor()`
- weight granularity = `PerTensor()`
- matmul 默认允许 fast accumulation

## 25. 权重是怎么从 bf16 变成 `Float8Tensor` 的

进入 `_float8_dynamic_activation_float8_weight_quantize_tensor(...)` 之后，TorchAO 会：

1. 检查硬件和形状是否兼容
2. 构造 activation 的量化参数 `act_quant_kwargs`
3. 用 `Float8Tensor.from_hp(...)` 把高精度权重转成量化权重

如果是默认的 version 2 路线，核心代码大致是：

```python
act_quant_kwargs = QuantizeTensorToFloat8Kwargs(...)
quantized_weight = Float8Tensor.from_hp(
    weight,
    float8_dtype=weight_dtype,
    granularity=weight_granularity,
    mm_config=mm_config,
    act_quant_kwargs=act_quant_kwargs,
)
```

注意这里有一个非常关键的设计：

- `act_quant_kwargs` 被挂在 weight tensor 上
- 这样在真正执行 `F.linear` 时，weight tensor 自己就知道“输入 activation 应该怎么量化”

也就是说，推理量化并不是提前把一切都静态准备好，而是把“如何量化输入”的规则也一起封装进了量化权重对象。

## 26. `Float8Tensor.from_hp(...)` 里到底发生了什么

`Float8Tensor` 定义在 [torchao/quantization/quantize_/workflows/float8/float8_tensor.py](../../torchao/quantization/quantize_/workflows/float8/float8_tensor.py)。

这个类和训练阶段的 `Float8TrainingTensor` 不同，它是面向推理量化工作流的 tensor subclass。

在 `from_hp(...)` 里，TorchAO 会做几件关键的事：

1. 根据 `PerTensor()` 计算 block size
2. 选择权重量化时使用的 quantize kernel
3. 计算 scale
4. 把高精度权重转成 raw float8 数据
5. 返回 `Float8Tensor(qdata, scale, ...)`

对于你的 `PerTensor()` 示例，一个重要细节是：

- 权重量化阶段通常不会走 fbgemm 的 per-tensor quantize kernel
- 而是走 torch 路径，使用 `_choose_scale_float8(...)` 和 `_quantize_affine_float8(...)`

这一步完成后，权重内部会包含：

- `qdata`: 原始 float8 数据
- `scale`: 对应 scale
- `act_quant_kwargs`: 以后量化 activation 用的配置
- `mm_config`: 后续 matmul 配置

## 27. 推理时为什么模块还是普通 `nn.Linear`，却能自动走 float8 kernel

因为普通 `nn.Linear` 的 forward 最终会调：

```python
F.linear(input, weight, bias)
```

而现在：

- `input` 是普通 bf16 tensor
- `weight` 是 `Float8Tensor`

只要 `weight` 是 tensor subclass，PyTorch 就会把这次 `F.linear` 调度到 `Float8Tensor` 注册的实现上。

也就是说，推理路径的关键不是“换模块”，而是：

- 让 `weight` 变成一个会接管 `F.linear` 的特殊 tensor

## 28. 推理 forward 的主调用链

当你执行：

```python
with torch.inference_mode():
    out = model(x)
```

对某个被量化过的线性层来说，主链大致是：

```text
nn.Linear.forward
  -> F.linear(input, weight, bias)
    -> Float8Tensor 对 F.linear 的 override
      -> 动态量化 input
      -> 选择 matmul kernel
      -> 调用 float8 matmul
```

在 [torchao/quantization/quantize_/workflows/float8/float8_tensor.py](../../torchao/quantization/quantize_/workflows/float8/float8_tensor.py) 里，这个 override 做了两件最核心的事。

### 第一步：把输入 activation 动态量化成 `Float8Tensor`

代码里是：

```python
if act_quant_kwargs is not None:
    input_tensor = _choose_quant_func_and_quantize_tensor(
        input_tensor, act_quant_kwargs
    )
```

所以即使你传进去的 `x` 是 bf16：

```python
x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
```

在真正做 matmul 之前，它会先被动态量化成 float8 activation。

这也是为什么这个 config 叫：

- dynamic activation + float8 weight

### 第二步：拿 activation 和 weight 的 `qdata/scale` 去做 matmul

接下来会取出：

- `input_tensor.qdata`
- `input_tensor.scale`
- `weight_tensor.qdata`
- `weight_tensor.scale`

然后根据 kernel 选择，进入不同分支。

## 29. 你的 `PerTensor()` 示例最终可能走哪种 kernel

这里要分成两层来看：

### 29.1 权重量化阶段

在 `Float8Tensor.from_hp(...)` 里：

- `PerTensor()` 通常会走 torch quantize 路径
- 而不是 fbgemm 的量化路径

### 29.2 推理 matmul 阶段

到了真正的 `F.linear` 执行阶段，如果满足条件，例如：

- 输入和权重都在 CUDA
- 有 `fbgemm_gpu_genai`
- 硬件满足要求

就可能走 fbgemm kernel：

- rowwise 时走 `torch.ops.fbgemm.f8f8bf16_rowwise`
- tensorwise 时走 `torch.ops.fbgemm.f8f8bf16`

否则就走 torch 路径，最终进入：

- `addmm_float8_unwrapped_inference(...)`
- `torch._scaled_mm(...)`

也就是说，对于你的示例，真正的 matmul 可能是：

- fbgemm float8 kernel
- 或 `torch._scaled_mm`

取决于硬件、库可用性和 kernel 选择。

## 30. 如果走 torch 路径，最终在哪里做矩阵乘法

torch 路径会落到 [torchao/float8/inference.py](../../torchao/float8/inference.py) 里的：

- `addmm_float8_unwrapped_inference(...)`

它最终调用的是：

```python
torch._scaled_mm(
    a_data,
    b_data,
    scale_a=a_scale,
    scale_b=b_scale,
    bias=bias,
    out_dtype=output_dtype,
    use_fast_accum=use_fast_accum,
)
```

所以推理路径里最底层的“真正 float8 matmul”通常还是围绕：

- `torch._scaled_mm`

展开的，只不过封装层和训练路径不同。

## 31. 你的输入 `x.shape = (1, 4096, 2048)` 为什么能喂给线性层

因为 `nn.Linear` 只要求最后一维是 `in_features`。

你的第一层是：

- `Linear(2048, 4096)`

输入是：

- `(1, 4096, 2048)`

所以 PyTorch 会把前面的维度都当成 batch-like 维度，最后一维 `2048` 当成特征维。

于是：

- 第一层输出是 `(1, 4096, 4096)`
- 第二层输出是 `(1, 4096, 128)`
- 第三层输出是 `(1, 4096, 1)`

最终 `out` 的形状就是：

- `(1, 4096, 1)`

## 32. 这一段代码里两个容易忽略的小点

### 32.1 `Float8Linear` 这个 import 实际上没用到

你写了：

```python
from torchao.float8.float8_linear import Float8Linear
```

但这段脚本本身并没有直接使用 `Float8Linear`。

保留它最多是为了调试，比如：

```python
print(type(model[0]))
isinstance(model[0], Float8Linear)
```

否则可以删掉。

### 32.2 `quantize_` 是原地修改

调用：

```python
quantize_(model, ...)
```

以后，`model` 本身就被改了，不会返回一个新的独立模型副本。

## 33. 这段推理代码翻译成一句“人话”

“先把训练好的 checkpoint 恢复出来，如果模型里还有训练阶段专用的 `Float8Linear`，就先回到普通 `nn.Linear` 语义，然后把每个线性层的权重改成 `Float8Tensor`，并在每次推理时把输入 activation 动态量化成 float8，最后用 float8 matmul kernel 完成线性计算。”

## 34. 和前面训练 float8 路线的本质区别

最容易混淆的就是这一点，我建议你硬记住下面这组对照。

### 训练 float8

- 入口是 `convert_to_float8_training(...)`
- 主要动作是 `nn.Linear -> Float8Linear`
- 参数通常保持高精度保存和更新
- 前向和反向 GEMM 都要走 float8 逻辑
- 重点是训练吞吐和可训练性

### 推理 float8 量化

- 入口是 `quantize_(..., Float8DynamicActivationFloat8WeightConfig(...))`
- 主要动作是 `module.weight -> Float8Tensor`
- 模块通常保持 `nn.Linear`
- 只需要考虑前向推理
- 重点是推理吞吐、显存和部署路径

## 35. 把两段代码串起来看

如果把你前后两段代码连起来，完整故事其实是：

```text
高精度模型
  -> convert_to_float8_training(...)
  -> 训练时用 Float8Linear 跑 float8 GEMM
  -> 保存 checkpoint

checkpoint
  -> torch.load(...)
  -> 取回训练好的模型和参数
  -> quantize_(..., Float8DynamicActivationFloat8WeightConfig(...))
  -> 把推理时需要的线性层权重换成 Float8Tensor
  -> 推理时把 activation 动态量化
  -> 调用 float8 inference kernel
```

## 36. 你现在应该再记住三句话

- 训练 float8 和推理 float8 量化是两条不同实现路线，不要把 `Float8Linear` 和 `Float8Tensor` 当成一回事。
- `quantize_` 在这条推理路径里主要改的是 `weight`，不是模块类型。
- 你的第二段代码本质上是在把“训练好的模型”重新包装成“适合 float8 推理的模型”。
