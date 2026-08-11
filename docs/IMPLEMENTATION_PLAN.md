# fastProve 实施方案（阶段 A–D）

> 本文档面向执行者。**按顺序做，不要跳步。** 每个任务给出：目标、涉及文件、
> 完整代码或改法、验证命令、验收标准、失败回滚方式。
>
> 所有数学恒等式已在 `docs/design_identity_check.py` 中用 FP64 独立验证
> （13 项，误差 ≤ 3e-14）。执行任何任务前先跑一次：
>
> ```bash
> python docs/design_identity_check.py
> ```
>
> 该脚本不 import fastprove，是独立的数学预言机（oracle）。
> **判据：该脚本通过但仓库测试失败 ⇒ 你的实现偏离了本方案；该脚本失败 ⇒ 环境
> 有问题（torch/FP64），先修环境。**

---

## 0. 如何使用本文档

### 0.1 每个任务的固定流程

```text
1. 读「目标」和「为什么」
2. 读「前置条件」，确认已满足
3. 按「步骤」改代码（代码块可直接复制）
4. 跑「验证命令」
5. 对照「验收标准」逐条打勾
6. 不达标 → 看「常见失败」；仍不行 → 按「回滚」恢复，然后上报，不要猜改
```

### 0.2 强制规则（违反即返工）

| 编号 | 规则 |
|---|---|
| R1 | 不得为了让测试通过而修改数学定义。测试失败先查实现，不改公式。 |
| R2 | 每个任务结束时 `python -m pytest -q` 必须全绿。不允许留失败测试进入下一任务。 |
| R3 | 每个任务单独一个 git commit，commit message 用任务编号（如 `A2: structured basis`）。 |
| R4 | `forward()` 中禁止出现 `torch.linalg.inv`、`solve`、`pinv`、`lstsq`。 |
| R5 | 所有 RMS 统计、margin、softmax reduction、噪声裁剪用 FP32 及以上。 |
| R6 | 离线转换、求逆、条件数检查用 FP64。 |
| R7 | 生产 API（`forward`）不得返回 hidden state、噪声态、attention 概率、router logits。 |
| R8 | 新增数值代码必须同时新增对应测试。 |

### 0.3 阶段总览

| 阶段 | 内容 | 预期产出 | 阻塞关系 |
|---|---|---|---|
| A | 结构化基元 + 安全卫生 | 快速 mix/unmix、ρ 的 Gram 计算 | 无 |
| B | 融合不解码路径（核心） | block 内不再出现明文 `h`；开销 ≤5% | 需 A |
| C | 补齐缺失组件 | 词表置换、SecureEmbedding/Head、Converter、MoE | 需 B |
| D | 实验与报告 | exact gate 全绿、性能指标、可引用报告 | 需 C |

---

## 1. 符号与张量约定

### 1.1 维度符号（全项目统一，写代码时用这些变量名）

```text
B      batch size
S      sequence length（当前 forward 的 query 数）
T      key/value 总长度（含 cache），T >= S
d      hidden_size                        （Llama-3.2-3B: 3072）
r      hidden_noise_dim                   （默认 16）
n      d + r                              （3088）
H      num_attention_heads                （24）
Hkv    num_key_value_heads                （8）
dh     head_dim = d / H                   （128）
rh     value_noise_dim_per_head           （默认 2）
dff    intermediate_size                  （8192）
V      vocab_size
b      basis_block_size（阶段 A 引入，默认 16）
m      n / b                              （193）
```

### 1.2 行向量数学 vs PyTorch 布局

**数学（本文档一律用行向量）：**

```text
Y = X W + bias        X: [..., in]   W: [in, out]
```

**PyTorch `F.linear`：**

```text
F.linear(x, weight, bias)   weight: [out, in]
```

转换：`weight_pt = weight_math.T.contiguous()`。仓库已有
`src/fastprove/conversion.py:math_to_torch_weight` / `torch_to_math_weight`。

**规则：所有 `*_math` 后缀的张量是数学布局 `[in, out]`；所有 `*_pt` 后缀是
PyTorch 布局 `[out, in]`。新增张量必须带后缀。** 本文档中所有部署权重
（`deployed_*`）一律是数学布局。

### 1.3 张量形状约定

```text
MixedState.mixed          [B, S, n]
Query                     [B, H,   S, dh]
Key                       [B, Hkv, T, dh]
MixedValue                [B, Hkv, T, dh + rh]
mixed_context             [B, H,   S, dh + rh]
valid_mask                [B, 1,   S, T]  (bool)
rho (RMS scale)           [B, S, 1]
```

### 1.4 核心量

| 符号 | 含义 | 形状 | 计算时机 |
|---|---|---|---|
| `M` | 混合基 `c = [h,e]M` | `[n,n]` | 离线 |
| `P` | `M⁻¹[:, :d]`，信号读出 | `[n,d]` | 离线 |
| `N` | `M⁻¹[:, d:]`，噪声读出 | `[n,r]` | 离线 |
| `M_top` | `M[:d]` | `[d,n]` | 离线 |
| `M_bot` | `M[d:]` | `[r,n]` | 离线 |
| `A_gram` | `P Pᵀ`，用于算 `‖h‖²` | `[n,n]`，块对角 | 离线 |
| `ρ` | `sqrt(‖h‖²/d + eps)` | `[B,S,1]` | 在线 |

**关键点：`P` 和 `N` 只在离线被吸收进部署权重，`forward` 里永远不出现
`c @ P`。** 唯一用到 `A_gram` 的地方是算标量 `ρ`。

---

## 2. 现状与目标

### 2.1 当前实现（阶段 A 开始前）

`src/fastprove/models/obfuscated.py:ObfuscatedDecoderBlock._run()` 的实际流程：

```text
state ──_unmix_checkpoint──► 明文 h, side_noise      ← 问题所在
  h @ attention_rotation → rms → @ q/k/v_weight_math   （明文 GEMM）
  RoPE → common_qk → attention → mixed_context
  _fused_value_unmix → 明文 context → @ o_weight_math   （明文 GEMM）
  h + attention_output ──_mix_checkpoint──► post_attention_state
post_attention_state ──_unmix_checkpoint──► 明文 ffn_signal   ← 又一次
  ... SwiGLU / down（明文 GEMM）...
  ──_mix_checkpoint──► final_state
```

每 block：**2 次完整解码 + 2 次完整重混 + 2 次稠密 d×d 旋转**。实测开销
（d=1024, 4 层, CPU FP32, seq=128）：prefill **+150.3%**，
prefill+8×decode **+184.2%**。目标是 ≤5%。

### 2.2 目标实现（阶段 B 完成后）

```text
state ──►（不解码）
  ρ = sqrt(blockwise_gram(c)/d + eps)              ← 只出标量
  q = c @ Wq_dep / ρ ; k = c @ Wk_dep / ρ ; cV = c @ Wv_dep / ρ
  RoPE → common_qk → attention → mixed_context
  c ← c + einsum(mixed_context, W_attn_out) + (c@N) @ W_nz_a + xi_a
  ρ₂ = sqrt(blockwise_gram(c)/d + eps)
  z' = silu(c @ Wg_dep / ρ₂) * (c @ Wu_dep / ρ₂)
  c ← c + z' @ W_ffn_out + (c@N) @ W_nz_f + xi_f
──► next state
```

**GEMM 数量与明文 block 完全相同。** 明文 `h`、明文 attention context `O`、
明文 FFN 输出全部不再作为张量出现。

### 2.3 每 token MAC 预算（Llama-3.2-3B: d=3072, r=16, n=3088, H=24, Hkv=8, dh=128, dff=8192）

| 项 | 明文 | 部署 | 增量 |
|---|---:|---:|---:|
| Q 投影 | 9.44M | 9.49M | +0.05M |
| K 投影 | 3.15M | 3.16M | +0.02M |
| V 投影 | 3.15M | 3.21M | +0.06M |
| 输出路径（O/unmix/residual/耦合） | 9.44M | 9.61M | +0.17M |
| Gate | 25.2M | 25.3M | +0.13M |
| Up | 25.2M | 25.3M | +0.13M |
| Down 路径（含 residual/耦合） | 25.2M | 25.3M | +0.13M |
| ρ 的 Gram（2 次） | 0 | 0.10M | +0.10M |
| 噪声进出 `(c@N)`、`@M_bot` | 0 | 0.25M | +0.25M |
| RoPE 后 C 变换（Q/K） | 0 | 0.79M | +0.79M |
| **合计（不含 attention 本体）** | **100.7M** | **102.5M** | **+1.8M ≈ +1.8%** |

KV cache：Key 不变，Value 每 head 从 `dh=128` 变 `dh+rh=130`，
整体 `(128+130)/(128+128) = +0.78%`。

**结论：算术层面 ≤5% 目标可达。** 但 eager PyTorch 参考实现由于额外小 kernel
和 Python 调度，实测仍会高于此（手册 §70 已预期）。性能报告必须区分
「reference 实现」与「fused 实现」，不得用算术预算冒充实测。

---

## 3. 全局不变量（任何时刻都必须成立）

执行者在每个任务后自查：

```text
I1  forward() 中无 inv/solve/pinv/lstsq                    → grep 检查
I2  forward() 中不出现形状为 [..., d] 的明文 hidden 张量     （阶段 B 后）
I3  残差相加两侧 basis fingerprint 相同
I4  masked 位置的 attention logit 恒为 -inf，不是大负数
I5  ρ > 0 恒成立（eps 保证），padding 行不产生 NaN
I6  exact 模式下 obfuscated logits 与 plaintext logits 在容差内一致
I7  相同 seed 两次运行产生逐位相同的指标
I8  服务端 state_dict() 不含 M / M⁻¹ / P / N / 旋转矩阵 / 逆词表
```

---

# 阶段 A：结构化基元与安全卫生

**目的**：把稠密 `M` 换成结构化 `M = Π₁ D B Π₂`（B 块对角正交），使
mix/unmix/‖h‖² 从 O(n²) 降到 O(n·b)；同时修掉密钥材料进 state_dict 的问题。
这是阶段 B 的前提——没有块对角结构，ρ 的计算会退化成 n² 的二次型。

**阶段 A 不改变任何数学结果**，只改变 `M` 的生成方式和存储方式。
exact 模式的输出在容差内应保持一致。

---

## 任务 A1：为配置增加 `basis_block_size`

### 目标
让基块宽 `b` 可配置，默认 16，并校验 `n % b == 0`。

### 涉及文件
- `src/fastprove/config.py`
- `configs/tiny_exact.yaml`
- `configs/tiny_approx.yaml`
- `configs/eval_sweep.yaml`

### 步骤

**A1.1** 在 `src/fastprove/config.py` 的 `ObfuscationConfig` 中加字段。找到：

```python
@dataclass(frozen=True)
class ObfuscationConfig:
    """Augmented-state dimensions and transform constraints."""

    hidden_noise_dim: int
    value_noise_dim_per_head: int
    max_condition_number: float
    noise_propagation_gamma: float
    refresh_mode: str
```

改成：

```python
@dataclass(frozen=True)
class ObfuscationConfig:
    """Augmented-state dimensions and transform constraints."""

    hidden_noise_dim: int
    value_noise_dim_per_head: int
    max_condition_number: float
    noise_propagation_gamma: float
    refresh_mode: str
    basis_block_size: int = 16
```

在同一个类的 `__post_init__` 末尾追加：

```python
        if self.basis_block_size <= 0:
            raise ValueError("basis_block_size must be positive")
```

**注意**：`n % b == 0` 的校验不能放在这里（`ObfuscationConfig` 不知道 `d`）。
它放在 `PrototypeConfig.__post_init__`。找到：

```python
@dataclass(frozen=True)
class PrototypeConfig:
    """Complete prototype configuration."""

    model: ModelConfig
    obfuscation: ObfuscationConfig
    attention: AttentionConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if (
            self.evaluation.sequence_length + self.evaluation.generation_tokens
            > self.model.max_sequence_length
        ):
            raise ValueError("evaluation and generation exceed model context")
```

在 `__post_init__` 末尾追加：

```python
        total = self.model.hidden_size + self.obfuscation.hidden_noise_dim
        block = self.obfuscation.basis_block_size
        if total % block != 0:
            raise ValueError(
                "basis_block_size %d must divide hidden_size + hidden_noise_dim "
                "= %d; adjust hidden_noise_dim or basis_block_size"
                % (block, total)
            )
        # Value bases always use a single dense orthogonal block: head_dim +
        # value_noise_dim_per_head is small (130 for a 3B model, ~0.12% of the
        # per-token MAC budget), so no block partition is needed there and no
        # divisibility constraint applies to it.
```

**A1.2** 三个 yaml 的 `obfuscation` 段加一行。以 `configs/tiny_exact.yaml` 为例，
现有：

```yaml
obfuscation:
  hidden_noise_dim: 8
  value_noise_dim_per_head: 2
  max_condition_number: 10.0
  noise_propagation_gamma: 0.5
  refresh_mode: fixed_debug
```

改成：

```yaml
obfuscation:
  hidden_noise_dim: 8
  value_noise_dim_per_head: 2
  max_condition_number: 10.0
  noise_propagation_gamma: 0.5
  refresh_mode: fixed_debug
  basis_block_size: 8
```

`configs/tiny_approx.yaml`、`configs/eval_sweep.yaml` 同样加
`basis_block_size: 8`。

> **为什么 tiny 用 8**：tiny 配置 `d=32, r=8`，`n=40`，`40 % 8 == 0` ✓。
>
> **Value 基不受这个约束**：它一律用单块（`block_size = dh + rh`）。
> 原因是 `dh + rh` 本身就小——实模型 `128 + 2 = 130`，一个 130×130 稠密正交块
> 每 KV head 每 token 只要 16.9K MAC，8 个 KV head 合计 135K，相对基线 113M
> 是 **0.12%**，可忽略。所以只有 hidden 基需要块对角结构。
>
> 早期草案曾要求 `block | (dh+rh)`，那是错的：tiny 配置 `dh+rh=10`、`block=8`
> 时既不整除又大于块宽，会直接无法构造。**当前校验只要求 `block | n`。**

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/ -k "config or cli_plan"
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from fastprove.config import load_config
for name in ('tiny_exact','tiny_approx'):
    c = load_config(Path('configs/%s.yaml' % name))
    print(name, 'block =', c.obfuscation.basis_block_size)
"
```

### 验收标准
- [ ] 两个 config 打印 `block = 8`
- [ ] `python -m pytest -q` 全绿
- [ ] 故意把 `basis_block_size` 改成 7 后 `load_config` 抛 `ValueError`，
      错误信息包含 `must divide`（改回 8）

### 回滚
`git checkout -- src/fastprove/config.py configs/`

---

## 任务 A2：`StructuredBasis`（代码已实现并验证，本任务是理解 + 验收）

### 状态说明

**这个任务的代码已经写好并测试通过，不需要你重新实现。** 文件：

- `src/fastprove/structured.py`（已存在，20 个测试全绿）
- `tests/test_structured_basis.py`（已存在）

你要做的是：读懂它、跑通它、理解它施加的**约束**（下面 A2.4 那条数值上界很重要，
后面阶段 B 会用到）。如果 `src/fastprove/structured.py` 在你的工作副本里不存在，
说明你的分支落后了，先 `git pull`。

### 目标（该模块提供的能力）

| 方法 | 作用 | 复杂度 | 可否用于 forward |
|---|---|---|---|
| `mix(augmented)` | `[h,e] @ M` | O(n·b) | ✅ |
| `unmix(mixed)` | `c @ M⁻¹` | O(n·b) | ❌ 仅转换/debug |
| `signal_norm_squared(mixed)` | `‖h‖²`，**不物化 `h`** | O(n·b) | ✅ |
| `rms_scale(mixed, eps)` | `sqrt(‖h‖²/d + eps)` | O(n·b) | ✅ |
| `signal_projection()` | `P = M⁻¹[:, :d]` FP64 | O(n²) | ❌ 仅离线 |
| `noise_projection()` | `N = M⁻¹[:, d:]` FP64 | O(n²) | ❌ 仅离线 |
| `signal_rows()` | `M_top = M[:d]` FP64 | O(n²) | ❌ 仅离线 |
| `noise_rows()` | `M_bot = M[d:]` FP64 | O(n²) | ❌ 仅离线 |
| `descriptor` | 公开元数据，无矩阵材料 | — | ✅ |

后四个是阶段 B 构建部署权重时用的，**只在 `__init__`/转换期调用，绝不在
`forward` 里调用**。

### A2.1 为什么必须是块对角

`M = Π₁ D B Π₂`，`B` 块对角正交（块宽 `b`）：

- `κ₂(M) = κ₂(D)`——置换和块正交因子的条件数都是 1，所以条件数完全由对角缩放
  的对数范围控制，`max_condition_number` 仍然生效
- 应用 `M` 或 `M⁻¹`：gather → 逐元素乘 → `m` 个 `b×b` GEMM → gather，
  即 O(n·b)。`n=3088, b=16` 时约 49K MAC，稠密要 9.5M，**快 194 倍**
- **关键**：`A_gram = P Pᵀ` 在 `perm_out` 序下**恰好块对角**（FP64 实测
  off-block max = 0.0）。所以 `‖h‖² = c A_gram cᵀ` 可以 blockwise 算，
  每块只产生一个标量，**`h` 从头到尾没有作为张量出现**

推导见附录 A.1、A.2。

### A2.2 单块回退（tiny 配置和 Value 基必须用到）

`generate_structured_basis` 中：

```python
    effective_block = block_size if total > block_size else total
```

当 `total <= block_size` 时退化为一个稠密正交块。这让 Value 基
（`dh + rh`，tiny 配置下是 `8 + 2 = 10`）在 `block_size=16` 时仍然合法。
**不要移除这个回退**，否则 tiny 配置无法构造。

### A2.3 指纹与 dtype 的顺序陷阱（已修复，说明原因）

`_fingerprint` 把浮点因子统一 canonical 到 FP64 再哈希，且
`generate_structured_basis` **先转成部署 dtype、再算指纹**：

```python
    stored_scales = scales.to(dtype=dtype)
    stored_blocks = blocks.to(dtype=dtype)
    fingerprint = _fingerprint(perm_in, stored_scales, stored_blocks, ...)
```

如果顺序反了（先哈希 FP64、再存 FP32），`__post_init__` 的指纹校验会直接抛
`basis fingerprint does not match its factors`。`gram_blocks` 是派生量，
**不参与指纹**，所以两步构造（先造 placeholder 求 `P`，再造带 Gram 的正式对象）
是自洽的。

同理，两处容差必须跟随存储 dtype，**不要改成固定值**：

```python
        atol = 1e-10 if self.blocks.dtype == _FP64 else 5e-6      # 正交性
        atol = 1e-9  if self.blocks.dtype == _FP64 else 1e-4      # 往返
```

### A2.4 ⚠️ 数值上界：`‖e‖/‖h‖` 必须有界（阶段 B 会依赖）

**这是本方案最容易被忽略、后果最严重的约束。手册没有提到它。**

`A_gram` 是通过**抵消**把噪声子空间消掉的（`A_gram` 半正定、秩 `d`、核空间
正好是噪声方向）。抵消在 FP32 下有精度代价，且误差按 `(‖e‖/‖h‖)²` 增长。
实测（d=64, r=8, b=8, FP64 基）：

| `‖e‖/‖h‖` | 二次型 FP32 相对误差 | 二次型 FP64 | 因子式 FP32 |
|---:|---:|---:|---:|
| 1 | 3.5e-7 | 7.5e-16 | 1.0e-7 |
| 10 | 1.5e-6 | 4.1e-15 | 2.1e-7 |
| 100 | **1.3e-4** | 3.7e-13 | 1.6e-6 |
| 1e3 | **2.0e-2** | 4.3e-11 | 1.1e-5 |
| 1e4 | **2.9e+0** | 1.6e-8 | 1.6e-4 |

复现命令在 `tests/test_structured_basis.py::test_gram_accuracy_degrades_with_noise`。

**三个选项，本方案选第一个：**

| 选项 | 精度 | 是否物化 `h` | 是否可用 |
|---|---|---|---|
| **二次型 FP32 + 幅度上界（本方案）** | `‖e‖/‖h‖ ≤ 30` 时 ≤1e-4 | ❌ 不物化 | ✅ 默认 |
| 因子式 FP32（`y = cP`，`‖h‖²=‖y‖²`） | 线性退化，更稳 | ⚠️ **物化 `h` 的缩放置换** | 仅在融合 kernel 内（`y` 留寄存器） |
| 二次型 FP64 | 极稳 | ❌ 不物化 | MPS 不支持，且慢 |

因子式看起来更好，但 `y = c P` 的非零坐标**就是 `h` 的缩放置换**——在 eager
参考实现里它是一个真实张量，直接违背阶段 B 的目标。所以默认走二次型，
用上界约束噪声幅度。

模块导出了强制手段：

```python
from fastprove.structured import (
    AUXILIARY_MAGNITUDE_BOUND,      # = 30.0
    check_auxiliary_magnitude,       # 超界抛 ValueError
)
```

**阶段 B 的转换期必须调用 `check_auxiliary_magnitude`**（见任务 B1）。
当前默认耦合尺度（`C` scale 0.02、`γ=0.5`）下实际比值约 0.05，离上界很远，
所以这条约束现在不影响任何东西——但它必须写死在代码里，否则将来有人为了
"更强混淆"把噪声尺度调大 1000 倍时，`ρ` 会**静默失准并污染信号路径**，
而不是只污染噪声路径。

> **不得把 `AUXILIARY_MAGNITUDE_BOUND` 调大来让某个测试通过。**
> 要调必须重新实测上表并更新 docstring 里的表格。

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_structured_basis.py -v
python -m pytest -q
python docs/design_identity_check.py
```

### 验收标准
- [ ] `tests/test_structured_basis.py` 20 个测试全部通过
- [ ] `test_signal_norm_ignores_auxiliary_state_inside_the_validated_bound` 通过
      —— 这条证明 `ρ` 真的只依赖信号
- [ ] `test_gram_accuracy_degrades_with_noise` 通过
      —— 这条把上界的实测依据钉在测试里
- [ ] `python -m pytest -q` 全绿
- [ ] 你能用自己的话回答：为什么 `forward` 里可以调 `rms_scale` 但不能调
      `signal_projection`？（答案：前者 O(n·b) 且只出标量；后者是 `[n,d]` 稠密
      矩阵，且 `c @ P` 就是明文 `h`）

### 回滚
该任务不改代码，无需回滚。

---

## 任务 A3：密钥材料不得进入服务端 `state_dict`

### 目标
`attention_rotation`、`ffn_rotation`、`common_qk` 当前用
`register_buffer(...)` 注册（默认 `persistent=True`），会进 `state_dict()`。
它们是密钥材料。改为 `persistent=False` 并加测试断言。

### 为什么
`docs/threat_model.md` §5bis.6(a) 已记录该问题。手册 §77 要求生产模式删除
解码矩阵。这是独立于 `M` 可逆性的泄漏通道。

### 涉及文件
- `src/fastprove/models/obfuscated.py`
- 新建 `tests/test_key_material_isolation.py`

### 前置条件
无（可与 A2 并行）。

### 步骤

**A3.1** 在 `src/fastprove/models/obfuscated.py` 的
`ObfuscatedDecoderBlock.__init__` 中找到（约 307–310 行）：

```python
        self.register_buffer("attention_rotation", attention_rotation)
        self.register_buffer("ffn_rotation", ffn_rotation)
        self.register_buffer("common_qk", common_qk)
```

改为：

```python
        # Key material: must not enter the server-side state_dict.
        self.register_buffer(
            "attention_rotation", attention_rotation, persistent=False
        )
        self.register_buffer("ffn_rotation", ffn_rotation, persistent=False)
        self.register_buffer("common_qk", common_qk, persistent=False)
```

> **阶段 B 之后**：`attention_rotation` 和 `ffn_rotation` 会被完全删除
> （它们的作用已被吸收进部署权重，在线是恒等的空转）。`common_qk` 保留，
> 因为它必须在 RoPE 之后应用，无法吸收。

**A3.2** 同一文件中，噪声耦合矩阵也是密钥材料。找到所有这些
`register_buffer`（约 396–560 行区间）并加 `persistent=False`：

```text
value_signal_coupling
value_side_propagator
value_fixed_refresh
attention_noise_propagator
swiglu_noise_propagator
down_noise_propagator
attention_noise_coupling
attention_aux_to_hidden
swiglu_noise_coupling
down_noise_coupling
attention_fixed_refresh
swiglu_fixed_refresh
down_fixed_refresh
```

在 `ObfuscatedTinyCausalLM.__init__` 中同样处理：

```text
initial_noise_coupling
initial_fixed_refresh
```

改法示例（其余照此模式）：

```python
        self.register_buffer(
            "value_signal_coupling",
            torch.stack([...]),
            persistent=False,
        )
```

**保留 `persistent=True`（即不加参数）的**：`kv_index`、
`q/k/v/o_weight_math`、`q/k/v_bias`、`gate/up/down_weight_math`、
`embedding_weight`、`final_norm_weight`、`lm_head_weight`。这些是部署权重，
服务端需要它们。

**A3.3** 新建 `tests/test_key_material_isolation.py`：

```python
"""Server-side state_dict must not contain key material."""

from __future__ import annotations

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM

_FORBIDDEN_SUBSTRINGS = (
    "rotation",
    "common_qk",
    "coupling",
    "propagator",
    "refresh",
    "gram",
    "perm_in",
    "perm_out",
    "scales",
    "blocks",
    "inverse",
)


def _converted():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=16,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=8,
    )
    plain = PlainTinyCausalLM(config, seed=1, debug_enabled=False)
    return ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=1,
        debug_enabled=False,
    ).module


def test_state_dict_excludes_key_material() -> None:
    keys = list(_converted().state_dict().keys())
    offending = [
        key
        for key in keys
        for token in _FORBIDDEN_SUBSTRINGS
        if token in key.lower()
    ]
    assert offending == [], "key material leaked into state_dict: %s" % offending


def test_state_dict_still_contains_deployed_weights() -> None:
    keys = " ".join(_converted().state_dict().keys())
    for required in ("embedding_weight", "lm_head_weight", "kv_index"):
        assert required in keys


def test_saved_and_reloaded_state_dict_round_trips(tmp_path) -> None:
    module = _converted()
    path = tmp_path / "server.pt"
    torch.save(module.state_dict(), path)
    reloaded = torch.load(path, weights_only=True)
    assert set(reloaded.keys()) == set(module.state_dict().keys())
```

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_key_material_isolation.py -v
python -m pytest -q
python - <<'PY'
import sys; sys.path.insert(0, 'src')
import torch
from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
cfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_layers=1, num_attention_heads=4, num_key_value_heads=2,
                  max_sequence_length=16)
obf = ObfuscationConfig(hidden_noise_dim=8, value_noise_dim_per_head=2,
                        max_condition_number=10.0, noise_propagation_gamma=0.5,
                        refresh_mode="per_request", basis_block_size=8)
p = PlainTinyCausalLM(cfg, seed=1, debug_enabled=False)
m = ObfuscatedTinyCausalLM.from_plain(p, obfuscation=obf, mode=AttentionMode.EXACT,
        approximation=None, seed=1, debug_enabled=False).module
print("state_dict keys:")
for k in m.state_dict(): print("  ", k)
PY
```

### 验收标准
- [ ] 打印的 key 列表中**不含** rotation / common_qk / coupling / propagator / refresh
- [ ] 列表中**含** `embedding_weight`、`lm_head_weight`、各 `*_weight_math`
- [ ] `python -m pytest -q` 全绿（若某测试依赖这些 buffer 出现在 state_dict 中，
      改测试，不改 `persistent=False`——密钥不进 checkpoint 是硬要求）

### 回滚
`git checkout -- src/fastprove/models/obfuscated.py && rm tests/test_key_material_isolation.py`

---

## 任务 A4：checkpoint 算术统一为 FP32

### 目标
`_checkpoint_compute_dtype` 在 CPU 上返回 FP64，是实测开销的三分之二来源
（+150% → +51%）。改为一律 FP32，并把 exact 容差的实际变化记录下来。

### 为什么
AGENTS.md 只要求「离线求逆和条件数检查用 FP64/FP32」，不要求在线 FP64。
在线 FP64 在 MPS 上根本不支持，在 CUDA 上是错误默认。

### 涉及文件
- `src/fastprove/models/obfuscated.py`
- `docs/implementation_notes.md`

### 前置条件
任务 A3 完成（同一文件，避免冲突）。

### 步骤

**A4.1** 先测量改动前的 exact 误差作为基线：

```bash
cd /Users/yin/code/fastProve
PYTHONPATH=src python - <<'PY' | tee /tmp/exact_before.txt
import torch
from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
torch.manual_seed(0)
cfg = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_sequence_length=32)
obf = ObfuscationConfig(hidden_noise_dim=8, value_noise_dim_per_head=2,
                        max_condition_number=10.0, noise_propagation_gamma=0.5,
                        refresh_mode="per_request", basis_block_size=8)
p = PlainTinyCausalLM(cfg, seed=7, debug_enabled=False).eval()
m = ObfuscatedTinyCausalLM.from_plain(p, obfuscation=obf, mode=AttentionMode.EXACT,
        approximation=None, seed=7, debug_enabled=False).module.eval()
ids = torch.randint(0, cfg.vocab_size, (2, 16))
ctx = RequestContext(global_seed=7, request_id="tol")
with torch.no_grad():
    a = p(ids); b = m(ids, request_context=ctx)
print("exact max abs err = %.6e" % (a-b).abs().max().item())
print("exact rel L2      = %.6e" % ((a-b).norm()/a.norm()).item())
PY
```

**A4.2** 在 `src/fastprove/models/obfuscated.py` 找到：

```python
def _checkpoint_compute_dtype(device: torch.device | str) -> torch.dtype:
    """Choose a runtime checkpoint dtype supported by the requested device.
    ...
    """

    return torch.float64 if torch.device(device).type == "cpu" else torch.float32
```

替换整个函数为：

```python
def _checkpoint_compute_dtype(device: torch.device | str) -> torch.dtype:
    """Return the runtime checkpoint arithmetic dtype.

    Offline basis generation, inversion and conditioning checks remain FP64
    (see ``structured.py``). Runtime arithmetic is FP32 on every device:

    * MPS does not implement FP64 tensors at all.
    * FP64 on CPU accounted for roughly two thirds of the measured reference
      overhead and is not required by any accuracy gate.
    * AGENTS.md requires FP32 or better for RMS statistics, margins, clipping
      and Softmax reductions; FP32 satisfies that.

    ``device`` is accepted for interface stability and forward compatibility.
    """

    del device
    return torch.float32
```

**A4.3** 重新测量并记录：

```bash
cd /Users/yin/code/fastProve
PYTHONPATH=src python - <<'PY' | tee /tmp/exact_after.txt
# 与 A4.1 完全相同的脚本
PY
diff /tmp/exact_before.txt /tmp/exact_after.txt || true
```

**A4.4** 把两个数字写进 `docs/implementation_notes.md`，在文件末尾追加一节：

```markdown
## 阶段 A：checkpoint 算术统一为 FP32（任务 A4）

`_checkpoint_compute_dtype` 原先在 CPU 上返回 FP64。实测该选择占参考实现
开销的约三分之二（d=1024/4 层/CPU/seq=128 prefill：FP64 +150.3%，
FP32 +50.7%）。离线基生成、求逆与条件数检查仍为 FP64。

exact 模式容差实测变化（tiny：d=32, r=8, 2 层, seq=16, 2 batch, seed=7）：

| checkpoint dtype | max abs err | rel L2 |
|---|---:|---:|
| FP64（改动前） | 见 results/raw/exact_tolerance_A4.json | |
| FP32（改动后） | 见 results/raw/exact_tolerance_A4.json | |

数据文件由 `scripts/record_exact_tolerance.py` 生成，未手工填写。
```

**A4.5** 为了让上面的表格有可复核的原始文件，新建
`scripts/record_exact_tolerance.py`：

```python
"""Record exact-mode tolerance for the current checkpoint arithmetic dtype.

Usage:
    python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_A4.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import (
    ObfuscatedTinyCausalLM,
    _checkpoint_compute_dtype,
)
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=8,
    )
    plain = PlainTinyCausalLM(
        config, seed=arguments.seed, debug_enabled=False
    ).eval()
    obfuscated = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=arguments.seed,
        debug_enabled=False,
    ).module.eval()

    torch.manual_seed(arguments.seed)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    context = RequestContext(global_seed=arguments.seed, request_id="tolerance")
    with torch.no_grad():
        reference = plain(input_ids)
        observed = obfuscated(input_ids, request_context=context)
    difference = reference - observed

    record = {
        "checkpoint_compute_dtype": str(_checkpoint_compute_dtype("cpu")),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
        "model": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "hidden_noise_dim": obfuscation.hidden_noise_dim,
            "basis_block_size": obfuscation.basis_block_size,
        },
        "exact_max_absolute_error": float(difference.abs().max()),
        "exact_relative_l2_error": float(
            difference.norm() / reference.norm()
        ),
        "nan_count": int(torch.isnan(observed).sum()),
        "inf_count": int(torch.isinf(observed).sum()),
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
```

跑一次：

```bash
cd /Users/yin/code/fastProve
PYTHONPATH=src python scripts/record_exact_tolerance.py \
    --output results/raw/exact_tolerance_A4.json
```

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q
cat results/raw/exact_tolerance_A4.json
grep -rn "float64" src/fastprove/models/obfuscated.py
```

### 验收标准
- [ ] `python -m pytest -q` 全绿
- [ ] `results/raw/exact_tolerance_A4.json` 中
      `checkpoint_compute_dtype == "torch.float32"`
- [ ] `exact_max_absolute_error` 仍满足现有测试容差（`tests/test_tiny_lm.py`
      用 `atol=2e-4`）。**若超出：不要放宽容差**，记录数值并上报——
      这说明基条件数或噪声尺度需要调，属于设计问题
- [ ] `nan_count == 0` 且 `inf_count == 0`
- [ ] `grep float64` 在 `obfuscated.py` 中不再出现于 checkpoint 路径

### 回滚
`git checkout -- src/fastprove/models/obfuscated.py`

---

## 任务 A5：把结构化基接入现有 block（保持数学不变）

### 目标
让 `ObfuscatedDecoderBlock` / `ObfuscatedTinyCausalLM` 用
`StructuredBasis` 替换 `BasisTransform`，**但仍走当前的解码-重混流程**。
这是纯替换任务，输出应在容差内不变。阶段 B 才改流程。

### 为什么
把「换基元」和「改流程」分成两个任务，任一步失败时可以独立定位。

### 涉及文件
- `src/fastprove/models/obfuscated.py`
- `src/fastprove/state.py`（`encode_debug` / `decode_debug` 接受两种基）

### 前置条件
任务 A2、A3、A4 全部完成。

### 步骤

**A5.1** 在 `src/fastprove/state.py` 中让 debug 编解码同时支持两种基类型。
把 `encode_debug` 和 `decode_debug` 的 `transform` 参数类型放宽，并用鸭子类型
调用。找到 `encode_debug`：

```python
def encode_debug(
    signal: torch.Tensor,
    noise: torch.Tensor,
    transform: BasisTransform,
    *,
    enabled: bool,
) -> MixedState:
```

改签名为（顶部 import 加 `from typing import Protocol, Tuple, Union`）：

```python
class _MixingBasis(Protocol):
    """Structural protocol shared by BasisTransform and StructuredBasis."""

    signal_dim: int
    noise_dim: int

    @property
    def descriptor(self) -> BasisDescriptor: ...

    def validate_integrity(self) -> None: ...


def encode_debug(
    signal: torch.Tensor,
    noise: torch.Tensor,
    transform: _MixingBasis,
    *,
    enabled: bool,
) -> MixedState:
```

函数体里把矩阵乘替换为多态调用。原有：

```python
    matrix = transform.matrix.to(device=signal.device, dtype=signal.dtype)
    augmented = torch.cat((signal, noise), dim=-1)
    return MixedState(augmented @ matrix, transform.descriptor)
```

改为：

```python
    augmented = torch.cat((signal, noise), dim=-1)
    if hasattr(transform, "mix"):
        mixed = transform.mix(augmented)
    else:
        matrix = transform.matrix.to(device=signal.device, dtype=signal.dtype)
        mixed = augmented @ matrix
    return MixedState(mixed, transform.descriptor)
```

同理 `decode_debug` 中：

```python
    inverse = transform.inverse.to(
        device=state.mixed.device, dtype=state.mixed.dtype
    )
    augmented = state.mixed @ inverse
```

改为：

```python
    if hasattr(transform, "unmix"):
        augmented = transform.unmix(state.mixed)
    else:
        inverse = transform.inverse.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        augmented = state.mixed @ inverse
```

**A5.2** 在 `src/fastprove/models/obfuscated.py` 中，把
`_make_hidden_checkpoint` 改为接受 `StructuredBasis`：

```python
def _make_hidden_checkpoint(basis, *, signal_dimension: int):
    """Capture a basis inside the designated fused-checkpoint reference.

    The captured factors are deliberately absent from ``state_dict`` and
    ``named_buffers``. Python closure inspection, hooks, or modified kernels
    remain outside the prototype threat model.

    ``basis`` is a ``StructuredBasis``. Both directions cost ``O(n*b)``.
    """

    descriptor = basis.descriptor

    def mix(signal: torch.Tensor, noise: torch.Tensor) -> MixedState:
        compute_dtype = _checkpoint_compute_dtype(signal.device)
        augmented = torch.cat((signal, noise), dim=-1).to(dtype=compute_dtype)
        return MixedState(
            basis.mix(augmented).to(dtype=signal.dtype), descriptor
        )

    def unmix(state: MixedState) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.basis != descriptor:
            raise ValueError("input mixed state basis does not match checkpoint")
        compute_dtype = _checkpoint_compute_dtype(state.mixed.device)
        augmented = basis.unmix(state.mixed.to(dtype=compute_dtype)).to(
            dtype=state.mixed.dtype
        )
        return (
            augmented[..., :signal_dimension],
            augmented[..., signal_dimension:],
        )

    return mix, unmix
```

**A5.3** `_make_value_checkpoint` 同样改为接受 `Tuple[StructuredBasis, ...]`：

```python
def _make_value_checkpoint(bases: Tuple["StructuredBasis", ...]):
    """Capture per-KV-head Value codecs outside persistent module state."""

    def mix(value_augmented: torch.Tensor) -> torch.Tensor:
        compute_dtype = _checkpoint_compute_dtype(value_augmented.device)
        source = value_augmented.to(dtype=compute_dtype)
        mixed = torch.stack(
            [
                bases[head].mix(source[:, head])
                for head in range(len(bases))
            ],
            dim=1,
        )
        return mixed.to(dtype=value_augmented.dtype)

    def unmix(
        mixed_context: torch.Tensor, kv_index: torch.Tensor
    ) -> torch.Tensor:
        compute_dtype = _checkpoint_compute_dtype(mixed_context.device)
        source = mixed_context.to(dtype=compute_dtype)
        index = kv_index.detach().cpu().tolist()
        unmixed = torch.stack(
            [
                bases[int(index[head])].unmix(source[:, head])
                for head in range(len(index))
            ],
            dim=1,
        )
        return unmixed.to(dtype=mixed_context.dtype)

    return mix, unmix
```

> **性能说明**：这里的 Python 循环在阶段 A 是可接受的（Value 基是单块，
> 循环长度是 `Hkv` 或 `H`）。阶段 B 会用一次 `einsum` 的部署权重完全取代它。

**A5.4** 替换基的生成。在 `ObfuscatedDecoderBlock.__init__` 中找到：

```python
        value_transforms = tuple(
            generate_transform(
                head_dim,
                value_noise,
                seed=seed,
                domain="block-%d-value-%d" % (self.layer_id, head),
                max_condition_number=obfuscation.max_condition_number,
            )
            for head in range(kv_heads)
        )
```

改为：

```python
        value_transforms = tuple(
            generate_structured_basis(
                head_dim,
                value_noise,
                seed=seed,
                domain="block-%d-value-%d" % (self.layer_id, head),
                block_size=obfuscation.basis_block_size,
                max_condition_number=obfuscation.max_condition_number,
            )
            for head in range(kv_heads)
        )
```

在 `ObfuscatedDecoderBlock.from_plain` 和
`ObfuscatedTinyCausalLM.from_plain` 中，找到 `generate_transform(` 调用并
同样替换为 `generate_structured_basis(`，加上
`block_size=obfuscation.basis_block_size`。

顶部 import 加：

```python
from ..structured import StructuredBasis, generate_structured_basis
```

并把类型标注 `hidden_transform: BasisTransform` 改为
`hidden_transform: StructuredBasis`（`ObfuscatedDecoderBlock.__init__`、
`ObfuscatedTinyCausalLM.__init__`、两个 `from_plain`、
`ObfuscatedBlockClient.__init__`）。

**A5.5** `value_basis_condition_numbers` / `value_basis_fingerprints` 的取值
方式不变（`StructuredBasis` 有同名属性）。确认这两行仍能工作：

```python
        self.value_basis_fingerprints = tuple(
            item.fingerprint for item in value_transforms
        )
        self.value_basis_condition_numbers = tuple(
            float(item.condition_number) for item in value_transforms
        )
```

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q
PYTHONPATH=src python scripts/record_exact_tolerance.py \
    --output results/raw/exact_tolerance_A5.json
python - <<'PY'
import json
before = json.load(open('results/raw/exact_tolerance_A4.json'))
after  = json.load(open('results/raw/exact_tolerance_A5.json'))
print("A4 max abs = %.3e" % before["exact_max_absolute_error"])
print("A5 max abs = %.3e" % after["exact_max_absolute_error"])
PY
```

并测量开销变化：

```bash
cd /Users/yin/code/fastProve
PYTHONPATH=src python - <<'PY'
import time, torch
from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
torch.set_num_threads(4)
cfg = ModelConfig(vocab_size=2048, hidden_size=1024, intermediate_size=2752,
                  num_layers=4, num_attention_heads=16, num_key_value_heads=4,
                  max_sequence_length=256)
obf = ObfuscationConfig(hidden_noise_dim=16, value_noise_dim_per_head=2,
                        max_condition_number=10.0, noise_propagation_gamma=0.5,
                        refresh_mode="per_request", basis_block_size=16)
p = PlainTinyCausalLM(cfg, seed=1, debug_enabled=False).eval()
m = ObfuscatedTinyCausalLM.from_plain(p, obfuscation=obf, mode=AttentionMode.EXACT,
        approximation=None, seed=1, debug_enabled=False).module.eval()
ctx = RequestContext(global_seed=1, request_id="b")
ids = torch.randint(0, cfg.vocab_size, (1,128))
def bench(fn, n=5):
    with torch.no_grad():
        fn(); ts=[]
        for _ in range(n):
            t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)
a=bench(lambda: p(ids)); b=bench(lambda: m(ids, request_context=ctx))
print("prefill plain %.4f  obf %.4f  overhead %+.1f%%" % (a,b,100*(b/a-1)))
PY
```

### 验收标准
- [ ] `python -m pytest -q` 全绿
- [ ] A5 的 `exact_max_absolute_error` 与 A4 同量级（不要求逐位相同，
      基的生成方式变了；要求仍在 `tests/test_tiny_lm.py` 的 `atol=2e-4` 内）
- [ ] prefill overhead 相比阶段 A 开始前（+150.3%）有明显下降。
      **预期落在 +30% ~ +60%**（仍走解码-重混流程，所以不会到 5%）
- [ ] `grep -n "generate_transform(" src/fastprove/models/obfuscated.py` 为空

### 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `basis_block_size` 不能整除 Value 基 | tiny 配置 `dh+rh=10` | `generate_structured_basis` 内 `effective_block` 已处理；确认你抄的是 A2 的完整版本 |
| exact 误差跳到 1e-2 量级 | `perm_in`/`perm_out` 在 mix/unmix 中方向搞反 | 跑 `tests/test_structured_basis.py::test_unmix_inverts_mix_exactly` 定位 |
| `state.basis != descriptor` | 指纹算法变了但缓存的 cache_identity 没更新 | 属正常；确认没有跨转换复用 cache |

### 回滚
`git checkout -- src/fastprove/models/obfuscated.py src/fastprove/state.py`

---

## 阶段 A 出口检查

全部任务完成后，逐条确认：

```bash
cd /Users/yin/code/fastProve
python docs/design_identity_check.py                    # 13 项恒等式
python -m pytest -q                                      # 全绿
grep -rn "linalg.inv\|linalg.solve\|pinv\|lstsq" src/fastprove/models/  # 应为空
git log --oneline -5                                     # 应见 A1..A5 五个 commit
```

- [ ] 五个任务各一个 commit
- [ ] `results/raw/exact_tolerance_A4.json`、`exact_tolerance_A5.json` 存在
- [ ] `docs/implementation_notes.md` 记录了 FP64→FP32 的容差变化
- [ ] prefill overhead 已从 +150% 降到 +60% 以内
- [ ] 服务端 `state_dict` 不含任何密钥材料

**阶段 A 未全部通过，不要开始阶段 B。** 阶段 B 依赖
`StructuredBasis.rms_scale` 和 `signal_norm_squared` 的正确性。

---

# 阶段 B：融合不解码路径（核心）

**目的**：让 block 内部不再出现明文 `h`、明文 attention context `O`、明文 FFN
输出。做法是把 `M⁻¹`、`M`、RMSNorm 的 `γ` 全部离线吸收进部署权重，在线只剩
「同样数量的 GEMM + 一个标量 `ρ` + 两条 `r` 宽的噪声通道」。

**阶段 B 不改变任何数学结果。** exact 模式输出应保持在容差内。变的是
「服务器看到什么」和「跑多快」。

## B0 先读这个：部署权重的完整定义

下面 8 个矩阵是阶段 B 的全部内容。**它们在 `__init__`/转换期算一次，
`forward` 里只做乘法。**

记 `P = M⁻¹[:, :d]`，`N = M⁻¹[:, d:]`，`M_top = M[:d]`，`M_bot = M[d:]`。

| 部署权重 | 定义 | 形状 | 在线用法 |
|---|---|---|---|
| `Wq` | `P diag(γ_a) W_Q` | `[n, H·dh]` | `q = (c @ Wq)/ρ + b_q` |
| `Wk` | `P diag(γ_a) W_K` | `[n, Hkv·dh]` | `k = (c @ Wk)/ρ + b_k` |
| `Wv[j]` | `[P diag(γ_a) W_V^(j) ∣ C_V^(j)] M_V^(j)` | `[Hkv, n, dh+rh]` | `c_V = (c @ Wv)/ρ + b_v` |
| `Wattn[i]` | `M_V⁻¹[kv(i)][:, :dh] W_O^(i) (M_top + C_O M_bot)`<br>`+ (1/H) M_V⁻¹[kv(i)][:, dh:] A_ux M_bot` | `[H, dh+rh, n]` | `c += einsum(mixed_ctx, Wattn)` |
| `Wnz_a` | `(G_O − I) M_bot` | `[r, n]` | `c += (c @ N) @ Wnz_a` |
| `Wg` | `P diag(γ_f) W_g[:, P_f]` | `[n, dff]` | `g' = (c @ Wg)/ρ₂` |
| `Wu` | `P diag(γ_f) (W_u[:, P_f] ⊙ D_f[P_f])` | `[n, dff]` | `u' = (c @ Wu)/ρ₂` |
| `Wffn` | `(D_f[P_f]⁻¹ W_d[P_f]) (M_top + C_d M_bot) + C_z M_bot` | `[dff, n]` | `c += z' @ Wffn` |
| `Wnz_f` | `(G_d − I) M_bot` | `[r, n]` | `c += (c @ N) @ Wnz_f` |

**为什么 `Wnz` 里是 `G − I` 而不是 `G`**：残差写法是 `c_next = c + Δ`，而
`c` 里已经含有 `e M_bot` 这一项。要让噪声从 `e` 变成 `eG`，增量必须是
`e(G − I) M_bot`。漏掉 `−I` 会让噪声变成 `e(1+G)`，链式衰减失效。

**`Wattn` 里的 `(1/H)`**：`H` 个 query head 各自贡献一份辅助项，求和后正好是
均值。等价于原实现的 `context_auxiliary.mean(1)`。

推导与逐项验证见附录 A.3。

---

## 任务 B1：部署权重构建器（代码已实现，本任务是理解 + 验收）

### 状态说明

**代码已写好并测试通过。** 文件：

- `src/fastprove/layers/deployed.py`（已存在）
- `tests/test_deployed_weights.py`（已存在，17 个测试全绿）

### 提供的接口

```python
from fastprove.layers.deployed import (
    build_deployed_attention,       # -> DeployedAttentionWeights
    build_deployed_feed_forward,    # -> DeployedFeedForwardWeights
    validate_auxiliary_budget,      # 转换期必须调用
)
```

两个构建器都接受 `dtype` 参数（默认 FP32 部署；测试用 FP64 隔离代数误差）。
内部一律 FP64 计算后再转 `dtype`。

### ⚠️ B1.1 精度瓶颈是 `ρ`，不是代数

实测（`tests/test_deployed_weights.py::test_rho_is_the_precision_bottleneck_not_the_algebra`）：

| 量 | 相对误差 |
|---|---:|
| `ρ`（FP32 Gram 归约，**部署路径**） | 1.9e-7 |
| `ρ`（FP64 Gram 归约，仅作对照） | 2.4e-16 |

FP32 归约是 AGENTS.md R5 的要求，**不是缺陷**。后果：

```text
整个 block 恒等式的可达容差 ≈ 1e-5（相对），不可能更好
```

因此 `tests/test_deployed_weights.py` 里 `_TOL = 1e-5`。
**不得把这个容差调松来掩盖真实误差**；如果你的实现误差超过 1e-5，是实现错了。

### ⚠️ B1.2 转换期必须调用 `validate_auxiliary_budget`

```python
validate_auxiliary_budget(
    basis=hidden_basis,
    sample_signal=<一批真实 embedding 或 hidden>,
    signal_noise_coupling=C_O,
    noise_propagator=G_O,
    context="block-%d-attention" % layer_id,
)
```

它做两件事：

1. 检查 `‖G‖₂ < 1`（噪声链必须收缩，否则 `e` 发散）
2. 估计稳态 `‖e‖/‖h‖ ≈ ‖hC‖/(1−‖G‖₂)`，并对照
   `AUXILIARY_MAGNITUDE_BOUND = 30.0`

超界抛 `ValueError`。原因见任务 A2.4：超过上界后 `ρ` 会**静默失准并污染信号
路径**。当前默认尺度下实际比值 < 1，离上界很远。

### ⚠️ B1.3 必须记录的安全后果

`Wnz` 的在线用法是 `(c @ N) @ Wnz`，所以 **`N` 进入了部署权重**。
我验证过这不可规避：`Z = N(G−I)M_bot` 的秩恒为 `r`，且对任意分解
`Z = U V`，`M @ U` 的信号块为 0、噪声块满秩。也就是说：

```text
观察部署权重的人可以恢复 e（up to 一个可逆 r×r 映射）
```

这**不会**额外暴露 `h`（`h` 本来就被 `M` 完全决定），但它让
`docs/threat_model.md` 5bis.6 记录的 `e ≈ hC` 通道**变得直接可利用**。

**任务 D4 必须把这一条写进威胁模型。** 不得省略。

唯一的替代方案是令 `G = 0`（噪声每层从信号重新生成，不做链式传播），
这样就不需要 `N`。但那放弃了手册 §12 的链式衰减性质。**本方案保留 `G = γP`
并接受这个泄漏，因为它不比已有结论更弱。**

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_deployed_weights.py -v
python -m pytest -q
```

### 验收标准
- [ ] 17 个测试全绿
- [ ] `test_block_output_decodes_to_the_plaintext_block_output` 通过
      —— 这是阶段 B 的核心恒等式
- [ ] `test_block_output_is_independent_of_the_incoming_auxiliary_state` 通过
      —— 证明信号路径不依赖 `e`
- [ ] `test_auxiliary_state_actually_changes_the_mixed_state` 通过
      —— 反向哨兵：证明噪声路径没有被静默置零
- [ ] 你能回答：为什么 `Wnz` 是 `(G − I) M_bot` 而不是 `G M_bot`？

### 回滚
不改代码，无需回滚。

---

## 任务 B2：重写 `ObfuscatedDecoderBlock`（attention 段）

### 目标
把 `_run()` 的 attention 段从「解码 → 明文 GEMM → 重混」改成「直接用部署权重」。

### 涉及文件
- `src/fastprove/models/obfuscated.py`

### 前置条件
阶段 A 全部完成；任务 B1 的测试全绿。

### 步骤

**B2.1** 在 `ObfuscatedDecoderBlock.__init__` 中，**删除**下列 buffer 注册
（它们的作用已被部署权重吸收，在线是纯空转）：

```text
attention_rotation      ← 删除。RMS 旋转不变 + R^T 已吸收进权重，在线乘 R 是恒等空转
ffn_rotation            ← 删除，同上
q_weight_math           ← 删除，被 deployed.query 取代
k_weight_math           ← 删除
v_weight_math           ← 删除
o_weight_math           ← 删除，被 deployed.output 取代
gate_weight_math        ← 删除，被 deployed.gate 取代
up_weight_math          ← 删除
down_weight_math        ← 删除，被 deployed.output 取代
```

**保留** `common_qk`（必须在 RoPE 之后应用，无法吸收）、`kv_index`、
`q_bias`/`k_bias`/`v_bias`。

> **为什么可以删 `attention_rotation`**：原实现算
> `rms_no_gamma(h @ R) @ (Rᵀ diag(γ) W_Q)`。因为 RMS 对正交变换不变，
> `rms_no_gamma(hR) = rms_no_gamma(h) R`，所以整体等于
> `rms_no_gamma(h) R Rᵀ diag(γ) W_Q = rms_no_gamma(h) diag(γ) W_Q`。
> 两次 `d×d` 稠密 GEMM 完全抵消。这是纯浪费。

**B2.2** 在 `__init__` 中改为构建并注册部署权重。在生成 `value_transforms`
之后、噪声耦合矩阵之后（部署权重依赖这些耦合矩阵，所以必须放在它们后面）加入：

```python
        # -- deployed attention weights (offline fusion) ------------------
        # Value noise coupling must be [n, Hkv, rh] to read directly from the
        # mixed state; the legacy [d, Hkv, rh] form read from plaintext h.
        value_signal_coupling = torch.stack(
            [
                _random_matrix(
                    hidden_transform.total_dim,
                    value_noise,
                    seed=seed,
                    domain="block-%d-value-C-%d" % (self.layer_id, head),
                    scale=0.03,
                )
                for head in range(kv_heads)
            ],
            dim=1,
        )
        validate_auxiliary_budget(
            basis=hidden_transform,
            sample_signal=torch.randn(
                64, hidden, generator=make_generator(seed, "budget-probe")
            ),
            signal_noise_coupling=self.attention_noise_coupling,
            noise_propagator=self.attention_noise_propagator,
            context="block-%d-attention" % self.layer_id,
        )
        deployed_attention = build_deployed_attention(
            basis=hidden_transform,
            value_bases=value_transforms,
            gamma_attention=plain.attention_norm_weight.detach(),
            q_weight_math=plain.q_proj.weight.detach().T.contiguous(),
            k_weight_math=plain.k_proj.weight.detach().T.contiguous(),
            v_weight_math=plain.v_proj.weight.detach().T.contiguous(),
            o_weight_math=plain.o_proj.weight.detach().T.contiguous(),
            q_bias=(
                plain.q_proj.bias.detach()
                if plain.q_proj.bias is not None
                else None
            ),
            k_bias=(
                plain.k_proj.bias.detach()
                if plain.k_proj.bias is not None
                else None
            ),
            v_bias=(
                plain.v_proj.bias.detach()
                if plain.v_proj.bias is not None
                else None
            ),
            value_signal_coupling=value_signal_coupling,
            signal_noise_coupling=self.attention_noise_coupling,
            noise_propagator=self.attention_noise_propagator,
            auxiliary_to_hidden=self.attention_aux_to_hidden,
            fixed_refresh=(
                self.attention_fixed_refresh if fixed else None
            ),
            kv_index=plain.kv_index.detach(),
            head_dim=head_dim,
        )
        self.register_buffer(
            "deployed_q", deployed_attention.query, persistent=True
        )
        self.register_buffer(
            "deployed_k", deployed_attention.key, persistent=True
        )
        self.register_buffer(
            "deployed_v", deployed_attention.value, persistent=True
        )
        self.register_buffer(
            "deployed_q_bias", deployed_attention.query_bias, persistent=True
        )
        self.register_buffer(
            "deployed_k_bias", deployed_attention.key_bias, persistent=True
        )
        self.register_buffer(
            "deployed_v_bias", deployed_attention.value_bias, persistent=True
        )
        self.register_buffer(
            "deployed_attn_out", deployed_attention.output, persistent=True
        )
        self.register_buffer(
            "deployed_attn_noise_out",
            deployed_attention.noise_out,
            persistent=True,
        )
        self.register_buffer(
            "deployed_attn_refresh_out",
            deployed_attention.refresh_out,
            persistent=True,
        )
```

顶部 import 加：

```python
from ..layers.deployed import (
    build_deployed_attention,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
```

> **`persistent=True` 是对的**：部署权重是服务端必须持有的东西，不是密钥。
> 任务 A3 的测试 `test_state_dict_excludes_key_material` 检查的是
> `coupling`/`propagator`/`rotation` 等原始密钥材料，`deployed_*` 不在禁止列表里。
> **但注意**：`deployed_attn_noise_out` 的构造含 `M_bot`，`(c@N)` 里含 `N`——
> 见 B1.3，这是已记录的、不可规避的泄漏。

**B2.3** 噪声读出矩阵 `N` 必须在线可用。加一个 buffer：

```python
        self.register_buffer(
            "noise_read",
            hidden_transform.noise_projection().to(dtype=torch.float32),
            persistent=True,
        )
```

**B2.4** 保存基对象供 `rms_scale` 使用。`StructuredBasis` 不是 tensor，
不能 `register_buffer`。按现有 `_make_hidden_checkpoint` 的闭包模式处理：

```python
        # rho is a scalar per token; the basis factors stay in the closure and
        # out of state_dict, matching the existing checkpoint convention.
        self._rms_scale = _make_rms_scale(hidden_transform)
```

在模块级加辅助函数：

```python
def _make_rms_scale(basis: StructuredBasis):
    """Capture a basis for the O(n*b) RMS scale outside persistent state.

    Returns a callable ``(mixed, eps) -> [..., 1]`` in FP32. Only a scalar per
    token is produced; the signal is never reconstructed.
    """

    def rms_scale(mixed: torch.Tensor, eps: float) -> torch.Tensor:
        return basis.rms_scale(mixed, eps)

    return rms_scale
```

**B2.5** 重写 `_run()` 的 attention 段。找到从

```python
        hidden, side_noise = self._unmix_checkpoint(state)
```

开始、到

```python
        post_attention_state = self._mix_checkpoint(
            post_attention_signal, attention_noise
        )
```

结束的整段，替换为：

```python
        # No decode: the signal is never reconstructed on this path.
        mixed = state.mixed
        batch, sequence, _ = mixed.shape
        if positions is None:
            positions = torch.arange(sequence, device=mixed.device)
        elif positions.shape != (sequence,):
            raise ValueError("positions must have shape [sequence]")
        positions = positions.to(device=mixed.device, dtype=torch.long)
        if torch.any(positions < 0) or torch.any(
            positions >= self.config.max_sequence_length
        ):
            raise ValueError("position is outside configured context")
        valid_tokens = self._mask_from_mixed(mixed, token_mask)

        def cast(name: str) -> torch.Tensor:
            tensor = getattr(self, name)
            return tensor.to(device=mixed.device, dtype=mixed.dtype)

        # rho carries the RMSNorm statistic; FP32 reduction per AGENTS.md.
        scale = self._rms_scale(mixed, self.config.rms_epsilon).to(
            device=mixed.device, dtype=mixed.dtype
        )
        q_flat = (mixed @ cast("deployed_q")) / scale + cast("deployed_q_bias")
        k_flat = (mixed @ cast("deployed_k")) / scale + cast("deployed_k_bias")
        q = q_flat.view(
            batch, sequence, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        k = k_flat.view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        current_value_mixed = torch.einsum(
            "btn,hne->bhte", mixed, cast("deployed_v")
        ) / scale.unsqueeze(1) + cast("deployed_v_bias")[None, :, None, :]

        q_rope = apply_rope(q, positions, theta=self.config.rope_theta)
        k_rope = apply_rope(k, positions, theta=self.config.rope_theta)
        q_prime, k_prime = apply_qk_orthogonal_after_rope(
            q_rope, k_rope, self.common_qk, self.kv_index
        )
```

然后 **cache 处理逻辑完全保持原样**（从 `if cache is None:` 到
`new_cache = (...)`），因为 `k_prime` / `current_value_mixed` 的形状和语义都没变。

cache 段之后的 attention 调用也保持原样。**唯一要改的是 attention 之后**：
把原来的

```python
        context_augmented = self._fused_value_unmix(mixed_context, self.kv_index)
        context_signal = context_augmented[..., : self.config.head_dim]
        context_auxiliary = context_augmented[..., self.config.head_dim :]
        context_flat = context_signal.transpose(1, 2).reshape(
            batch, sequence, self.config.hidden_size
        )
        attention_output = context_flat @ self.o_weight_math.to(...)
        ...
        post_attention_signal = hidden + attention_output
        auxiliary_mean = context_auxiliary.mean(dim=1)
        attention_noise = (...)
        post_attention_state = self._mix_checkpoint(
            post_attention_signal, attention_noise
        )
```

整段替换为：

```python
        # The mixed context is consumed directly: no Value unmix, and the clean
        # attention context O is never materialized.
        post_attention_mixed = (
            mixed
            + torch.einsum(
                "bhqd,hdn->bqn", mixed_context, cast("deployed_attn_out")
            )
            + (mixed @ cast("noise_read")) @ cast("deployed_attn_noise_out")
            + self._refresh_out(
                cast("deployed_attn_refresh_out"),
                request_context,
                "attention",
                mixed,
            )
        )
        post_attention_state = MixedState(post_attention_mixed, self.hidden_basis)
```

**B2.6** 新增两个辅助方法。`_mask_from_mixed` 替代原 `_mask`
（原方法用明文 `hidden` 取形状，现在用 `mixed`）：

```python
    def _mask_from_mixed(
        self, mixed: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Return a boolean [batch, sequence] validity mask."""

        if token_mask is None:
            return torch.ones(
                mixed.shape[0],
                mixed.shape[1],
                dtype=torch.bool,
                device=mixed.device,
            )
        if (
            token_mask.shape != mixed.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("token_mask must be boolean [batch, sequence]")
        return token_mask.to(device=mixed.device)
```

`_refresh_out` 把 `r` 维刷新噪声映射到 `n` 维增量：

```python
    def _refresh_out(
        self,
        fixed_out: torch.Tensor,
        context: RequestContext,
        domain: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mixed-basis increment contributed by the noise refresh.

        ``fixed_debug`` uses the pre-absorbed ``diag(xi) @ M_bot`` rows summed
        to a single ``[n]`` vector. ``per_request`` samples ``xi`` from the
        request-scoped generator and maps it through the same rows.
        """

        if not self.noise_injection_enabled:
            return torch.zeros(
                fixed_out.shape[-1],
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.obfuscation.refresh_mode == "fixed_debug":
            return fixed_out.sum(dim=0)
        generator = context.generator_for(
            "block", self.layer_id, domain, "refresh"
        )
        sampled = torch.randn(
            self.obfuscation.hidden_noise_dim,
            generator=generator,
            dtype=torch.float32,
        ) * (0.02 * self.refresh_noise_scale)
        rows = self._noise_rows.to(
            device=reference.device, dtype=reference.dtype
        )
        return sampled.to(device=reference.device, dtype=reference.dtype) @ rows
```

这需要 `M_bot` 在线可用，在 `__init__` 中加：

```python
        self.register_buffer(
            "_noise_rows",
            hidden_transform.noise_rows().to(dtype=torch.float32),
            persistent=True,
        )
```

> **注意 `fixed_debug` 的 `refresh_out` 语义**：`build_deployed_attention`
> 返回的 `refresh_out = diag(ξ) @ M_bot`，形状 `[r, n]`。`ξ @ M_bot` 等于
> `diag(ξ) @ M_bot` 的行和，所以 `.sum(dim=0)` 是对的。
> **不要写成 `.sum(dim=1)`。**

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_obfuscated_block.py -v
python -m pytest -q tests/test_deployed_weights.py
python -m pytest -q
PYTHONPATH=src python scripts/record_exact_tolerance.py \
    --output results/raw/exact_tolerance_B2.json
```

### 验收标准
- [ ] `python -m pytest -q` 全绿
- [ ] `exact_tolerance_B2.json` 的 `exact_max_absolute_error` 与
      `exact_tolerance_A5.json` 同量级（≤1e-5 相对）
- [ ] `grep -n "_unmix_checkpoint" src/fastprove/models/obfuscated.py`
      在 attention 段不再出现
- [ ] `grep -n "o_weight_math\|q_weight_math" src/fastprove/models/obfuscated.py` 为空

### 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| 误差 ~1e-2 | `Wnz` 漏了 `−I` | 检查 `build_deployed_attention` 的 `(propagator - identity)` |
| 误差 ~1e-1，随 `‖e‖` 增大 | 噪声耦合形状用了 `[d, Hkv, rh]` 而非 `[n, Hkv, rh]` | 按 B2.2 改成 `total_dim` |
| 辅助项差一个 `H` 倍 | `Wattn` 里漏了 `(1/H)` | 见 B0 表格 |
| `ρ` 全是 `sqrt(eps)` | `rms_scale` 传的是 `state` 而非 `state.mixed` | 传 tensor |
| padding 行 NaN | 不应发生（`eps` 保证 `ρ>0`）；若发生说明 `clamp_min(0)` 被删了 | 恢复 `clamp_min` |

### 回滚
`git checkout -- src/fastprove/models/obfuscated.py`

---

## 任务 B3：重写 `ObfuscatedDecoderBlock`（FFN 段）

### 目标
同 B2，处理 FFN 段。

### 前置条件
任务 B2 完成且测试全绿。

### 步骤

**B3.1** 在 `__init__` 中，`swiglu_transform` 生成之后，把原来的
`convert_swiglu_weights(...)` 调用**替换**为部署权重构建：

```python
        validate_auxiliary_budget(
            basis=hidden_transform,
            sample_signal=torch.randn(
                64, hidden, generator=make_generator(seed, "budget-probe-ffn")
            ),
            signal_noise_coupling=self.down_noise_coupling,
            noise_propagator=self.down_noise_propagator,
            context="block-%d-ffn" % self.layer_id,
        )
        deployed_ffn = build_deployed_feed_forward(
            basis=hidden_transform,
            gamma_ffn=plain.ffn_norm_weight.detach(),
            gate_weight_math=plain.gate_proj.weight.detach().T.contiguous(),
            up_weight_math=plain.up_proj.weight.detach().T.contiguous(),
            down_weight_math=plain.down_proj.weight.detach().T.contiguous(),
            neuron_permutation=swiglu_transform.permutation,
            neuron_scale=swiglu_transform.scale,
            swiglu_noise_coupling=self.swiglu_noise_coupling,
            down_noise_coupling=self.down_noise_coupling,
            noise_propagator=self.down_noise_propagator,
            fixed_refresh=self.down_fixed_refresh if fixed else None,
        )
        self.register_buffer("deployed_gate", deployed_ffn.gate, persistent=True)
        self.register_buffer("deployed_up", deployed_ffn.up, persistent=True)
        self.register_buffer(
            "deployed_ffn_out", deployed_ffn.output, persistent=True
        )
        self.register_buffer(
            "deployed_ffn_noise_out", deployed_ffn.noise_out, persistent=True
        )
        self.register_buffer(
            "deployed_ffn_refresh_out", deployed_ffn.refresh_out, persistent=True
        )
```

> **注意**：`swiglu_noise_coupling` 原本是 `[dff, r]`，正确，不用改形状。
> `build_deployed_feed_forward` 把 `C_z M_bot` 直接融进 `Wffn`，所以
> `refresh_swiglu_noise` 这个函数在 block 里**不再被调用**。
> `src/fastprove/layers/swiglu.py` 保留（`generate_swiglu_transform` 仍在用），
> 但 `refresh_swiglu_noise` 变成只有测试用——见任务 B6。

**B3.2** 重写 `_run()` 的 FFN 段。把从

```python
        ffn_signal, ffn_side_noise = self._unmix_checkpoint(post_attention_state)
```

到

```python
        final_state = self._mix_checkpoint(final_signal, final_noise)
```

整段替换为：

```python
        # FFN segment, also without decoding.
        post_mixed = post_attention_state.mixed
        scale_ffn = self._rms_scale(post_mixed, self.config.rms_epsilon).to(
            device=post_mixed.device, dtype=post_mixed.dtype
        )
        gate_prime = (post_mixed @ cast("deployed_gate")) / scale_ffn
        up_prime = (post_mixed @ cast("deployed_up")) / scale_ffn
        z_prime = F.silu(gate_prime) * up_prime
        final_mixed = (
            post_mixed
            + z_prime @ cast("deployed_ffn_out")
            + (post_mixed @ cast("noise_read")) @ cast("deployed_ffn_noise_out")
            + self._refresh_out(
                cast("deployed_ffn_refresh_out"),
                request_context,
                "down",
                post_mixed,
            )
        )
        final_state = MixedState(final_mixed, self.hidden_basis)
```

**B3.3** `forward_debug` 的诊断段引用了 `hidden`、`ffn_signal`、
`attention_noise`、`swiglu_noise`、`final_noise` 等已不存在的明文量。
**debug 路径允许解码**（AGENTS.md 明确区分 debug/production），所以在
`return_debug` 分支内显式解码：

```python
        if not return_debug:
            return final_state, None, new_cache
        # Debug path only: explicit decode for diagnostics. Never reachable from
        # forward(); gated by debug_enabled in forward_debug().
        decoded_in = self._debug_unmix(state.mixed)
        decoded_post = self._debug_unmix(post_attention_mixed)
        decoded_final = self._debug_unmix(final_mixed)
        hidden = decoded_in[..., : self.config.hidden_size]
        post_attention_signal = decoded_post[..., : self.config.hidden_size]
        attention_noise = decoded_post[..., self.config.hidden_size :]
        final_noise = decoded_final[..., self.config.hidden_size :]
        attention_output = post_attention_signal - hidden
```

并加辅助方法：

```python
    def _debug_unmix(self, mixed: torch.Tensor) -> torch.Tensor:
        """Decode a mixed state. Debug/diagnostics only.

        Not called from the production forward path. Present so that
        ``forward_debug`` can report plaintext-referenced error metrics.
        """

        if not self.debug_enabled:
            raise PermissionError("decode is disabled outside debug mode")
        return self._debug_basis_unmix(mixed)
```

`_debug_basis_unmix` 用闭包捕获（同 `_rms_scale` 模式）：

```python
        self._debug_basis_unmix = (
            (lambda tensor: hidden_transform.unmix(tensor))
            if debug_enabled
            else None
        )
```

`swiglu_noise_state` 这个 debug 字段现在没有直接对应量（`C_z` 已融进 `Wffn`）。
把 `ObfuscatedBlockDebug` 的该字段改为 `z_prime`（更有诊断价值），并更新
所有引用它的测试。

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_obfuscated_block.py tests/test_swiglu.py -v
python -m pytest -q
PYTHONPATH=src python scripts/record_exact_tolerance.py \
    --output results/raw/exact_tolerance_B3.json
```

### 验收标准
- [ ] `python -m pytest -q` 全绿
- [ ] `exact_tolerance_B3.json` 误差 ≤1e-5 相对
- [ ] `grep -c "_unmix_checkpoint\|_mix_checkpoint" src/fastprove/models/obfuscated.py`
      为 0（两个 checkpoint 函数在 block 里已完全消失）
- [ ] `forward()` 路径上 `grep -n "_debug_unmix"` 不出现

### 回滚
`git checkout -- src/fastprove/models/obfuscated.py`

---

## 任务 B4：`ObfuscatedTinyCausalLM` 的入口与出口

### 目标
模型级的 embedding 混合和最终解码也要改。**注意：本任务只做 basis 适配，
词表置换和 SecureEmbedding/Head 是阶段 C 的内容。**

### 步骤

**B4.1** 入口。原实现：

```python
        embedding = F.embedding(input_ids, self.embedding_weight)
        initial_noise = (
            embedding.float() @ self.initial_noise_coupling.float()
            + self._initial_refresh(request_context)
        ).to(dtype=embedding.dtype)
        state = self._fused_checkpoint_mix(embedding, initial_noise)
```

这里 embedding 是明文 `h₀`，然后混合。**这一步无法避免解码，因为
embedding 表本身就是明文的**——阶段 C 的 `SecureEmbedding` 会把
`[E, E_n] M₀` 预先融进词表，从而彻底消除这一步。阶段 B 保留现状，
只把 `_fused_checkpoint_mix` 换成 `basis.mix`：

```python
        embedding = F.embedding(input_ids, self.embedding_weight)
        initial_noise = (
            embedding.float() @ self.initial_noise_coupling.float()
            + self._initial_refresh(request_context)
        ).to(dtype=embedding.dtype)
        # Stage C replaces this with a pre-mixed vocabulary so the plaintext
        # embedding is never materialized. See task C2.
        state = MixedState(
            self._basis_mix(torch.cat((embedding, initial_noise), dim=-1)),
            self.hidden_basis,
        )
```

**B4.2** 出口。原实现解码后做 final norm 和 LM head：

```python
        final_signal, _ = self._fused_checkpoint_unmix(state)
        normalized = rms_norm_fp32(final_signal, self.final_norm_weight, ...)
        logits = F.linear(normalized, self.lm_head_weight)
```

改为不解码：

```python
        # Final norm + head without decoding: rho from the blockwise Gram, and
        # P diag(gamma_final) folded into the head.
        scale = self._rms_scale(state.mixed, self.config.rms_epsilon).to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        head = self.deployed_head.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        logits = (state.mixed @ head) / scale
```

在 `__init__` 中构建 `deployed_head`：

```python
        # P @ diag(gamma_final) @ W_head^T, in math layout [n, V].
        projection = hidden_transform.signal_projection()
        deployed_head = (
            projection * plain.final_norm_weight.detach().to(torch.float64)[None, :]
        ) @ plain.lm_head.weight.detach().T.contiguous().to(torch.float64)
        self.register_buffer(
            "deployed_head",
            deployed_head.to(dtype=torch.float32),
            persistent=True,
        )
```

并删除 `lm_head_weight` 和 `final_norm_weight` 两个 buffer。

> **⚠️ 权重共享（tied embedding）的内存后果**：Llama-3.2 是
> `tie_word_embeddings=true`，`lm_head.weight` 就是 `embed_tokens.weight`。
> 部署后 `deployed_head` 是独立的 `[n, V]` 矩阵，不再与 embedding 共享。
> 对 3B 模型（V=128256, n=3088, BF16）约 **0.79 GB 额外内存**，
> 相对原 0.79 GB 的 embedding 表是 **+100% 的词表侧内存**，
> 整体峰值内存增幅会**超过手册 §69 的 5% 目标**。
>
> 这是必须记录的取舍，**不得隐瞒**。两个选项：
>
> | 选项 | 内存 | 是否物化明文 |
> |---|---|---|
> | 独立 `deployed_head`（默认） | +0.79 GB | ❌ 不物化 |
> | 融合 final-norm + head，中间态留寄存器 | +0 | ⚠️ 需要融合 kernel；eager 下会物化 `hn` |
>
> 阶段 B 用默认选项（正确性优先）。**任务 D2 必须把实测峰值内存增幅报出来，
> 并明确它超出 5% 目标。** 融合 kernel 属阶段 E（本文档不覆盖）。

**B4.3** 加 `_basis_mix` / `_rms_scale` 闭包（同 B2.4 模式）。

### 验证命令

```bash
cd /Users/yin/code/fastProve
python -m pytest -q tests/test_tiny_lm.py -v
python -m pytest -q
```

### 验收标准
- [ ] `tests/test_tiny_lm.py::test_tiny_exact_logits_and_greedy_tokens_match_plaintext`
      通过（`atol=2e-4`）
- [ ] `python -m pytest -q` 全绿
- [ ] `grep -n "lm_head_weight\|final_norm_weight" src/fastprove/models/obfuscated.py` 为空

---

## 任务 B5：与 `ChainLinear` 的等价性测试

### 目标
`ChainLinear` 是手册 §11/§42 的通用形式，已实现且测试完备，但模型不再直接
调用它（部署权重是它的融合特化）。**必须证明特化与通用形式等价**，否则
`ChainLinear` 就是死代码，而 `docs/` 里对它的引用就是虚假声明。

### 步骤

新建 `tests/test_deployed_matches_chain_linear.py`：

```python
"""The fused deployed weights must equal the general ChainLinear composition.

``ChainLinear`` implements the manual's general augmented affine map
``K = [[W, C], [0, G]]`` with ``W_tilde = M_in^-1 K M_out``. The deployed
weights of Stage B are a *fusion* of several such maps. This test pins the
equivalence so that ``ChainLinear`` remains the documented reference form
rather than dead code.
"""

from __future__ import annotations

import torch

from fastprove.layers.linear import ChainLinear
from fastprove.structured import generate_structured_basis
from fastprove.transforms import BasisTransform

_FP64 = torch.float64


def _dense_transform(basis) -> BasisTransform:
    """Wrap a StructuredBasis as the dense BasisTransform ChainLinear expects."""

    dense = basis.dense()
    return BasisTransform(
        matrix=dense,
        inverse=basis.dense_inverse(),
        signal_dim=basis.signal_dim,
        noise_dim=basis.noise_dim,
        condition_number=float(torch.linalg.cond(dense)),
        fingerprint=None,  # filled below
    )


def test_deployed_residual_map_equals_chain_linear() -> None:
    """A single deployed residual step equals a ChainLinear with W = I + dW."""

    torch.manual_seed(0)
    signal_dim, noise_dim = 32, 8
    basis = generate_structured_basis(
        signal_dim, noise_dim, seed=1, domain="eq", block_size=8, dtype=_FP64
    )
    top = basis.signal_rows()
    bottom = basis.noise_rows()
    noise_read = basis.noise_projection()

    weight = torch.randn(signal_dim, signal_dim, dtype=_FP64) * 0.1
    coupling = torch.randn(signal_dim, noise_dim, dtype=_FP64) * 0.02
    propagator = 0.5 * torch.eye(noise_dim, dtype=_FP64)

    signal = torch.randn(4, signal_dim, dtype=_FP64)
    noise = torch.randn(4, noise_dim, dtype=_FP64) * 0.1
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))

    # Deployed (fused) form, as used by Stage B.
    delta = signal @ weight
    identity = torch.eye(noise_dim, dtype=_FP64)
    fused = (
        mixed
        + delta @ (top + coupling @ bottom)
        + (mixed @ noise_read) @ ((propagator - identity) @ bottom)
    )

    # General ChainLinear form: signal map I + W, noise map C and G.
    expected_signal = signal + delta
    expected_noise = signal @ weight @ coupling + noise @ propagator
    reference = basis.mix(
        torch.cat((expected_signal, expected_noise), dim=-1)
    )

    assert torch.allclose(fused, reference, atol=1e-10)


def test_chain_linear_still_satisfies_its_documented_identity() -> None:
    """Guard the general primitive itself (manual section 11)."""

    torch.manual_seed(1)
    signal_dim, noise_dim = 16, 4
    basis_in = generate_structured_basis(
        signal_dim, noise_dim, seed=2, domain="in", block_size=4, dtype=_FP64
    )
    basis_out = generate_structured_basis(
        signal_dim, noise_dim, seed=3, domain="out", block_size=4, dtype=_FP64
    )
    weight = torch.randn(signal_dim, signal_dim, dtype=_FP64) * 0.1
    bias = torch.randn(signal_dim, dtype=_FP64) * 0.05
    coupling = torch.randn(signal_dim, noise_dim, dtype=_FP64) * 0.02
    propagator = 0.5 * torch.eye(noise_dim, dtype=_FP64)
    refresh = torch.randn(noise_dim, dtype=_FP64) * 0.02

    signal = torch.randn(3, signal_dim, dtype=_FP64)
    noise = torch.randn(3, noise_dim, dtype=_FP64)

    # W_tilde = M_in^-1 K M_out applied to c_in must equal mixing the
    # plaintext result with M_out.
    augmented_map = torch.zeros(
        signal_dim + noise_dim, signal_dim + noise_dim, dtype=_FP64
    )
    augmented_map[:signal_dim, :signal_dim] = weight
    augmented_map[:signal_dim, signal_dim:] = coupling
    augmented_map[signal_dim:, signal_dim:] = propagator
    deployed = (
        basis_in.dense_inverse() @ augmented_map @ basis_out.dense()
    )
    deployed_bias = torch.cat((bias, refresh)) @ basis_out.dense()

    mixed_in = basis_in.mix(torch.cat((signal, noise), dim=-1))
    observed = mixed_in @ deployed + deployed_bias
    expected = basis_out.mix(
        torch.cat(
            (
                signal @ weight + bias,
                signal @ coupling + noise @ propagator + refresh,
            ),
            dim=-1,
        )
    )
    assert torch.allclose(observed, expected, atol=1e-10)
```

> **注意**：第一个 helper `_dense_transform` 里 `fingerprint=None` 会让
> `BasisTransform.__post_init__` 抛错。**该 helper 实际上没被用到，删掉它。**
> 我留在这里是为了说明：不要试图把 `StructuredBasis` 包装成
> `BasisTransform` 来复用 `ChainLinear` 模块本身——直接验证代数恒等式更清晰。
> 执行时请删除 `_dense_transform` 函数和 `BasisTransform` 的 import。

### 验收标准
- [ ] 两个测试通过
- [ ] 删除了 `_dense_transform` helper
- [ ] `python -m pytest -q` 全绿

---

## 任务 B6：清理死代码

### 目标
阶段 B 之后有些函数不再被生产路径调用。**要么删除，要么明确标注为
「通用参考形式 / 仅测试」**，不允许留下无人知道状态的代码。

### 步骤

**B6.1** 逐项处理：

| 符号 | 处理 |
|---|---|
| `_make_hidden_checkpoint` | **删除**（被 `_basis_mix`/`_rms_scale`/`_debug_basis_unmix` 取代） |
| `_make_value_checkpoint` | **删除**（Value mix 已融进 `deployed_v`，unmix 已融进 `deployed_attn_out`） |
| `_checkpoint_compute_dtype` | **删除**（不再有 checkpoint 算术） |
| `layers/rmsnorm.py:absorbed_rms_projection` | 保留，`deployed.py` 未用但它是手册 §15.3 的参考实现 → 在 docstring 注明「reference form; deployed path fuses this into build_deployed_*」 |
| `layers/swiglu.py:refresh_swiglu_noise` | 保留 + 注明同上 |
| `layers/swiglu.py:convert_swiglu_weights` | 保留 + 注明同上 |
| `layers/linear.py:ChainLinear` | 保留，由 B5 的测试锚定 |
| `layers/rmsnorm.py:rms_norm_fp32` | 保留（plain 模型在用） |

**B6.2** 对每个「保留但生产路径不用」的符号，在 docstring 第一段后加一句：

```python
    """<原有摘要>

    Reference form. The deployed forward path fuses this transformation into
    ``layers/deployed.py``; this function remains the documented general case
    and is exercised by ``tests/test_deployed_matches_chain_linear.py``.
    """
```

**B6.3** 加一个测试防止未来重新引入解码：

新建 `tests/test_no_decode_in_forward.py`：

```python
"""Static guard: the production forward path must not decode or invert."""

from __future__ import annotations

import ast
import pathlib

_FORBIDDEN_CALLS = {"inv", "solve", "pinv", "lstsq"}
_FORBIDDEN_NAMES = {"unmix", "_debug_unmix", "signal_projection"}


def _forward_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "forward",
            "_run",
            "generate_greedy",
        ):
            yield node


def test_forward_path_contains_no_inverse_or_decode() -> None:
    source = pathlib.Path("src/fastprove/models/obfuscated.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    offences = []
    for function in _forward_functions(tree):
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_CALLS | _FORBIDDEN_NAMES:
                    offences.append("%s -> .%s" % (function.name, node.attr))
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                offences.append("%s -> %s" % (function.name, node.id))
    # ``_run`` legitimately references ``_debug_unmix`` inside its
    # ``return_debug`` branch, which ``forward`` can never reach.
    offences = [item for item in offences if "_debug_unmix" not in item]
    assert offences == [], "decode/inverse reached the forward path: %s" % offences


def test_deployed_module_never_inverts_at_runtime() -> None:
    source = pathlib.Path("src/fastprove/layers/deployed.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("build_"):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in (
                    "inv",
                    "solve",
                ):
                    raise AssertionError(
                        "%s calls a runtime inverse; conversion must use "
                        "dense_inverse() from the basis instead" % node.name
                    )
```

### 验收标准
- [ ] `python -m pytest -q` 全绿
- [ ] `tests/test_no_decode_in_forward.py` 两个测试通过
- [ ] `grep -n "_make_hidden_checkpoint\|_make_value_checkpoint\|_checkpoint_compute_dtype" src/` 为空
- [ ] 每个保留的未用符号都有 "Reference form" 说明

---

## 任务 B7：开销实测

### 目标
拿到阶段 B 的真实 overhead 数字，写进原始记录。

### 步骤

**B7.1** 新建 `scripts/measure_overhead.py`：

```python
"""Measure reference-implementation overhead versus the plaintext model.

Usage:
    python scripts/measure_overhead.py --output results/raw/overhead_B7.json

Records prefill and decode timings for the plaintext and obfuscated reference
modules on identical inputs, seeds, dtype and device. This measures the *eager
reference* implementation. It is not a fused-kernel result and must never be
reported as one.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def _time(function, repeats: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
    return min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    hidden = arguments.hidden_size
    config = ModelConfig(
        vocab_size=2048,
        hidden_size=hidden,
        intermediate_size=int(hidden * 2.6875),
        num_layers=arguments.layers,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_sequence_length=arguments.sequence_length + arguments.decode_tokens + 8,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=16,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=16,
    )
    plain = PlainTinyCausalLM(
        config, seed=arguments.seed, debug_enabled=False
    ).eval()
    started = time.perf_counter()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=arguments.seed,
        debug_enabled=False,
    )
    conversion_seconds = time.perf_counter() - started
    obfuscated = converted.module.eval()
    context = RequestContext(global_seed=arguments.seed, request_id="overhead")

    torch.manual_seed(arguments.seed)
    input_ids = torch.randint(
        0, config.vocab_size, (1, arguments.sequence_length)
    )

    prefill_plain = _time(
        lambda: plain(input_ids), arguments.repeats, arguments.warmup
    )
    prefill_obfuscated = _time(
        lambda: obfuscated(input_ids, request_context=context),
        arguments.repeats,
        arguments.warmup,
    )

    def decode_plain() -> None:
        logits, cache = plain(input_ids, use_cache=True)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(arguments.decode_tokens):
            position = torch.tensor([arguments.sequence_length + step])
            logits, cache = plain(
                token, positions=position, cache=cache, use_cache=True
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)

    def decode_obfuscated() -> None:
        logits, cache = obfuscated(
            input_ids, request_context=context, use_cache=True
        )
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(arguments.decode_tokens):
            position = torch.tensor([arguments.sequence_length + step])
            logits, cache = obfuscated(
                token,
                positions=position,
                request_context=context,
                cache=cache,
                use_cache=True,
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)

    decode_plain_seconds = _time(decode_plain, arguments.repeats, arguments.warmup)
    decode_obfuscated_seconds = _time(
        decode_obfuscated, arguments.repeats, arguments.warmup
    )

    tokens = arguments.decode_tokens
    record = {
        "implementation": "eager reference, not a fused kernel",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": arguments.threads,
        "device": "cpu",
        "activation_dtype": "float32",
        "model": {
            "hidden_size": hidden,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "sequence_length": arguments.sequence_length,
            "decode_tokens": tokens,
        },
        "obfuscation": {
            "hidden_noise_dim": obfuscation.hidden_noise_dim,
            "value_noise_dim_per_head": obfuscation.value_noise_dim_per_head,
            "basis_block_size": obfuscation.basis_block_size,
        },
        "conversion_seconds": conversion_seconds,
        "prefill_plaintext_seconds": prefill_plain,
        "prefill_obfuscated_seconds": prefill_obfuscated,
        "prefill_overhead_fraction": prefill_obfuscated / prefill_plain - 1.0,
        "decode_plaintext_seconds": decode_plain_seconds,
        "decode_obfuscated_seconds": decode_obfuscated_seconds,
        "decode_overhead_fraction": (
            decode_obfuscated_seconds / decode_plain_seconds - 1.0
        ),
        "decode_plaintext_tpot_seconds": decode_plain_seconds / tokens,
        "decode_obfuscated_tpot_seconds": decode_obfuscated_seconds / tokens,
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
```

**B7.2** 跑三个规模：

```bash
cd /Users/yin/code/fastProve
for size in 512 1024 2048; do
  PYTHONPATH=src python scripts/measure_overhead.py \
    --hidden-size $size \
    --output results/raw/overhead_B7_d$size.json
done
```

### 验收标准
- [ ] 三个 JSON 生成成功
- [ ] `prefill_overhead_fraction` 相比阶段 A 结束时**明显下降**
- [ ] 记录实际数字，**不要与手册 §69 的 ≤4%/≤5% 目标混为一谈**——
      那是融合 kernel 的目标，本任务测的是 eager 参考实现
- [ ] 若 eager overhead 仍 >30%，用 `python -X importtime` 或
      `torch.profiler` 定位，把 top-5 热点写进 `docs/implementation_notes.md`

### ⚠️ 报告纪律

`results/` 里任何性能数字必须带上：

```text
implementation: eager reference, not a fused kernel
```

手册 §70 明确要求区分「Reference implementation」与「Optimized fused
implementation」。**不得用参考实现的数字声称达到或未达到 5% 目标。**

---

## 阶段 B 出口检查

```bash
cd /Users/yin/code/fastProve
python docs/design_identity_check.py
python -m pytest -q
grep -rn "_unmix_checkpoint\|_mix_checkpoint" src/fastprove/models/    # 应为空
grep -rn "linalg.inv\|linalg.solve" src/fastprove/models/              # 应为空
git log --oneline -8
```

- [ ] B1–B7 各一个 commit
- [ ] `results/raw/exact_tolerance_B{2,3}.json` 存在，误差 ≤1e-5 相对
- [ ] `results/raw/overhead_B7_d{512,1024,2048}.json` 存在
- [ ] `tests/test_no_decode_in_forward.py` 通过
- [ ] 全局不变量 I1、I2 成立
- [ ] `docs/implementation_notes.md` 记录了：`ρ` 的 FP32 瓶颈、
      `‖e‖/‖h‖` 上界、tied-embedding 的内存后果、`N` 进部署权重的泄漏

**阶段 B 未全部通过，不要开始阶段 C。**

---

# 阶段 C：补齐缺失组件

**目的**：把手册要求但当前完全没有的部分做出来——词表置换、SecureEmbedding、
SecureLMHead、客户端/服务端落盘拆分、MoE 专家。

阶段 C 之前，输入 token ID 和输出 logits 都是**完全明文**的，手册 §33 的在线
协议（客户端只做 encode/decode）不存在。

---

## 任务 C1：`TokenCodec` 与词表置换

### 目标
客户端持有词表置换 `τ` 和逆置换 `τ⁻¹`；服务端**只**持有按 `τ` 重排后的
embedding 和 LM head，不持有 `τ⁻¹`。

### 涉及文件
- 新建 `src/fastprove/codec.py`
- 新建 `tests/test_token_codec.py`

### 数学
客户端发送 `ĩ = τ(i)`。Embedding 表按同一置换重排行，所以 `ĩ` 查到的仍是
原 token `i` 的词向量。LM head 输出 `ℓ̃ = ℓ P_voc`，因为 softmax 只发生词表位置
置换：`softmax(ℓ P_voc) = softmax(ℓ) P_voc`。客户端用 `τ⁻¹` 还原。

### 步骤

**C1.1** 新建 `src/fastprove/codec.py`：

```python
"""Client-side vocabulary permutation.

The client holds ``tau`` and ``tau^-1``. The server holds only the row-permuted
embedding table and the column-permuted head, never ``tau^-1``.

Row-vector convention. ``encode`` maps plaintext token ids to obfuscated ids;
``decode`` inverts it. Both are index gathers, not matrix products.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .seed import make_generator


@dataclass(frozen=True)
class TokenCodec:
    """Vocabulary permutation held by the client only."""

    permutation: torch.Tensor           # [V] int64, tau
    inverse_permutation: torch.Tensor   # [V] int64, tau^-1
    vocab_size: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.permutation.shape != (self.vocab_size,):
            raise ValueError("permutation shape must be [vocab_size]")
        if self.inverse_permutation.shape != (self.vocab_size,):
            raise ValueError("inverse shape must be [vocab_size]")
        for tensor in (self.permutation, self.inverse_permutation):
            if tensor.dtype != torch.int64:
                raise ValueError("token permutations must be int64")
            if not torch.equal(
                torch.sort(tensor.cpu()).values, torch.arange(self.vocab_size)
            ):
                raise ValueError("permutation must be a bijection on the vocab")
        if not torch.equal(
            self.permutation[self.inverse_permutation],
            torch.arange(self.vocab_size),
        ):
            raise ValueError("inverse_permutation does not invert permutation")

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map plaintext token ids to obfuscated ids. Client side only."""

        self._validate_ids(token_ids)
        return self.permutation.to(token_ids.device)[token_ids]

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map obfuscated token ids back. Client side only."""

        self._validate_ids(token_ids)
        return self.inverse_permutation.to(token_ids.device)[token_ids]

    def _validate_ids(self, token_ids: torch.Tensor) -> None:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token ids must be integral")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("token id outside vocabulary")

    def save_client_secret(self, path: str | Path) -> None:
        """Persist client material. Must never be written to a server dir."""

        torch.save(
            {
                "permutation": self.permutation,
                "inverse_permutation": self.inverse_permutation,
                "vocab_size": self.vocab_size,
                "fingerprint": self.fingerprint,
            },
            Path(path),
        )


def generate_token_codec(vocab_size: int, *, seed: int, domain: str) -> TokenCodec:
    """Generate a deterministic vocabulary permutation."""

    if vocab_size <= 1:
        raise ValueError("vocab_size must exceed one")
    generator = make_generator(seed, domain, "vocab-permutation", vocab_size)
    permutation = torch.randperm(vocab_size, generator=generator)
    inverse = torch.argsort(permutation)
    digest = hashlib.sha256()
    digest.update(("%d:" % vocab_size).encode("ascii"))
    digest.update(permutation.numpy().tobytes())
    return TokenCodec(
        permutation=permutation,
        inverse_permutation=inverse,
        vocab_size=vocab_size,
        fingerprint=digest.hexdigest(),
    )
```

**C1.2** 测试要覆盖：

| 测试 | 断言 |
|---|---|
| round trip | `decode(encode(ids)) == ids` 逐元素相等 |
| 确定性 | 同 seed 同 domain 产生相同 `fingerprint` |
| 域分离 | 不同 domain 产生不同置换 |
| 非双射拒绝 | 构造 `permutation=[0,0,1]` 抛 `ValueError` |
| 逆不匹配拒绝 | 传错的 `inverse_permutation` 抛 `ValueError` |
| 越界 token | `encode(tensor([V]))` 抛 `ValueError` |
| 非整型 | `encode(tensor([0.0]))` 抛 `ValueError` |
| 落盘不含服务端可用材料 | `save_client_secret` 后加载，确认含 `inverse_permutation` |

### 验收标准
- [ ] 上表 8 项全部有对应测试且通过
- [ ] `grep -rn "inverse_permutation" src/fastprove/models/` 为空
      （模型侧不得引用逆置换）

---

## 任务 C2：`SecureEmbedding`（预混合词表）

### 目标
消除任务 B4.1 里遗留的明文 embedding。把 `[E, E_n] M₀` 预先融进词表，
服务端 `F.embedding` 一次查表**直接得到混合态**。

### 数学
手册 §14.3：

```text
Ẽ = Π_voc^T [E, E_n] M₀
```

其中 `Π_voc` 是词表置换（行重排），`E_n` 是噪声 embedding 表 `[V, r]`。
服务端输入混淆 token ID，输出 `c₀`，客户端在线不参与。

`E_n` 的构造有两种选择：

| 选择 | 形式 | 性质 |
|---|---|---|
| 独立随机表（推荐） | `E_n ~ N(0, σ²)`，`[V, r]` | 噪声与信号统计独立 |
| 线性耦合 `E_n = E C` | `[V, r]` | 等价于当前实现，但 `e ≈ hC` 泄漏更强 |

**选独立随机表**，因为它切断了 `docs/threat_model.md` 5bis.6 记录的
`e ≈ hC` 通道在第 0 层的入口。注意这只影响第 0 层；后续层的 `e' = hC + eG + ξ`
仍然含 `hC` 项。

### 步骤

**C2.1** 新建 `src/fastprove/layers/embedding.py`，提供
`build_secure_embedding(...) -> torch.Tensor`（返回 `[V, n]` 的 `Ẽ`）和
`SecureEmbedding(nn.Module)`。

`forward` 必须只有一行 `F.embedding`：

```python
    def forward(self, input_ids: torch.Tensor) -> MixedState:
        """Return the mixed state for already-permuted token ids.

        ``input_ids`` must already be vocabulary-permuted by the client codec.
        No mixing happens here; ``M_0`` is absorbed into the table offline.
        """

        self._validate(input_ids)
        return MixedState(
            F.embedding(input_ids, self.table), self.basis_descriptor
        )
```

**C2.2** 构建器签名与校验：

```python
def build_secure_embedding(
    *,
    embedding_math: torch.Tensor,       # [V, d]
    noise_embedding: torch.Tensor,      # [V, r]
    basis: StructuredBasis,
    token_codec: TokenCodec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``Pi_voc^T [E, E_n] M_0`` in shape ``[V, n]``."""
```

必须校验：`embedding_math.shape == (V, d)`、`noise_embedding.shape == (V, r)`、
`token_codec.vocab_size == V`、`basis.signal_dim == d`、`basis.noise_dim == r`。

**行置换方向极易搞错。** 正确的是：

```python
    mixed = basis.mix(torch.cat((embedding_math, noise_embedding), dim=-1))
    table = torch.empty_like(mixed)
    table[token_codec.permutation] = mixed
```

即「混淆 ID 位置存放原 ID 的行」。等价写法 `table = mixed[inverse_permutation]`。
**必须有测试断言两种写法一致**，否则方向错了不会被发现（两者都是合法置换）。

**C2.3** 在 `ObfuscatedTinyCausalLM` 中替换 B4.1 的三行为：

```python
        state = self.embedding(input_ids)
```

删除 `embedding_weight`、`initial_noise_coupling`、`initial_fixed_refresh`
三个 buffer 和 `_initial_refresh` 方法。

> **per_request 刷新怎么办**：预混合词表是静态的，无法带 per-request `ξ₀`。
> 若要保留第 0 层的 per-request 刷新，在 embedding 之后加一次
> `c₀ += ξ₀ @ M_bot`（`r × n`，成本 49K MAC，可忽略）。**推荐保留**，
> 否则第 0 层退化为 `fixed`，而 `refresh_mode: per_request` 的配置就名不符实。

### 验收标准
- [ ] `test_permutation_direction_is_consistent`：两种写法结果相同
- [ ] `test_secure_embedding_decodes_to_plaintext_embedding`：
      `decode(SecureEmbedding(encode(ids)))` 的信号部分 == `E[ids]`
- [ ] `test_forward_contains_a_single_embedding_lookup`：用 AST 检查
      `forward` 里只有一次 `F.embedding` 且无矩阵乘
- [ ] `grep -n "embedding_weight" src/fastprove/models/obfuscated.py` 为空

---

## 任务 C3：`SecureLMHead`（仅词表置换）

### 目标
任务 B4.2 的 `deployed_head` 还缺词表置换。补上，并处理 tied-embedding 的
内存取舍。

### 数学
手册 §32：`ℓ̃ = ℓ Π_voc`，`softmax(ℓ̃) = softmax(ℓ) Π_voc`。
合并 B4.2 的 final-norm 吸收：

```text
W̃_head = P diag(γ_final) W_head Π_voc        形状 [n, V]
logits  = (c @ W̃_head) / ρ_final
```

### 步骤

**C3.1** 新建 `build_secure_lm_head(...)`，列置换方向：

```python
    head = (projection * gamma_final[None, :]) @ head_math   # [n, V]
    permuted = torch.empty_like(head)
    permuted[:, token_codec.permutation] = head
```

**列**置换用 `permuted[:, perm] = head`；**行**置换（embedding）用
`table[perm] = mixed`。两者不同，**必须各有一个方向一致性测试**。

**C3.2** ⚠️ 内存取舍必须显式配置，不得默认埋掉。

在 `ObfuscationConfig` 加字段：

```python
    lm_head_mode: str = "untied_deployed"
```

`__post_init__` 校验 `in ("untied_deployed", "fused_norm_head")`。

| 模式 | 内存（3B/BF16） | 是否物化明文 | 状态 |
|---|---|---|---|
| `untied_deployed`（默认） | +0.79 GB（词表侧 +100%） | ❌ 不物化 | 阶段 C 实现 |
| `fused_norm_head` | +0 | ⚠️ eager 下物化归一化态 | **抛 `NotImplementedError`** |

`fused_norm_head` 必须抛异常而不是静默回退到 eager：

```python
        if mode == "fused_norm_head":
            raise NotImplementedError(
                "fused_norm_head requires a fused kernel that keeps the "
                "normalized state in registers; the eager path would "
                "materialize plaintext. Use untied_deployed and record the "
                "memory cost, or implement the kernel first."
            )
```

这符合 AGENTS.md「Never silently fall back from a requested secure/fused path
to an eager path」。

**C3.3** 端到端接线。`ObfuscatedTinyCausalLM.forward` 的输入输出都变成混淆域：

```text
客户端: text -> tokenize -> codec.encode -> 混淆 ids
服务端: 混淆 ids -> forward -> 混淆 logits
客户端: argmax -> codec.decode -> 原 token -> detokenize
```

**服务端 argmax 的位置在混淆词表里，客户端 `decode` 后才是原 token。**
`generate_greedy` 全程在混淆域做 argmax 并把混淆 token 喂回去——这是对的，
因为 embedding 表也是混淆域的。**只在最外层返回给客户端时 decode 一次。**

### 验收标准
- [ ] `test_head_column_permutation_direction_is_consistent`
- [ ] `test_softmax_commutes_with_vocabulary_permutation`：
      `softmax(ℓ̃)[τ(i)] == softmax(ℓ)[i]`
- [ ] `test_greedy_token_sequence_matches_plaintext_after_decode`：
      100 条 prompt，逆置换后 token 序列**完全一致**（手册 §67 要求 100%）
- [ ] `test_fused_norm_head_raises_not_implemented`
- [ ] `test_server_state_dict_has_no_inverse_permutation`

---

## 任务 C4：`ModelConverter`（客户端/服务端落盘拆分）

### 目标
把「转换」从进程内变成落盘产物：一个服务端目录 + 一个客户端密钥目录。
只有这样才能验证「服务端不持有逆词表/解码矩阵」。

### 步骤

**C4.1** 新建 `src/fastprove/converter.py`，接口按手册 §55：

```python
class ModelConverter:
    def convert(
        self,
        plain_model: nn.Module,
        config: PrototypeConfig,
        client_secret_dir: str | Path,
        server_model_dir: str | Path,
    ) -> ConversionManifest:
        """Write a server model and a separate client secret bundle."""
```

**C4.2** 两个目录的内容必须严格划分：

```text
server_model_dir/
├── model.safetensors        部署权重（deployed_*, 预混合词表, 置换后 head）
├── runtime_config.json      维度、模式、basis fingerprint、条件数
└── manifest.json            架构、hash、转换时间、版本

client_secret_dir/
├── token_codec.pt           tau, tau^-1
├── bases.pt                 hidden basis + value bases 的全部因子
└── manifest.json            指纹，用于与服务端交叉校验
```

**C4.3** 转换器必须做的检查（手册 §55）：

| 检查 | 失败行为 |
|---|---|
| 架构是否支持 | `NotImplementedError` |
| `hidden_size % num_attention_heads == 0` | `ValueError` |
| `head_dim % 2 == 0`（RoPE） | `ValueError` |
| `n % basis_block_size == 0` | `ValueError` |
| tied embedding 检测 | 记录到 manifest，按 `lm_head_mode` 处理 |
| 所有 basis 条件数 ≤ 上界 | `ValueError` |
| `validate_auxiliary_budget` 每层通过 | `ValueError` |
| **服务端 checkpoint 不含密钥** | `RuntimeError` |

最后一项是最重要的，实现为显式扫描：

```python
_FORBIDDEN_SERVER_KEYS = (
    "perm_in", "perm_out", "scales", "blocks", "gram",
    "inverse", "rotation", "common_qk", "coupling", "propagator",
    "tau", "token_codec",
)

def _assert_server_bundle_is_clean(state: dict) -> None:
    """Fail conversion if key material reached the server bundle."""

    offending = [
        key
        for key in state
        for token in _FORBIDDEN_SERVER_KEYS
        if token in key.lower()
    ]
    if offending:
        raise RuntimeError(
            "server checkpoint contains key material: %s" % offending
        )
```

> **注意**：`common_qk` 是密钥但服务端**必须**用它（RoPE 之后应用，无法吸收）。
> 这是一个真实矛盾。处理方式：把 `common_qk` 归为「服务端必须持有的密钥」
> 并在 `runtime_config.json` 里显式标注，同时在威胁模型记录
> 「Q/K 正交基对服务端可见，因此 QK 几何结构不受保护」。
> **不要为了让检查通过而把它偷偷改名。** 从 `_FORBIDDEN_SERVER_KEYS`
> 移除 `common_qk`，并在该常量上方写注释说明为什么。

**C4.4** 加载侧：`load_server_model(server_model_dir)` 只用服务端目录就能跑
forward；`load_client(client_secret_dir)` 提供 encode/decode。

### 验收标准
- [ ] `test_server_bundle_contains_no_key_material`
- [ ] `test_server_model_runs_without_the_client_bundle`：删掉
      `client_secret_dir` 后仍能 forward
- [ ] `test_client_bundle_can_decode_server_output`
- [ ] `test_conversion_rejects_a_mismatched_architecture`
- [ ] `test_manifest_fingerprints_cross_validate`
- [ ] `runtime_config.json` 显式列出 `common_qk` 为服务端可见密钥

---

## 任务 C5：MoE 专家路径

### 目标
`StableRouter` 已实现并测试完备，但没有任何模型使用它。补上
expert / dispatcher / combine（手册 §30、§52）。

### 前置条件
C1–C4 完成。**Dense 模型必须先完全正确**（手册 §60）。

### 步骤

**C5.1** 专家复用 Dense FFN 的部署形式。每个专家独立域分离密钥：

```text
LAYER_%d_EXPERT_%d
```

**C5.2** 物理专家必须与 `r' = r P_E` 一致重排。`layers/router.py` 已提供
`reorder_experts(experts, permutation)`，直接用。

**C5.3** Router 输入是混合态，logits 要不解码地算出：

```text
W̃_router = P diag(γ_ffn) W_r P_E        形状 [n, E]
r' = (c @ W̃_router) / ρ
```

即专家置换 `P_E` 也吸收进部署权重。

**C5.4** 门控权重必须用**干净** logits（手册 §29.5）。`StableRouter` 的
`RouterConfig.noisy_gate_weights` 默认 `False`，保持默认。

**C5.5** dispatch/combine 的所有专家输出必须回到同一 residual basis，
用 `MixedState.add` 做校验（它已检查 fingerprint 一致）。

### 验收标准
- [ ] `test_expert_set_matches_plaintext_exactly`：手册 §66 要求 100%
- [ ] `test_tie_breaking_matches_plaintext`：相同 logits 的场景
- [ ] `test_margin_bounded_noise_preserves_expert_set`：`2τ < Δ_k`
- [ ] `test_gate_weights_use_clean_logits`
- [ ] `test_physical_experts_are_reordered_consistently`
- [ ] `test_all_expert_outputs_share_the_residual_basis`
- [ ] 覆盖 Top-1 / Top-2 / Top-4，大 margin / 小 margin / 相同 logits

---

## 阶段 C 出口检查

- [ ] C1–C5 各一个 commit
- [ ] `python -m pytest -q` 全绿
- [ ] 服务端目录可独立运行，且不含逆词表和基因子
- [ ] 100 条 prompt 的 greedy token 序列逆置换后 100% 一致
- [ ] `docs/threat_model.md` 已记录 `common_qk` 服务端可见

---

# 阶段 D：实验与报告

**目的**：产出可引用的结果。当前 `results/REPORT.md` 自己标了
`⚠️ SUPERSEDED`，且 gate 表 `HARD: FAIL`。

---

## 任务 D1：修复 exact 模式硬门禁

### 目标
`results/REPORT.md` 现有两条 HARD FAIL：

```text
LM Head inverse-permuted argmax    | hard | FAIL | 0.962963 | 1
Greedy release-validation sequences| hard | FAIL | 0.75     | 1
```

exact 模式**必须** 100%。这两条不修，所有近似噪声结论都不成立。

### 排查顺序（手册 §67，严格按序）

| 步 | 检查 | 具体做法 |
|---|---|---|
| 1 | 词表置换方向 | 任务 C2/C3 的方向一致性测试；`decode(encode(ids)) == ids` |
| 2 | Embedding 行序 | `SecureEmbedding(encode(i))` 解码后 == `E[i]` |
| 3 | 权重转置 | 每个 `build_deployed_*` 的输入是否真是数学布局；`_require_math_layout` 已拦截明显错误 |
| 4 | RMSNorm gamma | 是否重复应用（`γ` 既在 `deployed_gate` 又在别处） |
| 5 | RoPE | `positions` 在 cache decode 时是否是绝对位置 |
| 6 | Head 置换 | GQA 的 `kv_index` 映射；`common_qk` 是否按 KV head 而非 Q head 索引 |
| 7 | Down Projection | `D_f⁻¹` 的置换顺序：`(1/scale[perm])[:, None] * down[perm]` |
| 8 | LM Head | 列置换方向 |
| 9 | 采样种子 | plaintext 与 obfuscated 是否用同一 `RequestContext` |

### 关键：BF16 与 exact 门禁的关系

现有报告的 `e2e_logit_max_absolute_error = 1.32` 是 **BF16** 下的。
BF16 的 logit 舍入本身就有这个量级，**不能用它判断 exact 是否成立**。

**门禁必须在 FP32 下判定**：

```bash
# exact 门禁：FP32
PYTHONPATH=src python scripts/run_pretrained_compare.py \
    --activation-dtype float32 --check-gates ...

# BF16：单独记录为一个 condition，不作为 exact 门禁
PYTHONPATH=src python scripts/run_pretrained_compare.py \
    --activation-dtype bfloat16 ...
```

### 验收标准
- [ ] FP32 下 `LM Head inverse-permuted argmax == 1.0`
- [ ] FP32 下 `Greedy release-validation sequences == 1.0`
- [ ] BF16 结果单独记录，标注为独立 condition 而非 exact 门禁
- [ ] 若某条仍不为 1.0：**记录具体失败样本的 token 位置和 logit 差值**，
      不要放宽门禁。按上表定位到具体步骤

---

## 任务 D2：性能指标采集

### 目标
补齐手册 §68 要求但当前 raw 记录里**完全没有**的指标。

### 必须记录（分别记录 plaintext / obfuscated）

```text
TTFT                    prefill 首 token 延迟
TPOT                    decode 每 token 延迟
tokens/s
prefill latency
decode latency
peak memory             进程 RSS + 分配器峰值
KV cache memory         字节数，含 dh -> dh+rh 的增量
conversion time
kernel count            用 torch.profiler 计数
```

### 强制标注

每条记录必须含：

```json
{
  "implementation": "eager reference, not a fused kernel",
  "device": "...", "activation_dtype": "...",
  "torch_version": "...", "gpu_name": "..."
}
```

### 必须报出的负面结果

| 项 | 预期 | 处理 |
|---|---|---|
| eager overhead vs 手册 §69 的 ≤5% | 大概率**不达标** | 照实报，注明是参考实现 |
| tied-embedding 峰值内存 | **超出 5%**（见 B4.2） | 照实报，给出 `untied_deployed` 的字节数 |
| KV cache 增幅 | `(dh+rh)/dh`，`rh=2,dh=128` 时 +0.78% | 达标，报实测 |

### 验收标准
- [ ] 每项指标都有原始 JSON
- [ ] 每条记录带 `implementation` 标注
- [ ] 峰值内存超标这一条**明确写出来**，不得省略

---

## 任务 D3：噪声 sweep 重跑

### 目标
按 AGENTS.md 的配置跑完整 sweep：

```text
tau_max:        0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.10
alpha:          0.50, 0.80, 0.95
preserve_top_k: 4, 8, 16
```

### 前置条件
**任务 D1 必须通过。** exact 门禁失败时，sweep 的所有点必须记录为
`skipped`，不得记录为结果——`scripts/run_accuracy_sweep.py` 已实现这个
gating，不要绕过。

### 数据完整性（AGENTS.md 硬要求）

| 要求 | 做法 |
|---|---|
| plaintext 与 obfuscated 用完全相同样本 | `evaluation/token_cache.py` 缓存选定的 example id |
| 每个配置点有记录或显式失败记录 | 检查 expected-run manifest 对账 |
| 不静默省略任何点 | sweep 结束后核对点数 = 7×3×3 |
| 固定 seed 可复现 | 跑两次比对指标逐位相同 |

### 已知的 harness 缺陷（`results/REPORT.md` 自述）

重跑前必须确认这四条已修（`scripts/run_osnip_style_benchmarks.py` 声称已修）：

1. MMLU 头部截断（前 200 行只覆盖 2/57 学科）→ 需 subject-stratified 抽样
2. PIQA 全部 1838 行 `label == 0` → 需 degenerate-label 拒绝
3. 多选未做长度归一化 → 需 `acc_norm`
4. 明文基线 0.493 vs OSNIP 论文 0.597（差 10.4pp）→ **基线必须先与公开参考对上**

**第 4 条是前置条件**：基线没校准，retention 数字没有信息量。

### 验收标准
- [ ] 63 个配置点全部有记录或显式失败记录
- [ ] 明文基线与公开参考的差距 ≤2pp，或**明确记录无法对上的原因**
- [ ] 固定 seed 两次运行指标逐位相同
- [ ] Softmax 专属指标齐全：KL、JS、Top-k 重叠、rank correlation、
      margin 分布、实际噪声 ∞-范数、零噪声查询比例

---

## 任务 D4：威胁模型与报告更新

### 目标
把阶段 A–C 新产生的安全后果写进威胁模型，重写 `results/REPORT.md`。

### 必须新增到 `docs/threat_model.md` 的条目

| 编号 | 内容 | 来源 |
|---|---|---|
| 1 | **块对角基降低已知明文攻击代价**：每输出坐标只依赖 `b` 个输入坐标，恢复 `M` 从 `O(n)` 样本/块降到 `O(b)`。`r ≪ d` 时大部分块不含噪声坐标 | 任务 A2 |
| 2 | **`N` 进入部署权重**：`(c @ N) @ Wnz` 使观察部署权重者可恢复 `e`（up to 可逆 `r×r`）。已验证不可规避（`Z` 秩恒为 `r`） | 任务 B1.3 |
| 3 | **`common_qk` 服务端可见**：RoPE 之后应用，无法吸收。QK 几何结构不受保护 | 任务 C4.3 |
| 4 | **`ρ` 的 FP32 精度上界**：`‖e‖/‖h‖ ≤ 30`。想靠加大噪声提升混淆强度会污染信号路径 | 任务 A2.4 |
| 5 | **tied-embedding 的内存代价**：`untied_deployed` 使词表侧内存 +100%，峰值内存超出手册 §69 的 5% 目标 | 任务 B4.2 |

### `results/REPORT.md` 必须区分的四类结论

手册与 AGENTS.md 都要求分开陈述，**不得混为一谈**：

```text
1. 数学精确性       exact 模式的恒等式是否成立（FP32 门禁）
2. 浮点偏差         ρ 的 FP32 瓶颈 1.9e-7；BF16 的独立偏差
3. 任务精度退化     近似模式相对 plaintext 的 Δ_abs / drop_abs
4. 性能开销         eager 参考 vs 融合 kernel（必须分开）
5. 安全限制         上表 5 条 + 原有 5bis 的三条泄漏通道
```

### 不得声明的内容（AGENTS.md）

```text
✗ activations 是标准 LWE 密文
✗ 全同态 / 端到端加密推理
✗ 因为用了 ML-KEM/LWE-derived KDF 就有 LWE 安全性
✗ 能防御可任意修改 kernel、dump 寄存器的服务器
✗ 任何没有 recorded run 支撑的性能/精度/隐私数字
✗ 用 eager 参考实现的速度代表最终方案
```

允许的措辞：

```text
✓ augmented covariant obfuscation
✓ chained auxiliary noise
✓ bounded logit perturbation
✓ approximate privacy-preserving inference prototype
✓ exact mode / approximate mode
```

### 验收标准
- [ ] 上表 5 条全部写入威胁模型，各带复现命令
- [ ] `results/REPORT.md` 五类结论分节陈述
- [ ] 每个数字都能追溯到 `results/raw/` 里的具体文件与行
- [ ] 移除 `⚠️ SUPERSEDED` 标记（仅当 D1–D3 全部通过）
- [ ] 报告明确写出未达标项，不只写达标项

---

# 附录 A：数学推导

## A.1 结构化基的条件数与代价

`M = Π₁ D B Π₂`。置换矩阵和块正交矩阵都是正交的，`κ₂ = 1`。对角矩阵
`D = diag(exp(s))`，`s` 均匀取自 `[−½ln κ_max, ½ln κ_max]`，所以

```text
κ₂(M) = κ₂(D) = max(exp s) / min(exp s) ≤ κ_max
```

这就是 `generate_structured_basis` 用 `scales.max()/scales.min()` 作为
`condition_number` 的依据——不需要算 SVD。
`structured_condition_number()` 用 FP64 稠密 SVD 交叉校验。

应用代价：

```text
z @ M  =  blockwise(z[argsort(Π₁)] ⊙ d, B)[argsort(Π₂)]
```

gather O(n) + 逐元素 O(n) + `m` 个 `b×b` GEMM 即 O(n·b) + gather O(n)。
`n=3088, b=16`：49K MAC vs 稠密 9.5M，**194×**。

## A.2 Gram 技巧

`P = M⁻¹[:, :d]`。因为 `M⁻¹ = Π₂ᵀ Bᵀ D⁻¹ Π₁ᵀ`，取前 `d` 列相当于右乘一个
选择矩阵 `S_d`：

```text
P = Π₂ᵀ Bᵀ D⁻¹ Π₁ᵀ S_d
A = P Pᵀ = Π₂ᵀ Bᵀ D⁻¹ (Π₁ᵀ S_d S_dᵀ Π₁) D⁻¹ B Π₂
```

`Π₁ᵀ S_d S_dᵀ Π₁` 是 0/1 对角矩阵（`J`），所以

```text
A = Π₂ᵀ (Bᵀ Λ B) Π₂,     Λ = D⁻¹ J D⁻¹ 对角
```

`B` 块对角 ⇒ `Bᵀ Λ B` 块对角 ⇒ **`A` 在 `Π₂` 序下块对角**。
FP64 实测 off-block max = 0.0。因此

```text
‖h‖² = c A cᵀ = Σ_j g_j A_j g_jᵀ,     g = c[Π₂]
```

每块只产生一个标量，`h` 从不出现。

**精度**：`A` 靠抵消消掉噪声子空间，FP32 下相对误差 ~`(‖e‖/‖h‖)²`。
实测见任务 A2.4 的表。这是选 `AUXILIARY_MAGNITUDE_BOUND = 30` 的依据。

## A.3 部署权重的逐项推导

### QKV

明文：`Q = (h/ρ) diag(γ_a) W_Q`，`ρ = rms(h)`。
因为 `h = c P`：

```text
Q = (c P / ρ) diag(γ_a) W_Q = (c @ [P diag(γ_a) W_Q]) / ρ
```

所以 `Wq = P diag(γ_a) W_Q`，形状 `[n, H·dh]`。`bias` 在除 `ρ` **之后**加
（明文里 bias 不参与归一化）。K 同理。

### Value

明文 `V = (h/ρ) diag(γ_a) W_V`，噪声 `e_V = c C_V`。混合：

```text
c_V = [V, e_V] M_V
```

要一次 GEMM 拿到，把 `1/ρ` 提到外面：

```text
c_V = (c @ [P diag(γ_a) W_V | C_V] M_V) / ρ
```

**注意这使 `e_V = (c C_V)/ρ` 而非 `c C_V`。** 这是刻意的设计选择——
辅助态可以是 `(c, ρ)` 的任意确定性函数。已在
`docs/design_identity_check.py` 中注明。

### 注意力输出（最大融合）

明文：`ΔH = O W_O`，`O = A V`，`O` 是 `c_V` 的信号部分。
`mixed_ctx = A c_V`（每 q head `[..., dh+rh]`）。要得到

```text
c_post = c + ΔH (M_top + C_O M_bot) + e(G_O − I) M_bot + ē_V A_ux M_bot
```

把 Value unmix、`W_O`、residual 重混、噪声耦合、辅助项全部合并：

```text
Wattn[i] = M_V⁻¹[kv(i)][:, :dh] W_O^(i) (M_top + C_O M_bot)
         + (1/H) M_V⁻¹[kv(i)][:, dh:] A_ux M_bot
```

于是 `c_post = c + einsum(mixed_ctx, Wattn) + (c N) Wnz_a + ξ_O M_bot`。

**结果：明文 `O` 从不物化。** 这比手册 §21.2（只要求 `A` 不写回显存）更强。

### FFN

SwiGLU 协变（手册 §26）：`silu(g P_f) ⊙ (u D_f P_f) = (silu(g) ⊙ u) D_f P_f`，
即 `z' = z D_f P_f`。补偿 Down（§27）：`W_d' = P_fᵀ D_f⁻¹ W_d`，使
`z' W_d' = z W_d`。合并 residual 与两个噪声耦合：

```text
Wffn = (D_f[P_f]⁻¹ W_d[P_f]) (M_top + C_d M_bot) + C_z M_bot
```

### 为什么 `Wnz` 是 `(G − I) M_bot`

残差写法 `c_next = c + Δ`，而 `c` 已含 `e M_bot`。要让噪声从 `e` 变 `eG`：

```text
需要增量 = eG M_bot − e M_bot = e(G − I) M_bot
```

`e = c N`，所以 `Δ_noise = (c N)(G − I) M_bot`。**漏掉 `−I` 会让噪声变成
`e(1+G)`，链式衰减失效、`‖e‖` 增长，进而触发 A2.4 的精度问题。**

---

# 附录 B：故障排查

## B.1 exact 误差按量级定位

| 误差量级 | 最可能原因 | 首先检查 |
|---|---|---|
| ~1e-7（相对） | 正常，`ρ` 的 FP32 瓶颈 | 无需处理 |
| ~1e-5 | 部署权重存成了 FP32（正常） | 用 FP64 构建复测以隔离代数 |
| ~1e-3 | `γ` 重复应用，或 `bias` 加在除 `ρ` 之前 | A.3 的 QKV 段 |
| ~1e-2 | `Wnz` 漏 `−I` | A.3 末段 |
| ~1e-1 | 置换方向错（词表/神经元/`Π₂`） | 各方向一致性测试 |
| ~1e0 | 权重布局错（数学 vs PyTorch） | `_require_math_layout` |
| 完全不相关 | 基不匹配（跨转换复用 cache） | `cache_identity` |

## B.2 噪声路径异常

| 现象 | 原因 |
|---|---|
| 信号随 `e` 变化 | `Wattn`/`Wffn` 里 `M_top` 与 `M_bot` 弄反 |
| `e` 恒为 0 | `noise_injection_enabled` 被关；或 `Wnz` 全零 |
| `‖e‖` 逐层增长 | `‖G‖₂ ≥ 1`，或 `Wnz` 漏 `−I` |
| `ρ` 失准 | `‖e‖/‖h‖` 超过 30，见 A2.4 |

## B.3 cache decode 与全前缀不一致

| 检查 | 说明 |
|---|---|
| `positions` 是绝对位置 | decode 第 `t` 步应传 `[prompt_len + t]` |
| `key_positions` 拼接顺序 | cache 在前、当前在后 |
| 近似噪声坐标 | `_sample_bounded_noise` 用绝对 q/k 位置，已实现 |
| `ρ` 按 token 独立 | Value cache 存的是已除 `ρ` 的值，正确 |
| cache 身份 | `request_seed` + `request_id` + `cache_identity` 三者都要匹配 |

---

# 附录 C：符号速查

| 符号 | 形状 | 含义 | 可否在 forward 出现 |
|---|---|---|---|
| `c` / `state.mixed` | `[B,S,n]` | 服务端混合态 | ✅ |
| `h` | `[B,S,d]` | 明文隐藏态 | ❌ **阶段 B 后禁止** |
| `e` | `[B,S,r]` | 辅助噪声态 | ❌ |
| `ρ` | `[B,S,1]` | `sqrt(‖h‖²/d+eps)` | ✅ 标量 |
| `M` | `[n,n]` | 混合基 | ❌ 仅离线 |
| `P` | `[n,d]` | `M⁻¹[:, :d]` | ❌ 仅离线 |
| `N` | `[n,r]` | `M⁻¹[:, d:]` | ✅ 在部署权重中（见 B1.3） |
| `M_top` / `M_bot` | `[d,n]`/`[r,n]` | `M` 的上/下块 | ❌ 仅离线 |
| `A_gram` | `[m,b,b]` | `P Pᵀ` 的块 | ✅ 仅用于 `ρ` |
| `mixed_ctx` | `[B,H,S,dh+rh]` | 混合注意力上下文 | ✅ |
| `O` | `[B,H,S,dh]` | 明文注意力上下文 | ❌ **不物化** |
| `z'` | `[B,S,dff]` | `z D_f P_f` | ✅ |
| `common_qk` | `[Hkv,dh,dh]` | RoPE 后 Q/K 正交基 | ✅ 服务端可见密钥 |

---

# 附录 D：交付物清单

阶段 A–D 全部完成后应存在：

```text
src/fastprove/
├── structured.py                    A2 ✅ 已实现
├── codec.py                         C1
├── converter.py                     C4
└── layers/
    ├── deployed.py                  B1 ✅ 已实现
    └── embedding.py                 C2, C3

tests/
├── test_structured_basis.py         A2 ✅ 20 passed
├── test_deployed_weights.py         B1 ✅ 17 passed
├── test_key_material_isolation.py   A3
├── test_deployed_matches_chain_linear.py  B5
├── test_no_decode_in_forward.py     B6
├── test_token_codec.py              C1
├── test_secure_embedding.py         C2
├── test_secure_lm_head.py           C3
├── test_converter.py                C4
└── test_moe.py                      C5

scripts/
├── record_exact_tolerance.py        A4
└── measure_overhead.py              B7

docs/
├── IMPLEMENTATION_PLAN.md           本文档
├── design_identity_check.py         ✅ 13 项恒等式全绿
├── mathematics.md                   附录 A 的正式版
├── threat_model.md                  + D4 的 5 条新条目
└── implementation_notes.md          + 各阶段实测记录

results/raw/
├── exact_tolerance_A{4,5}.json
├── exact_tolerance_B{2,3}.json
├── overhead_B7_d{512,1024,2048}.json
└── <D2/D3 的性能与 sweep 记录>
```

## 最终验收（AGENTS.md「Definition of Done」）

- [ ] plaintext / exact / topk_preserving / free_bounded 四种模式都存在
- [ ] exact 模式数学测试通过（**FP32 判定**）
- [ ] 至少一次有意义的预训练小模型评测已跑，或明确记录模型/数据/算力缺失
- [ ] 配置的噪声 sweep 已产出原始结果（63 点全覆盖）
- [ ] 精度退化在表格和图中总结
- [ ] `results/REPORT.md` 说明最佳操作点与限制
- [ ] 所有数字可从本仓库的命令和原始工件复现
