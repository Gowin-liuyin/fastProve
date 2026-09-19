# AGENTS.md

状态：CURRENT（上游规则，中文译本；与历史英文原文语义一致）
日期：2026-09-19
上游依据：用户需求与仓库历史（本文件为最高约束）
适用范围：项目使命、数学规范、威胁模型、实现阶段与验收门

## 项目使命（Project Mission）

本仓库构建一个用于 Transformer 推理协变混淆（covariant obfuscation）的可复现研究原型。

当前目标是：

1. 实现一个明文 Transformer 参考实现；
2. 实现一条精确协变混淆路径，其输出应与明文参考在浮点误差范围内一致；
3. 实现一条带界 logit 噪声的近似 Softmax 混淆路径；
4. 测量随噪声强度增加，相对明文推理损失了多少精度；
5. 产出可复现的代码、原始指标、图表与报告。

该原型用于检验数学与工程可行性。它不是生产密码系统。

## 研究背景（Research Background）

研究起点是增广状态协变变换。隐藏状态被扩展一个低维噪声状态，然后由可逆基混合：

\[
h_\ell\in\mathbb R^{d_\ell},\qquad
e_\ell\in\mathbb R^{r_\ell},
\]

\[
z_\ell=[h_\ell,e_\ell],
\qquad
c_\ell=z_\ell M_\ell.
\]

其中：

- \(h_\ell\) 是模型的语义信号；
- \(e_\ell\) 是辅助噪声状态；
- \(r_\ell\ll d_\ell\)；
- \(M_\ell\in\mathbb R^{(d_\ell+r_\ell)\times(d_\ell+r_\ell)}\) 可逆；
- \(c_\ell\) 是暴露给混淆服务端路径的状态。

较早的一维增广矩阵构造是 \(r_\ell=1\) 的特例。它的线性链代数是有效的，但其原始
非线性桥不完整：一旦纯乘性噪声在非线性层之前被置零，后续非零乘子无法重新产生
它。因此本项目使用仿射噪声刷新与显式非线性检查点。

## 可声明与不可声明（Claims and Non-Claims）

项目只能声明由数学或生成的实验工件所证明的内容。

允许的术语：

- 增广协变混淆（augmented covariant obfuscation）；
- 链式辅助噪声（chained auxiliary noise）；
- 有界 logit 扰动（bounded logit perturbation）；
- 近似隐私保护推理原型（approximate privacy-preserving inference prototype）；
- 精确模式与近似模式（exact mode and approximate mode）。

不得声明：

- 激活是标准 LWE 密文；
- 全同态或端到端加密推理；
- 仅因为使用了 ML-KEM、LWE 派生 KDF 或 LWE 启发的种子，就声称具备 LWE 安全性；
- 能防御可任意修改内核并转储寄存器的服务端；
- 任何未由记录在案的运行产生的性能、精度或隐私数字。

若日后加入 LWE/ML-KEM，其角色仅是密钥建立或参数派生，除非有单独的归约证明更多。

## 威胁模型（Threat Model）

初始原型评估的是：针对持久化张量与普通框架输出的诚实但好奇（honest-but-curious）
观察者的表示混淆。

原型可以假设指定的融合操作不返回其内部干净临时值。这是一项实现假设，不是密码学
保证。

若攻击者能够：

- 修改 CUDA/Triton/PyTorch 内核；
- 挂接任意 hook；
- 转储寄存器或临时缓冲区；
- 读取全部客户端秘密；

则仅凭本原型无法保护 Softmax logits、概率或被提取的信号分量。更强的保护需要可信
执行环境、MPC、HE 或另一个可信边界。

每份报告都必须写明这一限制。

## 规范数学规格（Canonical Mathematical Specification）

全文采用行向量数学：

\[
Y=XW+b.
\]

PyTorch `torch.nn.functional.linear` 以 `[out_features, in_features]` 存储权重，
因此每个转换函数都必须写明权重处于数学布局还是 PyTorch 布局。

### 混合状态（Mixed State）

\[
c_{\rm in}=[h,e]M_{\rm in}.
\]

默认原型维度：

```text
hidden noise dimension r: 8 or 16
attention value noise per head r_h: 1, 2, or 4
```

使用良态变换。“稠密”或“全非零”既不代表安全，也不代表数值稳定。

推荐结构：

\[
M=\Pi D B,
\]

其中：

- \(\Pi\) 是排列；
- \(D\) 是有界非零对角缩放；
- \(B\) 是正交或小块正交变换；
- 条件数被检查并记录。

默认条件数上限应可配置且保守，例如 `max_condition_number: 10`。

绝不在 `forward` 中计算矩阵逆。在 FP64 或 FP32 中离线计算变换、验证它们，并存储
转换后的权重。

### 仿射线性层与噪声刷新（Affine Linear Layer and Noise Refresh）

对于：

\[
y=hW+b,
\]

定义：

\[
e'=hC+eG+\xi.
\]

令：

\[
K=
\begin{bmatrix}
W&C\\
0&G
\end{bmatrix}.
\]

部署：

\[
\widetilde W=M_{\rm in}^{-1}KM_{\rm out},
\]

\[
\widetilde b=[b,\xi]M_{\rm out}.
\]

于是：

\[
c_{\rm out}
=
c_{\rm in}\widetilde W+\widetilde b
=
[hW+b,\;hC+eG+\xi]M_{\rm out}.
\]

这条恒等式必须有直接的单元测试。

使用稳定的噪声传播器，例如：

\[
G=\gamma P_e,\qquad 0<\gamma<1,
\]

其中 \(P_e\) 是排列或带符号排列。

必须区分两种刷新模式：

- `fixed_debug`：固定的 \(\xi\) 存入转换后的偏置；只用于正确性测试；
- `per_request`：\(\xi\) 在指定融合操作内部由可复现的逐请求、逐层种子生成。

不得把确定性的固定刷新描述为强隐私。

## 非线性检查点规则（Nonlinear Checkpoint Rule）

一般稠密增广变换不与非线性函数交换。

在非线性检查点：

1. 在参考或融合操作内部导出被允许的变换后信号；
2. 把旧噪声保留在旁路上，而不是销毁它；
3. 只对数学上有效的信号表示应用非线性函数；
4. 构造刷新后的噪声状态；
5. 返回重新混合后的状态；
6. 不通过公共生产模式 API 暴露干净的中间张量。

参考/调试代码可以为测试暴露解码状态，但每个这样的函数都必须命名清晰，并在生产
模式配置下禁用。

## RMSNorm

对于无 gamma 的 RMS 归一化与正交矩阵 \(R\)：

\[
u=hR,\qquad RR^T=I,
\]

\[
\operatorname{rms}(u)=\operatorname{rms}(h).
\]

学习到的逐元素缩放 \(\Gamma=\operatorname{diag}(\gamma)\) 通常不与任意 \(R\) 交换。
它必须被吸收进后续投影。

对于 Query：

\[
Q=
\frac{h}{\operatorname{rms}(h)}
\Gamma W_Q,
\]

而 \(u=hR\) 时使用：

\[
Q'=
\frac{u}{\operatorname{rms}(u)}
R^T\Gamma W_QC_Q.
\]

Key、Value、Gate 与 Up 需要等价公式。

RMS 统计、epsilon 加法与归约必须使用 FP32。

## RoPE 与 Q/K（RoPE and Q/K）

首选的原型顺序是：

1. 在检查点内获得有效的 Query 与 Key 投影；
2. 应用 RoPE；
3. 在 RoPE 之后应用共同正交变换 \(C\)。

\[
Q'=\operatorname{RoPE}(Q)C,
\]

\[
K'=\operatorname{RoPE}(K)C.
\]

因此：

\[
Q'K'^T=Q_{\rm rope}K_{\rm rope}^T.
\]

在 RoPE 之后应用 \(C\) 可以避免一个不必要且通常错误的交换假设。

对于分组查询注意力（GQA），每个共享同一 KV 头的 Query 头必须使用与该 KV 头兼容
的变换。为 GQA 映射添加显式测试。

## Value 路径（Value Path）

Value 可以携带增广噪声状态，因为注意力概率从左侧线性乘 Value：

\[
c_V=[V,e_V]M_V.
\]

对任意注意力矩阵 \(A\)：

\[
Ac_V=[AV,Ae_V]M_V.
\]

这是让噪声通过注意力块的主要机制。

## Softmax 模式（Softmax Modes）

Softmax 按行作用，不与一般稠密基变换交换。

精确协变族是有限的。对于列排列 \(P\) 与行相关公共偏移 \(a\mathbf 1^T\)：

\[
\operatorname{softmax}(SP+a\mathbf 1^T)
=
\operatorname{softmax}(S)P.
\]

不要向 Softmax 添加有限值的零 dummy 位置。它会改变归一化分母。任何 dummy 或无效
位置都必须保持被掩码为 \(-\infty\)。

在同一接口后实现三种注意力模式。

### `plaintext`

\[
A=\operatorname{softmax}(S).
\]

这是参考基线。

### `exact`

使用共同正交 Q/K 变换、无 logit 扰动与混合 Value 路径：

\[
S=\frac{Q'K'^T}{\sqrt{d_h}},
\]

\[
O_{\rm exact}
=
\operatorname{softmax}(S)c_V.
\]

该模式必须在数值容差内与明文信号输出一致。

可选的位置键排列日后可以加入：

\[
\widetilde S=SP+a\mathbf 1^T,\qquad
\widetilde V=P^Tc_V.
\]

于是：

\[
\operatorname{softmax}(\widetilde S)\widetilde V
=
\operatorname{softmax}(S)c_V.
\]

掩码与 KV 布局必须一致地排列。该可选扩展只隐藏位置标签；它不隐藏分数多重集合或
排序结构。

### `approximate`

只向有效 logits 添加有界噪声：

\[
\widehat S=S+\eta.
\]

被掩码的位置必须保持精确的 \(-\infty\)。

实现两种调度：

1. `topk_preserving`：使用干净的 Top-k 边界 margin 限制噪声；
2. `free_bounded`：使用固定误差预算并允许排序变化。

对于 `topk_preserving`，对有效 logits 排序并定义：

\[
\Delta_k=S_{(k)}-S_{(k+1)}.
\]

令：

\[
\tau
=
\min\left(
\tau_{\max},
\frac{\alpha\Delta_k}{2},
\tau_{\rm error}
\right),
\qquad 0<\alpha<1,
\]

并生成：

\[
\|\eta\|_\infty\le\tau.
\]

于是：

\[
2\tau<\Delta_k
\]

保证 Top-k 集合不变，显式处理的 tie 除外。

对于 `free_bounded`：

\[
\tau=\min(\tau_{\max},\tau_{\rm error}),
\]

且允许排序变化。该模式很重要，因为保持完整排序与隐藏排序是相互冲突的目标。

计算：

\[
\widehat A=\operatorname{softmax}(\widehat S),
\]

\[
\widehat O=\widehat A c_V.
\]

跟踪：

\[
\widehat O-O
=
\left(
\operatorname{softmax}(S+\eta)
-\operatorname{softmax}(S)
\right)V.
\]

噪声生成必须在记录的种子下可复现。对采样噪声做居中与重缩放，使其实际无穷范数
满足所请求的界。

margin 计算、掩码、噪声裁剪与 Softmax 归约必须使用 FP32。

不要从生产模式 API 返回注意力概率。

## SwiGLU

对于：

\[
z=\operatorname{SiLU}(g)\odot u,
\]

使用共享神经元排列 \(P_f\) 与非零有界对角缩放 \(D_f\)：

\[
g'=gP_f,
\qquad
u'=uD_fP_f.
\]

于是：

\[
\operatorname{SiLU}(g')\odot u'
=
zD_fP_f.
\]

转换 Down 权重：

\[
W_d'=P_f^TD_f^{-1}W_d,
\]

所以：

\[
(zD_fP_f)W_d'=zW_d.
\]

让先前的噪声经由旁路传递，并在逐元素乘积后刷新：

\[
e_z=z'C_z+e_{\rm side}G_z+\xi_z.
\]

不要声称 SiLU 与任意缩放或稠密混合交换。

## MoE Router 与 Top-k（MoE Router and Top-k）

对于专家排列 \(P_E\)：

\[
r'=rP_E+\eta_r.
\]

若：

\[
2\|\eta_r\|_\infty
<
r_{(k)}-r_{(k+1)},
\]

则在计入 \(P_E\) 之后，被选中的专家集合不变。

物理专家必须一致地重排。

实现：

- 无 router 噪声的精确专家排列；
- margin 有界 router 噪声；
- 可选的近似噪声 gate 权重。

Tie 打破规则必须与明文实现一致，并有显式测试。

## 原型范围（Prototype Scope）

从小处开始。不要以 7B 模型或自定义 CUDA 起步。

要求的推进顺序是：

1. 确定性小张量数学测试；
2. 一个明文 Transformer 块；
3. 一个精确混淆 Transformer 块；
4. 一个近似混淆块；
5. 本地构造的 tiny 多层 causal LM；
6. 本地可用的小型预训练 Llama-like checkpoint（若有）；
7. 只有在正确性之后，才做可选的 Triton 或融合内核工作。

若本地没有有意义的预训练 checkpoint，随机 tiny 模型只能用于正确性。不得把随机模型
结果描述为有意义的语言模型精度。

下载大模型或数据集前先询问。

## 期望仓库布局（Expected Repository Layout）

创建并维护：

```text
fastProve/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── tiny_exact.yaml
│   ├── tiny_approx.yaml
│   └── eval_sweep.yaml
├── src/
│   └── fastprove/
│       ├── seed.py  hashing.py  io.py  metrics.py  policy.py  modes.py
│       ├── basis_common.py  transforms.py  structured.py
│       ├── state.py  codec.py  coa_views.py  config.py
│       ├── conversion.py
│       ├── noise_budget.py  augmented_noise.py  bounded_noise.py
│       ├── converter/
│       │   ├── __init__.py
│       │   ├── hygiene.py
│       │   ├── load.py
│       │   ├── client.py
│       │   ├── save.py
│       │   └── basis_io.py
│       ├── layers/
│       │   ├── attention.py
│       │   ├── rmsnorm.py
│       │   ├── deployed.py
│       │   ├── linear.py
│       │   ├── swiglu.py
│       │   ├── router.py
│       │   ├── moe.py
│       │   ├── embedding.py
│       │   └── head.py
│       ├── models/
│       │   ├── plain.py
│       │   ├── obfuscated.py
│       │   ├── obfuscated_types.py
│       │   ├── client.py
│       │   └── block_builder.py
│       ├── pretrained/
│       │   ├── _checkpoint_io.py
│       │   ├── llama.py
│       │   └── qwen2.py
│       └── evaluation/
│           ├── environment.py  runtime.py  diagnostics.py  exact_gate.py
│           ├── layer_metrics.py  performance.py  spec_config.py  runner.py
│           ├── metrics.py  accuracy.py  artifacts.py  sweep.py  selection.py
│           ├── token_cache.py  pretrained.py  attack_records.py
│           ├── correctness.py  prompts.py
│           ├── benchmarks/
│           │   ├── tasks.py
│           │   ├── scoring.py
│           │   └── bundle.py
│           └── report.py  report_format.py  report_rows.py  report_tradeoff.py
│               report_figures.py  report_tables.py  report_markdown.py
├── evals/                         # 顶层第二套评测 harness（python -m evals）
│   ├── report.py  conditions.py  keys.py  thresholds.py  stats.py
│   ├── model_factory.py  metrics_common.py
│   ├── layer1_operator_equiv.py  layer2_attention.py  layer3_moe.py
│   ├── layer4_output_logit.py  layer5_downstream.py
│   └── configs/
├── scripts/
│   ├── run_correctness.py
│   ├── run_accuracy_sweep.py
│   ├── build_report.py
│   ├── verify_refactor_equivalence.py  check_doc_links.py
│   ├── record_*.py  verify_*.py  measure_overhead.py  ...
│   └── archive/
├── tests/
│   ├── conftest.py  _fixtures.py
│   └── {math,layers,models,deployment,security,evaluation,cli,pretrained}/
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── REPORT.md
└── docs/
    ├── INDEX.md
    ├── architecture.md
    ├── runbook.md
    ├── STATUS.md
    ├── CLEANUP_PLAN.md
    ├── IMPLEMENTATION_PROMPT_dual_channel.md
    ├── mathematics.md
    ├── threat_model.md
    ├── implementation_notes.md
    ├── PROBLEMS.md
    ├── technical_specification.md  work_factor_security.md
    ├── augmented_noisy_obfuscation_plan.md
    ├── literature-review/
    ├── config_reference.md  scripts_index.md   # 计划中（CLEANUP_PLAN §8.3）
    ├── verification/  handoff_evidence/
    └── archive/
```

> 本节只是对现状的描述性清单，不是额外规则；目录随重构演进，权威结构以
> `docs/architecture.md` 与源码为准。带「计划中」标注的文档尚未创建。

不要提交下载的权重、数据集、秘密、缓存或大体积生成物。

## 实现阶段（Implementation Phases）

### 阶段 0：环境审计（Phase 0: Environment Audit）

在改代码之前：

- 检查 Python 与 PyTorch 版本；
- 检查可用 CUDA、MPS、CPU、RAM 与磁盘；
- 把结果记录到 `docs/implementation_notes.md`；
- 判断本地是否存在预训练模型与评测数据集；
- 制定最小依赖计划。

CPU 足以运行单元测试与 tiny 正确性检查。完整模型评测前，必须显式核实算力资源。

### 阶段 1：数学内核（Phase 1: Mathematical Kernel）

实现：

- 带种子的变换生成；
- 条件数验证；
- `MixedState`；
- 仅限调试/测试的 encode/decode 助手；
- `ChainLinear`；
- 仿射噪声刷新；
- 对每条恒等式的直接数值测试。

若线性恒等式失败，不得继续。

### 阶段 2：一个精确块（Phase 2: One Exact Block）

实现共享同一组基础权重的小型明文块与精确混淆块：

```text
RMSNorm
→ Q/K/V
→ RoPE
→ Attention
→ Output projection
→ Residual
→ RMSNorm
→ SwiGLU
→ Down
→ Residual
```

比较每个允许解码的信号检查点。

### 阶段 3：近似 Softmax（Phase 3: Approximate Softmax）

实现：

- `plaintext`；
- `exact`；
- `approximate/topk_preserving`；
- `approximate/free_bounded`。

噪声必须在掩码逻辑确立之后添加，且只加到有效 logits 上。

初始必需扫描：

```text
tau_max:
  0
  0.001
  0.003
  0.01
  0.03
  0.05
  0.10

alpha:
  0.50
  0.80
  0.95

preserve_top_k:
  4
  8
  16
```

扫描必须可配置；不要在层实现里硬编码它。

### 阶段 4：端到端评测（Phase 4: End-to-End Evaluation）

在完全相同的以下条件下运行明文与混淆推理：

- 模型 checkpoint；
- 分词后的样本；
- 序列长度；
- 种子；
- dtype；
- 设备；
- 生成设置。

每次运行存一条机器可读记录。

### 阶段 5：报告（Phase 5: Report）

生成：

- 原始 JSON 或 JSONL 指标；
- 汇总 CSV；
- 精度/噪声图；
- `results/REPORT.md`；
- 精确命令与环境元数据。

报告必须区分数学精确性、浮点偏差、任务精度退化、性能开销与安全限制。

## 评测指标（Evaluation Metrics）

### 层级正确性（Layer-Level Correctness）

记录：

- 最大绝对误差；
- 平均绝对误差；
- 相对 L2 误差；
- 余弦相似度；
- NaN/Inf 计数；
- QK 分数误差；
- exact 模式 Softmax 概率误差；
- Value/注意力输出信号误差；
- Top-k 重叠；
- tie 与掩码行为。

### 端到端精度（End-to-End Accuracy）

对于 causal 语言模型，至少记录：

- 负对数似然；
- 困惑度；
- 下一 token top-1 精度；
- 下一 token top-5 精度；
- 明文/混淆 token 一致率；
- 贪心生成 token 精确匹配；
- 有意义的序列级精确匹配。

若使用下游标注任务，记录：

- 适配数据集的任务指标；
- 绝对退化（百分点）；
- 相对退化；
- 可行时给出样本数与置信区间。

主要比较：

\[
\Delta_{\rm abs}
=
\operatorname{metric}_{\rm obfuscated}
-
\operatorname{metric}_{\rm plaintext}.
\]

对越大越好的指标，还要报告：

\[
\operatorname{drop}_{\rm abs}
=
\operatorname{metric}_{\rm plaintext}
-
\operatorname{metric}_{\rm obfuscated}.
\]

对困惑度，同时报告绝对与相对上升。

### Softmax 专项指标（Softmax-Specific Metrics）

记录：

- 明文与扰动注意力分布之间的 KL 散度；
- Jensen-Shannon 散度；
- Top-k 集合重叠；
- 有效位置上的秩相关；
- Top-k 集合发生变化的 query 占比；
- 干净边界 margin 分布；
- 实际噪声无穷范数；
- 收到零或近零噪声的 query 占比；
- 注意力输出相对 L2 误差。

### 性能（Performance）

分别记录：

- 转换时间；
- prefill 延迟；
- decode 延迟或 TPOT；
- 每秒 token 数；
- 峰值内存；
- KV-cache 内存；
- 参考实现与优化实现的对照。

在 exact 正确性通过前不要优化性能。

## 初始验收标准（Initial Acceptance Criteria）

数学/单元测试门：

- 所有变换往返通过；
- 所有矩阵形状有断言；
- `forward` 中不发生矩阵求逆；
- 小张量测试中 FP32 ChainLinear 最大绝对信号误差不超过 `1e-5`；
- exact 注意力 QK 分数在记录的 FP32 容差内一致；
- exact 模式单块输出在记录的容差内与明文块一致；
- 掩码、因果、cache 与 tie 测试通过；
- 固定种子复现相同指标。

端到端 exact 模式门：

- 无 NaN 或 Inf；
- exact 模式只出现浮点级偏差；
- 在接受近似噪声结论之前，任何贪心不匹配都必须先被调查。

近似模式完成门：

- 每个配置的噪声点都有记录结果或显式失败记录；
- 报告指出最佳精度/隐私操作点；
- 不得静默省略任何结果；
- 结论引用生成的工件路径。

## 算力与实验守卫（Compute and Experiment Guard）

完整评测之前：

1. 可用时用 `nvidia-smi` 检查 CUDA；
2. 检查 PyTorch CUDA 与 MPS 支持；
3. 记录设备名、内存、dtype 与软件版本；
4. 把模型需求与可用资源对比；
5. 不得把计划中的 GPU 评测静默改到 CPU；
6. 算力或数据不可用时不得编造结果。

允许 tiny CPU 单元测试。

## 数据与模型完整性（Data and Model Integrity）

- 记录精确模型标识、本地路径、可用时的 revision/commit 与（可行的）文件哈希。
- 记录数据集名称、划分、样本数、预处理与来源。
- 绝不在不同样本上评测明文与混淆模式。
- 缓存选定的评测样本 ID 或分词输入，使所有模式使用相同数据。
- 未经明确授权，绝不在秘密或私有材料上训练或评测。
- 下载大体积工件前先询问。

## 编码规则（Coding Rules）

- 参考实现使用 Python 与 PyTorch。
- 公共 API 使用类型标注与 docstring。
- 明确区分数学布局与 PyTorch 布局的转换。
- 离线求逆与条件数检查使用 FP64 或 FP32。
- RMS 统计、logits、margin、裁剪与 Softmax 归约使用 FP32。
- 模型激活 dtype 可配置。
- 绝不把请求的安全/融合路径静默回退到 eager 路径。
- 把调试解码 API 与生产 API 分开。
- 生产模式输出不得暴露解码后的隐藏状态、噪声状态、注意力概率或 router logits。
- 保持测试小、确定、快速。
- 实验常量优先放进配置文件而不是硬编码。
- 保留原始结果；派生汇总必须能从原始文件复现。
- 不得仅为让失败测试通过而修改数学定义。要诊断差异。

## Agent 工作方式（Working Style for Agents）

Agent 应在仓库范围内自主执行当前阶段。

实现之前：

- 检查现有文件与未提交改动；
- 说明假设；
- 确认当前阶段；
- 制定简短计划。

实现期间：

- 实现最小完整纵向切片；
- 每个数学组件后运行针对性测试；
- 立即报告意外差异；
- 不得仅凭代码编译就声称完成。

交接时：

- 汇总改动的文件；
- 列出精确的测试与实验命令；
- 分开报告通过、失败与未执行的检查；
- 链接原始结果与报告文件；
- 说明下一个逻辑上必需的阶段。

## 当前项目的完成定义（Definition of Done for the Current Project）

初始项目只有在以下条件全部满足时才算完成：

1. `plaintext`、`exact`、top-k-preserving 近似与 free-bounded 近似模式均存在；
2. exact 模式数学测试通过；
3. 至少一次有意义的预训练小模型评测已运行，或明确记录模型/数据/算力的缺失；
4. 配置的噪声扫描已产出原始结果；
5. 相对明文的精度退化已汇总为表与图；
6. `results/REPORT.md` 说明了最佳操作点与限制；
7. 所有报告数字都可从本仓库内的命令与原始工件复现。
