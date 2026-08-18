# fastProve 问题清单

> 本文档记录实现过程中发现的**全部问题**，按严重度排序。每条包含：现象、根因、
> 实测数据、复现命令、工程可消除性、以及它落在哪一层（数学 / 实现 / 目标）。
>
> 编写原则：
> - 只写有 recorded run 支撑的数字。推测与实测严格分开标注。
> - 不回避对项目不利的结论。
> - 我自己在分析过程中犯的错误也记录在案（§7），因为它们影响过判断。
>
> 最后更新：阶段 A–D 完成后的复核轮次。测试基线 303 passed。

---

## 0. 摘要

问题分三类，**性质完全不同，不可混为一谈**：

| 类别 | 数量 | 是否可修 | 说明 |
|---|---:|---|---|
| **A. 构造的固有泄漏** | 7 条 | ❌ 不可用工程手段消除 | 由「输出精确 + 低开销 + 无可信边界」三个约束强制 |
| **B. 实现缺陷** | 3 条 | ✅ 已全部修复 | 普通 bug，含一条我自己文档引入的 |
| **C. 证据链缺口** | 4 条 | ⏳ 待补 | 不是错误，是尚未验证的部分 |

**最重要的一条**（§1.1）：服务端要算对 RMSNorm 就必须算对 `ρ`，而 `ρ` 代数上就是
`‖h‖`。第 0 层的 `‖h₀‖` 是 token 指纹，实测可恢复 **92.5%（Qwen2）/ 93.4%（Llama）**
的原始输入 token。词表置换对此**完全无效**，融合内核假设**也挡不住**。

**结论定性**：线性层的协变构造是严格正确的（13 项恒等式 + 303 测试）。
隐私目标在非线性层需要的标量上失效。这不是实现失败，是目标与约束互斥。

---

## 1. 类别 A：构造的固有泄漏（不可修）

### 1.1 【最严重】`ρ` 泄漏 `‖h‖`，可直接恢复输入 token

**现象**

服务端只看到词表置换后的 token ID，例如
`[24139, 36140, 43166, 69916, 19409, 83762]`。按设计它不该知道原文。
实测它能恢复出原句。

**根因**

RMSNorm 需要 `ρ = sqrt(‖h‖²/d + eps)`。这不是可选的中间量——算错 `ρ` 则
RMSNorm 错、模型输出错。而：

```
ρ² = ‖h‖²/d + eps   ⟹   ‖h‖² = (ρ² − eps)·d
```

第 0 层 `h₀ = E[token]` 恰好是 embedding 表的一行，所以 `‖h₀‖` 是**逐 token 常量**。
base checkpoint 公开，攻击者离线建「范数 → token」查找表即可。

**为什么词表置换无效**

**行范数对行置换不变。** 置换 `τ` 只重排 embedding 的行，任何一行的范数都不变。
所以任务 C1/C2（TokenCodec + 预混合词表）在这条通道上贡献为零。

**实测数据**（n=4000 随机 token）

| 模型 | `ρ` 存储精度 | 平均候选数 | 唯一且正确恢复 |
|---|---|---:|---:|
| Qwen2-1.5B | FP64 | 1.06 | **98.7%** |
| Qwen2-1.5B | **FP32** | 1.75 | **92.5%** |
| Qwen2-1.5B | BF16 | 4360.42 | 0.0% |
| Llama-3.2-3B | FP64 | 1.00 | **100.0%** |
| Llama-3.2-3B | **FP32** | 1.07 | **93.4%** |
| Llama-3.2-3B | BF16 | 4491.47 | 0.0% |

embedding 行范数互不相同的比例：Qwen2 **99.05%**，Llama-3.2 **100.00%**。

端到端演示（Qwen2，8 token）：

```
位  服务端观测 rho     候选  恢复
0   0.0336763351     1    'èµ·'        ✓
1   0.0351185429     1    ','          ✓
2   0.0300626124     1    ' I'         ✓
3   0.0317753381     1    ' am'        ✓
4   0.0294984194     1    ' a'         ✓
5   0.0290927488     1    ' patient'   ✓
6   0.0279302316     1    ' with'      ✓
7   0.0288048360     1    ' diabetes'  ✓
8/8 精确恢复 → "I am a patient with diabetes"
```

**攻击前提（都很弱）**

| 前提 | 是否需要 |
|---|---|
| 已知明文对 `(h, c)` | ❌ 不需要 |
| 知道混合基 `M` | ❌ 不需要 |
| 修改 kernel | ❌ 不需要 |
| 读取寄存器 | ❌ 不需要 |
| 公开 base checkpoint | ✅ 需要（DeepSeek/Llama 都是公开模型） |

**核心张力**

`AGENTS.md` R5 要求 RMS 统计用 **FP32 或更高**，exact 门禁也在 FP32 下判定。
**正是这个精度让攻击成立。** BF16 的 `ρ` 恢复率降到 0%，但违反 R5、且
4.7e-03 的相对误差远超门禁的 1e-3 logit 容差。

BF16 不是解法，它把隐私失败换成了正确性失败。

**适用范围（不夸大）**

该攻击针对第 0 层，因为那里 `h₀` 恰好等于单个 embedding 行。更深层 `h` 是
上下文相关的，「范数 → token」一对一关系不成立（但见 §1.2）。第 0 层就是**输入**，
而输入正是要隐藏的东西。

**复现**

```bash
PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py \
    --model-path <本地明文 checkpoint 目录> \
    --output results/raw/token_recovery_from_rho.json
```

原始记录：`results/raw/token_recovery_from_rho.json`、
`results/raw/token_recovery_from_rho_llama.json`

**工程可消除性**：❌ 不可。`ρ = rms(h)` 是模型语义的一部分。

---

### 1.2 融合内核假设挡不住 `ρ`

**现象**

最自然的辩护是：`ρ` 只存在于融合内核的寄存器中，从不写回显存，攻击者读不到。
方案文档 §15.4 明确要求「信号提取 → RMSNorm → Q/K/V 在一个融合流程中完成」，
威胁模型也假设「指定的融合操作不返回其内部干净的临时值」。

**这个辩护不成立。**

**根因**

融合 RMS-QKV 操作必须输出 `Q`（注意力要用它），而服务端必然持有 `c` 和部署权重
`W_q`。因为：

```
Q = (c · W_q) / ρ        ⟹        ρ = (c · W_q) / Q
```

一次逐元素除法。`ρ` 不是任何意义上的「内部临时值」——它由**必须同时存在**的
输入和输出代数确定。

**实测数据**

| `Q` 存储精度 | 反推 `ρ` 相对误差 | 反推 `‖h‖` 相对误差 |
|---|---:|---:|
| FP64 | **0.000e+00** | 4.41e-16 |
| FP32 | 1.45e-08 | 1.45e-08 |
| BF16 | 7.89e-04 | 7.89e-04 |

注意最后一行：**即使 `Q` 用 BF16 输出，反推 `‖h‖` 的精度（7.9e-04）仍好于
`ρ` 自身用 BF16 存储时的精度（4.7e-03）。** 所以 §1.1 里那条 BF16 缓解比
表面看起来更弱。

**复现**

```bash
PYTHONPATH=src python scripts/verify_rho_recoverable_from_fused_output.py \
    --output results/raw/rho_recoverable_from_fused_output.json
```

**工程可消除性**：❌ 不可。除非 `Q` 也不输出，但那样注意力无法计算。

**这条的意义**：它说明方案文档的安全边界描述有**层次上的遗漏**。文档假设的威胁是
「读取内部临时值」，真实威胁是「从必须存在的输出反推」。后者不需要任何特权。

---

### 1.3 范数泄漏不随深度衰减（边云切分无法消除）

**现象**

对 §1.1 的自然缓解是边云切分：客户端本地算 embedding + 前 k 层，服务端从第 k 层
开始，这样 `ρ₀` 从不出现在服务端。

**实测表明这不能消除通道。**

**根因**

残差流保留 embedding 贡献：`h_ℓ = h₀ + Σ(更新)`，所以 `‖h_ℓ‖` 始终与 `‖h₀‖`
强相关。

**实测数据**（Qwen2-1.5B 全 28 层，120 条真实 prompt，范数作唯一特征做 token
最近邻预测）

| 层 ℓ | top-1 准确率 | 稀有 token (1-2 次) | 最高频基线 |
|---:|---:|---:|---:|
| 0 | 98.7% | 100.0% | 8.7% |
| 1 | 77.9% | 49.4% | 8.7% |
| 2 | 77.6% | 50.6% | 8.7% |
| 24 | 74.6% | 49.4% | 8.7% |
| 28 | **74.9%** | **49.4%** | 8.7% |

**⚠️ 语料局限（必读，不得直接引用 top-1 那一列）**

该语料高度模板化：train 去重后仅 **365** 个 token，test 去重 331 个，
test 中 **98.7%** 的 token 在 train 出现过，熵 7.29 bits（均匀应为 8.37）。
最近邻攻击在这种语料上被显著抬高。

**可引用的是稀有 token 那一列：49.4%，基线 8.7%，5.7 倍。** 它说明范数通道在
所有深度都携带真实 token 信息，而非仅在利用重复。

§1.1 的第 0 层数字（92~93%）**与语料无关**——它直接用公开 embedding 表，
不需要训练集。引用第 0 层请用那个数字。

**复现**

```bash
PYTHONPATH=src python scripts/verify_norm_leak_by_depth.py \
    --model-path <checkpoint> \
    --prompt-file results/raw/real_scenario_prompts_1500.jsonl \
    --output results/raw/norm_leak_by_depth.json
```

**工程可消除性**：⚠️ 部分。切分到第 1 层可把 98.7% 降到 77.9%（稀有 49.4%），
但无法归零。且切分违反方案 §5 原则三（客户端只编解码）。

---

### 1.4 服务端必须持有 Gram 块，`h` 的欧氏几何精确暴露

**现象**

部署路径用 `A_gram = P Pᵀ` 的块对角形式算 `ρ`，所以 `deployed_gram_blocks` 和
`deployed_gram_perm` **必然**在服务端 bundle 中。

**根因**

`A = P Pᵀ` 只把 `P` 确定到一个右正交因子：对任意满足 `A = R Rᵀ` 的 `[n,d]` 因子
`R`（例如对称特征基），都有 `c R = h Q`，`Q` 正交。

**实测数据**（d=256, r=16, b=16, FP64）

| 泄漏量 | 最大绝对误差 | 不用 Gram 的基线 |
|---|---:|---:|
| 逐 token 范数 `‖h‖` | 2.13e-14 | 11.84 |
| 所有成对内积 | 8.53e-13 | — |
| 所有成对距离 | 1.07e-06 | — |

误差在 FP64 舍入量级 ⇒ 几何是**精确**暴露而非近似。

**这推翻了威胁模型原 5bis.3 的说法。** 原文说 `κ ≤ 10` 给出距离结构保真度
**上界**（相关性 0.66，拉伸比 [1.19, 1.84]）。实际距离不是被界住，而是可精确读出。
embedding-inversion 与语义分类类攻击所需的条件被完全满足。

**复现**

```bash
PYTHONPATH=src python scripts/verify_gram_leakage.py \
    --output results/raw/gram_leakage.json
```

**工程可消除性**：❌ 不可。隐藏 Gram 块会使 `ρ` 退化为 O(n²) 稠密二次型
（违反 ≤5% 目标），且 §1.1 的范数通道仍然精确。

---

### 1.5 `M` 可逆 ⇒ `h` 无损可恢复，加大噪声无用

**现象**

`c = [h,e]M`，`M` 是实数可逆矩阵，无模运算、无量化。所以
`h = c·(M⁻¹)[:, :d]` 精确成立。

**实测数据**

| `‖e‖` 量级 | `‖cP − h‖/‖h‖` |
|---|---:|
| 10⁰ | 2.1e-15 |
| 10⁶ | 5.6e-10 |
| 10¹² | 5.7e-04 |

**噪声放大 10¹² 倍后仍可恢复到 5.7e-04。** 残差来源是浮点精度，不是噪声的
遮蔽作用。

**推论**：增广是**无损**变换。增大 `‖e‖` 不增加恢复难度，`r/d` 比例不改变结论。
所有机密性都压在「`M` 保密」这一个假设上，而不在噪声上。

**工程可消除性**：❌ 不可（在实数可逆构造下）。要改变需换代数结构
（mod-q + 离散误差分布），那与 ≤5% 开销目标冲突。

---

### 1.6 噪声读出矩阵 `N` 进入部署权重，`e` 可恢复

**现象**

残差项的在线形式是 `(c @ N) @ Wnz`，其中 `Wnz = (G − I) M_bot`。所以 `N` 在
部署权重里。

**根因（已验证不可规避）**

`Z = N(G−I)M_bot` 的秩**恒为 `r`**，且对任意分解 `Z = UV`，`M @ U` 的信号块为 0、
噪声块满秩。实测：`rank(Z) = 8`（r=8），`‖M@Ur 上块‖ = 5.79e-16`，
`rank(下块) = 8`。

因此观察部署权重者可恢复 `e`（up to 一个可逆 `r×r` 映射）。

**工程可消除性**：⚠️ 唯一替代是令 `G = 0`（放弃链式衰减）。当前保留 `G = γP`
并接受此泄漏，因为它不比 §1.5 的结论更弱。

**附带影响**：这使 §1.7 的 `e ≈ hC` 通道直接可利用。

---

### 1.7 默认参数下辅助噪声 98% 是 `h` 的线性像

**现象**

刷新式为 `e' = hC + eG + ξ`。以 d=3072、`C` scale=0.02、`ξ` scale=0.02 计：
`‖hC‖ ≈ 2.94`，`‖ξ‖ ≈ 0.056`。

即「辅助噪声」中约 **98%** 是 `h` 的线性像。

**推论**：这是独立于 `M` 可逆性的第二条泄漏通道。即使将来把 `M` 换成有损投影，
只要保留 `e ≈ hC` 且 `C` 可估，泄漏依然存在。

**工程可消除性**：⚠️ 可调（增大 `ξ` 相对 `C` 的尺度），但受 §2.1 的
`‖e‖/‖h‖ ≤ 30` 上界约束，且 §1.5 说明加大噪声本身不增加 `h` 的恢复难度。

---

## 2. 类别 A 的附带约束（不是泄漏，但限制了方案空间）

### 2.1 `ρ` 的 FP32 精度给噪声幅度设了硬上界

`A_gram` 靠**抵消**消去噪声子空间，FP32 相对误差按 `(‖e‖/‖h‖)²` 增长：

| `‖e‖/‖h‖` | 二次型 FP32 | 二次型 FP64 | 因子式 FP32 |
|---:|---:|---:|---:|
| 1 | 3.5e-07 | 7.5e-16 | 1.0e-07 |
| 10 | 1.5e-06 | 4.1e-15 | 2.1e-07 |
| 100 | **1.3e-04** | 3.7e-13 | 1.6e-06 |
| 1e3 | **2.0e-02** | 4.3e-11 | 1.1e-05 |
| 1e4 | **2.9e+00** | 1.6e-08 | 1.6e-04 |

**后果**：想靠加大噪声提升混淆强度会**静默污染信号路径**（`ρ` 失准），
而不是只污染噪声路径。`AUXILIARY_MAGNITUDE_BOUND = 30` 是实测上界，
转换期由 `validate_auxiliary_budget` 强制。

因子式（`y = cP`，`‖h‖²=‖y‖²`）数值更稳，但会**物化 `h` 的缩放置换**，
直接违背阶段 B 的目标。所以选二次型 + 幅度上界。

**这条方案文档完全没有提到。**

### 2.2 块对角基降低已知明文攻击的样本代价

`M = Π₁ D B Π₂` 中 `B` 块宽为 `b`，混合态每个输出坐标只依赖 `b` 个输入坐标。
恢复每块内的 `M` 只需 `O(b)` 个已知明文样本（稠密基为 `O(n)`）。
`r ≪ d` 时大部分块不含噪声坐标。

代价从 9.5M MAC 降到 49K MAC 的同时，攻击者按块分解的代价也成比例下降。

### 2.3 `common_qk` 服务端必须可见

它在 RoPE 之后应用，无法吸收进任何部署权重。后果：`Q'K'ᵀ = QKᵀ` 的几何
（分数多集与排序结构）对服务端可见。

已在 `runtime_config.json` 的 `server_visible_key_material` 中显式标注
（**不允许改名规避检查**）。

### 2.4 tied embedding 使峰值内存超出 5% 目标

Llama-3.2 的 `lm_head` 与 `embed_tokens` 共享权重。`untied_deployed` 模式下
`deployed_head` 是独立 `[n, V]` 矩阵，词表侧内存从共享的 `[V, d]` 变为
`[V, n] + [n, V]`，比值 `2n/d ≈ 2.03`（d=3072, r=16）。

3B/BF16 投影约 **+0.79 GB**（**未实测**：本地无 tied 预训练 checkpoint 的
完整评测）。

`fused_norm_head` 模式需要融合 kernel（eager 会物化明文归一化态），
当前实现抛 `NotImplementedError`，不静默回退。

---

## 3. 类别 B：实现缺陷（已全部修复）

### 3.1 【我的文档引入】`test_no_decode_in_forward.py` 是空测试

**现象**

这是阶段 B 唯一防止「解码重新爬回生产路径」的守卫。往 block 的 `forward()` 里
注入一行真实解码，它照样 **2 passed**。

**根因**

`docs/IMPLEMENTATION_PLAN.md` 早期版本给出的静态 AST 实现有两处缺陷：
1. 最后那行宽泛过滤 `if "_debug_unmix" not in item` 把真正的泄漏也一起放过
2. 实际方法名 `_debug_basis_unmix` 根本不在禁止列表里

更根本的是：**纯静态分析无法区分「在 `_run` 的 `return_debug` 分支内」和
「在主路径上」**，除非做控制流分析。

**责任**：这个 bug 由我的文档代码带来，不是实现者的问题。

**修复**

改为**运行时哨兵**：把解码闭包替换成抛异常的函数，跑每条生产入口。8 个测试，
含两个元测试（确认哨兵真的替换到了东西、确认 debug 路径会触发哨兵——否则前面
几条可能只是因为哨兵不可达而通过）。

**注入自检确认有效**：注入后 3 failed，恢复后 8 passed。

**教训**：一个抓不到注入泄漏的守卫比没有守卫更糟，因为它给出虚假保证。
「303 tests 全绿」这个数字本身需要审视——测试通过不等于测试有效。

### 3.2 三处硬编码 dtype 字面量，使 72 条 sweep 记录溯源字段错误

**现象**

`run_accuracy_sweep.py`、`runner.py`（两处）、`preflight_experiment.py` 都写着
`"float64" if device == "cpu" else "float32"`。任务 A4 已把它改成 FP32、
阶段 B 已删除该函数，所以 72 条记录里的 `checkpoint_compute_dtype: "float64"`
**全是错的**。

违反 `AGENTS.md`「recorded dtype 必须是 executed dtype 而非 label」。

**影响范围（已查清）**

容差档只由 `activation_dtype` 和 `pretrained` 决定，**不依赖 `checkpoint_dtype`**。
所以这是溯源错误，**没有污染任何门禁判定**，已有结果不作废。

**修复**

加 `structured.py:reduction_compute_dtype_name()` 作单一真相来源，五处全部改为派生。
重跑 sweep 到 `eval_sweep_dtype_fixed.jsonl` 验证：72 个 spec 的精度指标
**逐位相同**，唯一差异就是被修的 label（另有 1508 项墙钟时间/内存差异，
这类本质不可逐位复现）。

### 3.3 威胁模型缺 Gram 泄漏条目，且原 5bis.3 低估了泄漏

见 §1.4。原文用 `κ ≤ 10` 描述距离保真度上界，实际是精确暴露。
已补 5bis.8 条目 6 并在 5bis.3 加前向警告。

---

## 4. 类别 C：证据链缺口（待补，不是错误）

### 4.1 【最关键】所有精度结论跑在随机 tiny 模型上

**现状**

exact 门禁、72 点 sweep 全部在 `fastprove-random-tiny-correctness-only` 上跑的，
`meaningful_lm_evidence: False`。

**为什么这是个问题**

它同时削弱两个方向的结论：
- 正面的（构造在真实模型上数学精确）——缺实测
- 负面的（这类构造泄漏）——容易被质疑成「你们实现有问题」

**已确认可解**

预训练 checkpoint **就在本机**：

```
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/
├── deepseek-r1-distill-qwen-1.5b/base_model/   (Qwen2ForCausalLM, d=1536, 28层)
└── llama-3.2-3b-instruct/base_model/           (LlamaForCausalLM, d=3072, 28层)
```

Llama 的 `config.json` sha256 与 `results/REPORT.md` 自己记录的值**逐字相同**。
`pretrained_evaluation_status.json` 里「本地无 Llama 系 checkpoint」的记录是**误判**。

**建议优先做 Qwen-1.5B**：`tie_word_embeddings: false`（避开 §2.4）、
无 rope_scaling（避开 §4.2）、1.5B BF16 约 3.5 GB 本机 MPS 可跑、
Qwen2 adapter 已完整接入 `evals/model_factory.py`。

### 4.2 未实现 llama3 rope_scaling，Llama 基线本身不对

**现象**

Llama-3.2-3B 的 `config.json` 有：

```json
"rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                 "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                 "original_max_position_embeddings": 8192}
```

而 `layers/rmsnorm.py:apply_rope` 只实现单一 theta 的原始 RoPE。

**实测影响**（64 个频率对中 29 个被除以 32、6 个插值、29 个不变）

| 位置 | max\|cos 差\| |
|---:|---:|
| 32 | 0.0015 |
| 128 | 0.023 |
| 512 | **0.33** |
| 2048 | **1.61** |

**关键区分**：这**不影响 exact 门禁**——明文与混淆路径用同一个（错的）RoPE，
协变恒等式仍然成立。它影响的是**与 HF 参考实现的一致性**，所以任何在 Llama 上
测的 PPL / 任务精度，其基线本身是错的。

seq_len=32 时偏差很小（0.0015），所以旧报告那 10.4pp 基线缺口主因是它自述的
其他三条（MMLU 头部截断、PIQA 标签全 0、无长度归一化）。但 MMLU 提示词通常
几百 token，一旦在真实长度上跑就会被击穿。

### 4.3 加噪缓解的真实任务精度未测

**现状**

给 `ρ` 加有界噪声是目前唯一测通的缓解。真实 Qwen2-1.5B 全 28 层：

| τ | token 恢复率 | 平均候选 | logit max err | argmax 一致 | 归类 |
|---|---:|---:|---:|---:|---|
| 0 | **93.8%** | 1.8 | 0 | 100% | exact |
| 1e-6 | 49.2% | 8.6 | 1.1e-3 | 100% | approximate |
| **1e-5** | **3.9%** | 50.8 | 2.0e-3 | **100%** | approximate |
| 1e-4 | 0.2% | 125.5 | 2.2e-2 | 100% | approximate |
| 3e-4 | 0.1% | 277.7 | 6.5e-2 | 100% | approximate |

放大倍数次线性：6 层 1.78×、14 层 2.80×、28 层 4.76×（约 √L，随机噪声部分相消）。

**缺口**：`argmax 一致 100%` **不等于**任务精度不掉。PPL / MMLU 在 τ=1e-5 下
掉多少，**没测**。这决定该方案到底可不可用。

**附带发现**：`max|err| ≤ 1e-3` 这个门禁阈值比功能需求严得多——argmax 一致性
到 τ=3e-4 都是 100%。它是 harness 选的容差，不是数学必需。

### 4.4 深层泄漏的语料局限

见 §1.3 的语料诊断。75% 那个数字不能直接引用，需要一个 token 分布更均匀的语料
重测。稀有 token 的 49.4% 可引用。

---

## 5. 问题落在哪一层

这是理解整份清单的关键。**方案分三层，问题只出在第三层。**

| 层次 | 内容 | 状态 |
|---|---|---|
| **数学核心** | `ChainLinear` 的 `M⁻¹KM` 恒等式、SwiGLU 协变 `z' = zD_fP_f`、Down 补偿 `W_d' = P_fᵀD_f⁻¹W_d`、Value 路径 `Ac_V = [AV, Ae_V]M_V`、RMSNorm γ 吸收、RoPE 后共同正交变换 | ✅ **严格正确**。13 项 FP64 恒等式 + 303 测试。实现比方案文档更严（最大融合形态使明文注意力上下文 `O` 从不物化，强于文档 §21.2 只要求 `A` 不写回显存） |
| **工程实现** | 部署权重融合、结构化基、块对角 Gram、KV cache、debug/production 分离 | ✅ 正确。部署 block 的 GEMM 数量**与明文相同**，算术开销 +1.8%。3 条实现缺陷已修 |
| **隐私目标** | 「服务端不知道输入是什么」 | ❌ **未达成，且在方案自身约束下达不成** |

### 方案文档漏掉了什么

文档第三部分（统一数学模型）把注意力全部放在**线性层的协变性**上，§11 的推导是
完整的。非线性检查点那几章（§15、§21、§26、§29）处理的是「怎么让非线性层算对」，
方法是「在融合内核内提取合法表示」。

**它漏掉的是：这些非线性层需要的标量本身就是敏感信息，而且不能不算。**

| 非线性层 | 需要的标量 | 这个标量是什么 |
|---|---|---|
| RMSNorm | `rms(h)` | 就是 `‖h‖` |
| Softmax | 真实分数排序 | 就是 QK 几何 |
| Top-k | 真实间隔 `Δ_k` | 就是路由结构 |

文档的安全边界写的是「若服务器能修改 kernel 并主动导出寄存器中的 `A`，本方案
无法阻止」（§21.3）。它假设的威胁是**读取内部临时值**。

真实情况是：这些量可以从**必须存在的输出**里反推（§1.2），不需要读寄存器，
不需要改 kernel，不需要已知明文对，不需要知道 `M`。

**这是一个层次上的遗漏，不是实现偏差。**

### 形式化表述

以下四条不能同时成立：

```
C1  输出与明文模型逐位一致（或误差 ≤ 门禁容差）
C2  在线开销 ≤5%，客户端只做编解码
C3  无可信边界（不用 TEE / MPC / HE）
C4  服务端无法恢复输入 h
```

- C1 ⟹ `ρ` 必须算对 ⟹ `‖h‖` 泄漏（§1.1、§1.2）
- C2 ⟹ Gram 块必须在服务端 ⟹ `h` 的欧氏几何精确暴露（§1.4）
- C3 ⟹ 无法把 `ρ` 的计算移出观察范围
- 三者合起来 ⟹ ¬C4

每放弃一条，剩下三条可以成立。方案文档隐含假设四条同时成立。

---

## 6. 缓解方案对照

| 路线 | 消除程度 | 精度代价 | 需要改什么 | 实测状态 |
|---|---|---|---|---|
| **1. `ρ` 加有界噪声** | 恢复率 93.8% → 3.9% (τ=1e-5) | 放弃 exact 名分（max err 2.0e-3 > 1e-3 门禁），但 argmax 100% | 只改 `ρ` 的计算 | ✅ 已测（§4.3），**任务精度未测** |
| **2. 边云切分前 k 层** | 98.7% → 77.9%（稀有 49.4%），**不能归零** | 零 | 客户端承担 k 层计算，违反文档 §5 原则三 | ✅ 已测（§1.3） |
| **3. 范数均衡 embedding** | 第 0 层完全消除 | 零 | 改 embedding + 首层权重；只解决第 0 层 | ❌ 未实现 |
| **4. 换代数结构 (mod-q)** | 可能真正解决 | 未知 | 重写数学核心 | ❌ 与 C2 冲突（大整数模运算） |
| **5. TEE / MPC 包住 `c` 流** | 完全解决 | 零 | 引入可信边界 | ❌ 违反 C3，且削弱原始动机 |

**路线 1 + 2 组合**是目前唯一有实测支撑的可行配置：路线 2 处理第 0 层
（客户端算 embedding + 1~2 层，开销远小于 5%），路线 1 处理深层残余。

注意：路线 1 本质上是把文档 §29.4 的自适应阈值思路
（`τ = min(τ_max, αΔ_k/2)`，用有界扰动换隐私）从 Top-k 推广到 `ρ`。
**是在方案已有的机制上推广到它漏掉的地方，不是替换它。**

---

## 7. 我在分析过程中犯的错误

记录在案，因为它们影响过判断。

### 7.1 【已纠正】曾建议边云切分能消除泄漏

我先说「客户端算前 k 层能彻底消除 token 恢复攻击」。实测（§1.3）表明泄漏
**不随深度衰减**——第 28 层仍有 49.4%（稀有 token，基线 8.7%）。

原因是残差流保留 embedding 贡献，这一点我事前没想到。

### 7.2 【已纠正】缩放不变性测试用了过严容差

测「每 token 缩放 `α_t` 能否绕过泄漏」时，我用 `atol=1e-12` 判定
`RMSNorm(αh) == RMSNorm(h)`，得到 `False`，并据此说「分支对 α 不变不成立」。

实际是 `eps` 破坏严格不变性（最大差 1.85e-04），`eps=0` 时差 8.88e-16。
**「分支对 α 不变」实质上是成立的**，我那两行结论是测试 bug。

真正的结构性障碍在别处：残差分支不带 `α`，`α(h+F) ≠ αh+F`（最大差 3.82），
要修正服务端就必须知道 `α_t`，而那要么可算（攻击者也能算）、要么需要客户端
逐层参与（违反 C2）。

### 7.3 【已纠正】token 恢复率计算的分桶 bug

第一版用 `(n/prec).round().long()` 分桶，得到「平均候选 0.76 个」——不可能，
每个 token 至少在自己的桶里。浮点取整导致 key 查找不一致。
改用排序 + 二分查找窗口后数字单调自洽。

---

## 8. 全部复现命令

```bash
cd /Users/yin/code/fastProve

# 数学恒等式（13 项，独立于实现的 FP64 预言机）
python docs/design_identity_check.py

# 单元与正确性测试
python -m pytest -q                      # 303 passed

# 类别 A 泄漏
PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py \
    --model-path <checkpoint> --output results/raw/token_recovery_from_rho.json
PYTHONPATH=src python scripts/verify_rho_recoverable_from_fused_output.py \
    --output results/raw/rho_recoverable_from_fused_output.json
PYTHONPATH=src python scripts/verify_norm_leak_by_depth.py \
    --model-path <checkpoint> \
    --prompt-file results/raw/real_scenario_prompts_1500.jsonl \
    --output results/raw/norm_leak_by_depth.json
PYTHONPATH=src python scripts/verify_gram_leakage.py \
    --output results/raw/gram_leakage.json
PYTHONPATH=src python scripts/verify_recoverability_bound.py    # §1.5、§1.6

# 精度与开销
PYTHONPATH=src python scripts/record_exact_tolerance.py \
    --output results/raw/exact_tolerance_new.json
PYTHONPATH=src python scripts/measure_overhead.py --hidden-size 1024 \
    --output results/raw/overhead_new.json
```

**checkpoint 路径**：

```
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/deepseek-r1-distill-qwen-1.5b/base_model
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/llama-3.2-3b-instruct/base_model
```

⚠️ **不要**使用同目录下的 `obfuscated_full/` 或 `hf_upload/`——那是 legacy
ModelSplit 系统的产物，不是明文基座。`pretrained/llama.py` 的 `_LEGACY_MARKERS`
会正确拦截它们。

---

## 9. 原始记录索引

| 文件 | 内容 |
|---|---|
| `results/raw/token_recovery_from_rho.json` | §1.1 Qwen2 的 token 恢复率 |
| `results/raw/token_recovery_from_rho_llama.json` | §1.1 Llama 的 token 恢复率 |
| `results/raw/rho_recoverable_from_fused_output.json` | §1.2 融合内核失效 |
| `results/raw/norm_leak_by_depth.json` | §1.3 深度衰减 + 语料诊断 |
| `results/raw/gram_leakage.json` | §1.4 Gram 几何暴露 |
| `results/raw/recoverability_bound.json` | §1.5、§1.6 可恢复性上界 |
| `results/raw/exact_tolerance_{A4,A4_before,A5,B2,B3}.json` | 各阶段 exact 容差 |
| `results/raw/overhead_B7_d{512,1024,2048}.json` | eager 参考实现开销 |
| `results/raw/exact_gate_D1.json` | FP32 硬门禁（全 PASS） |
| `results/raw/eval_sweep.jsonl` | 72 点噪声 sweep（dtype label 有 §3.2 的错） |
| `results/raw/eval_sweep_dtype_fixed.jsonl` | 同上，dtype 已修正 |
| `docs/threat_model.md` §5bis.1–5bis.9 | 全部泄漏通道的正式记录 |

---

## 10. 不得声明的内容

`AGENTS.md` 的约束，逐条都适用于本清单涉及的结论：

```
✗ activations 是标准 LWE 密文
✗ 全同态 / 端到端加密推理
✗ 因为用了 ML-KEM / LWE-derived KDF 就有 LWE 安全性
✗ 能防御可任意修改 kernel、dump 寄存器的服务器
✗ 任何没有 recorded run 支撑的性能 / 精度 / 隐私数字
✗ 用 eager 参考实现的速度代表最终方案
```

允许的措辞：

```
✓ augmented covariant obfuscation
✓ chained auxiliary noise
✓ bounded logit perturbation
✓ approximate privacy-preserving inference prototype
✓ exact mode / approximate mode
✓ 「在 C1–C3 约束下隐藏度上界为零」（有 §1–§5 的工件支撑）
```

**特别注意**：§4.3 那条加噪缓解在补测任务精度之前，**不得**声称
「τ=1e-5 是可用操作点」。argmax 一致不等于任务精度不掉。
