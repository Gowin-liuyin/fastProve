# fastProve 启动提示词

将下面整段提示词交给编码代理，并把工作目录设置为：

```text
/Users/yin/code/fastProve
```

---

你现在负责从零完成 `fastProve` 原型项目。请在 `/Users/yin/code/fastProve` 内持续工作，直到形成可以运行、可以测试、可以比较明文精度下降的完整原型。

开始前必须完整阅读项目根目录的 `AGENTS.md`。其中的数学定义、威胁模型、实现阶段、实验指标和验收标准都是本任务的强制约束。不要自行弱化或跳过。

## 核心目标

实现并公平比较以下四种推理模式：

1. `plaintext`：原始明文 Transformer 基线；
2. `exact`：零 Softmax 噪声的精确协变混淆模式；
3. `approximate/topk_preserving`：根据 Attention margin 限制噪声，尽量保持重要 Top-k；
4. `approximate/free_bounded`：使用固定误差预算，允许部分排序改变。

最终必须回答：

- 精确模式是否只产生浮点级误差；
- 不同 Softmax 噪声强度下，困惑度、next-token accuracy、生成 token 一致率分别下降多少；
- Top-k 保持策略与自由有界噪声策略的差异；
- 哪个噪声配置是当前实验中的最佳精度/扰动折中点；
- 哪些安全结论不能由本原型推出。

## 执行方式

不要只输出设计建议或伪代码。请直接创建文件、实现代码、运行测试、保存原始结果并生成报告。

在开始时：

1. 检查目录和已有文件，保护用户已有修改；
2. 检查 Python、PyTorch、CUDA、MPS、CPU、内存和磁盘环境；
3. 检查本机是否已有可用的小型 Llama-like 预训练模型和评测数据；
4. 给出简短工作计划，然后立即执行；
5. 将环境审计写入 `docs/implementation_notes.md`。

除非出现真正需要用户授权的外部操作，例如下载大型模型、下载大型数据集或使用远程 GPU，否则不要因为非关键选择停下来提问。对类名、目录名、默认随机种子等低风险问题直接采用合理默认值。

## 第一阶段：建立可验证的数学内核

按照 `AGENTS.md` 创建项目结构、`pyproject.toml`、配置系统和测试框架。

优先实现：

- `MixedState`；
- 可复现的随机种子与域分离；
- 条件数受控的混合矩阵生成；
- debug-only encode/decode；
- `ChainLinear`；
- affine noise refresh；
- 数学布局与 PyTorch 权重布局转换；
- 小张量单元测试。

必须实际验证：

\[
c_{\rm out}
=
c_{\rm in}\widetilde W+\widetilde b
=
[hW+b,\;hC+eG+\xi]M_{\rm out}.
\]

不允许在 `forward` 中求逆。

如果该等式的测试失败，先定位数学、布局、转置或数值条件问题，不要继续组装 Transformer。

## 第二阶段：明文 Block 与精确混淆 Block

使用完全相同的基础权重，实现一个小型 Decoder Block：

```text
RMSNorm
→ Q/K/V
→ RoPE
→ causal attention
→ output projection
→ residual
→ RMSNorm
→ SwiGLU
→ down projection
→ residual
```

要求：

- RMSNorm 的 gamma 必须按 `AGENTS.md` 吸收到后续投影；
- 先应用 RoPE，再对 Q/K 应用共同正交变换；
- Q/K 点积必须与明文路径一致；
- Value 使用增广混合状态；
- SwiGLU 使用 Gate/Up 同步置换和 Down 权重补偿；
- 噪声经过非线性检查点时走旁路并在之后刷新，不能永久清零；
- causal mask、padding mask、全 mask 边界和 tie 情况必须测试；
- debug 解码只能存在于测试/调试路径。

先用 FP32 完成正确性，再增加 BF16 可选测试。

## 第三阶段：实现近似 Softmax

在同一 Attention 接口下实现：

```text
plaintext
exact
approximate/topk_preserving
approximate/free_bounded
```

实现要求：

- logits、margin、noise clipping 和 Softmax reduction 使用 FP32；
- 噪声只加在有效 logits 上；
- masked position 必须保持 `-inf`；
- 噪声由记录的 seed 确定性生成；
- 实际噪声必须满足配置的 infinity-norm 上界；
- `topk_preserving` 使用：

\[
\Delta_k=S_{(k)}-S_{(k+1)},
\]

\[
\tau=
\min\left(
\tau_{\max},
\frac{\alpha\Delta_k}{2},
\tau_{\rm error}
\right);
\]

- `free_bounded` 不使用 margin 上限，允许排序改变；
- 生产接口不得返回 Attention probability；
- 测试代码可以捕获 probability，用于比较 KL、JS、Top-k overlap 和 rank correlation。

## 第四阶段：组装 Tiny Causal LM

先创建本地 tiny model 完成端到端正确性测试。

要求：

- 明文模型和各混淆模式共享相同基础权重；
- 相同 token 输入、相同 mask、相同 seed；
- 支持 teacher-forced next-token evaluation；
- 支持 greedy generation；
- 精确模式出现 greedy token 不一致时，必须先调查，不能直接进入噪声结论。

随机 tiny model 只能证明正确性，不能作为有意义的语言模型精度证据。

## 第五阶段：真实小模型精度评测

优先使用本机已有的小型 Llama-like 预训练 checkpoint 和本地已有的公开评测数据。

如果没有：

- 完成所有无需下载的实现和正确性测试；
- 列出建议使用的最小模型、数据规模、磁盘和显存需求；
- 在下载前向用户请求一次明确许可；
- 不得把随机模型结果描述为真实语言模型精度。

明文和混淆模式必须使用完全相同的：

- checkpoint；
- 数据样本及顺序；
- tokenizer；
- sequence length；
- dtype；
- device；
- batch size；
- generation 参数。

缓存评测样本 ID 或 tokenized input，保证所有模式严格对齐。

## 第六阶段：噪声消融

至少扫描：

```text
tau_max = [0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.10]
alpha = [0.50, 0.80, 0.95]
preserve_top_k = [4, 8, 16]
```

如果组合数量对当前算力过大：

1. 先运行小规模校准集；
2. 根据校准结果筛选候选配置；
3. 再在完整评测集上运行；
4. 在报告中明确筛选规则，不得只保留有利结果。

每次运行保存独立的 JSON/JSONL 记录，包括：

- 配置；
- seed；
- 模型和数据标识；
- 软件与设备环境；
- 样本数；
- 成功或失败状态；
- 所有指标；
- 总耗时。

## 必须计算的指标

层级指标：

- max/mean absolute error；
- relative L2 error；
- cosine similarity；
- QK score error；
- Softmax KL divergence；
- Jensen-Shannon divergence；
- Top-k set overlap；
- rank correlation；
- Attention output relative error；
- NaN/Inf count。

端到端指标：

- negative log-likelihood；
- perplexity；
- next-token top-1 accuracy；
- next-token top-5 accuracy；
- plaintext/obfuscated token agreement；
- greedy token exact match；
- sequence exact match（适用时）。

性能指标：

- conversion time；
- prefill latency；
- decode latency/TPOT；
- tokens/s；
- peak memory；
- KV-cache memory。

报告精度下降时必须同时给出：

- 明文绝对值；
- 混淆绝对值；
- 绝对下降；
- 相对下降。

## 输出工件

最终至少生成：

```text
README.md
docs/mathematics.md
docs/threat_model.md
docs/implementation_notes.md
configs/tiny_exact.yaml
configs/tiny_approx.yaml
configs/eval_sweep.yaml
src/fastprove/...
tests/...
scripts/run_correctness.py
scripts/run_accuracy_sweep.py
scripts/build_report.py
results/raw/...
results/tables/...
results/figures/...
results/REPORT.md
```

`results/REPORT.md` 必须包括：

1. 实验目标；
2. 数学实现摘要；
3. 环境、模型和数据；
4. 精确模式正确性；
5. 噪声强度—精度曲线；
6. Softmax 排序与 Top-k 变化；
7. 最佳折中配置；
8. 性能开销；
9. 威胁模型和安全限制；
10. 完整复现命令。

## 质量和诚实性要求

- 不得伪造实验结果；
- 未执行的实验必须明确标记；
- 不得把“没有把概率返回给 Python”写成密码学安全；
- 不得把 LWE/ML-KEM 密钥派生写成整个系统具有 LWE 安全性；
- 不得用随机 tiny model 声称真实语言能力保持；
- 不得只报告最佳配置而隐藏失败点；
- 每个结论必须能追溯到原始结果文件；
- 代码通过导入或编译不代表任务完成；
- 测试失败时保留证据并定位原因。

请持续执行，直到：

- 四种模式均已实现；
- 精确模式测试通过；
- 噪声扫描完成，或因明确的模型/数据/算力外部条件而无法继续；
- 原始结果、汇总表、图和报告已经生成；
- 最终回复列出修改文件、运行命令、真实结果、失败项和下一步。

不要在只完成脚手架时宣布完成。
