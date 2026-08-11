# Llama-3.2-3B-Instruct 明文 vs 当前 fastProve 混淆 — 中文评测报告

> ## ⚠️ 本报告数字已作废（SUPERSEDED，2026-08-06）
>
> 本报告的全部任务精度与 retention 数字由存在缺陷的评测 harness 产生，
> **不得对外引用、不得写入论文或组会材料**。已定位的 disqualifying 缺陷：
>
> 1. **样本选择为语料前缀截断**，而多个本机语料并非乱序：
>    `mmlu_test.jsonl` 按 subject 排序（前 200 行只覆盖 57 个科目中的
>    `abstract_algebra` 与 `anatomy` 两个），`anli_r3_test.jsonl` 按 label
>    排序（前 200 行只有 6 个游程，随机应约 134）。
> 2. **PIQA gold label 退化**：`piqa_val.jsonl` 全部 1838 行 label 均为 0。
>    其"准确率"实为模型选中 sol1 的比例，且在 plain/structural/full 下
>    恒为 92/200，向平均值贡献了一个硬编码的 100% retention。
> 3. **多选打分未做长度归一化**：按 logprob 求和比较候选，系统性偏向
>    短候选，故 PIQA/HellaSwag/ARC 与公开数字不可比。
> 4. **明文基线未与公开参考值对账**：本仓库明文 avg 0.493，而 OSNIP 论文
>    对同一 Llama-3.2-3B-Instruct 的 non-private 参考为 0.597（差 10.4pp）。
>    基线本身失准时，其上计算的 retention 不承载效用保持的信息。
>
> 修复已落入 `scripts/run_osnip_style_benchmarks.py`（seeded 采样 + MMLU
> 按 subject 分层 + 退化标签拒绝 + `acc_norm` + 置信区间/chance floor 标注）。
> **重跑并与公开参考值对账后**，方可产生新的可引用报告。
>
> 另见 [`docs/threat_model.md`](../docs/threat_model.md) 第 5bis 节：本报告
> 中的任何精度或 retention 数字**都不是隐私证据**。

> 本报告仅比较**明文**与**本仓库当前协变混淆方案**（structural / full）。
> **不包含**旧版 ModelSplit 混淆对照。
> 术语：混淆态相对明文（LWE-keyed / LWE-inspired 低开销协变混淆），不是标准 LWE 密文，也不是全同态加密推理。

## 1. 测试概览

| 项目 | 内容 |
|---|---|
| 明文基座 | `/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct` |
| 明文校验 | 通过 |
| weights_sha256 | `963e552937a68f2166e65810a13ccc5b9d3a173517ff58ed9ebd2b36a5fe557e` |
| config_sha256 | `39fb36dc5416f445ebc4e71cb71fbcf6727e80a35836d8ba1a1474c318467b7a` |
| 每 key 的 Prompt 数 | **1500** |
| 独立 master key 数 | **3** |
| 模式 | structural, full |
| 序列长度 / 生成长度 | 64 / 4 |
| dtype / device | bfloat16 / cuda |
| 真实场景 Prompt | 是 |
| 种子 | 20260802 |
| GPU | NVIDIA GeForce RTX 5090 |
| torch | 2.12.0+cu130 |
| 运行状态 | complete |
| 原始结果 | `results/raw/llama_compare_large.json`（见下文路径） |

### 数据量说明

- 每个 master key 使用**同一组** 1500 条真实场景 Prompt（中英混合：知识问答、指令跟随、代码、推理、办公邮件、摘要、翻译等）。
- 模式 × 密钥单元数：6（应等于 模式数 × key 数）。
- Teacher-forced 有效 next-token 决策数（单 key 示例，取 pairs 中最大值）： **48417**。
- Greedy：默认对全部 Prompt 去 padding 后生成对比（`greedy_n_samples` 应等于 1500）。

## 2. 精度差距总表（相对明文）

指标说明：**一致率**为混淆与明文 next-token argmax 相同的比例（越高越好）；**greedy 整段一致率**为生成后缀完全相同的样本比例；**ΔPPL_rel** = PPL_obf/PPL_plain − 1（相对增幅，越接近 0 越好）；**top1_drop_pp** 为相对标签的准确率百分点差（可与「相对明文一致率」区分）。

| 模式 | key | top1 一致率 | greedy 整段一致 | greedy n | ΔPPL_rel | e2e logit max |
|---|---|---:|---:|---:|---:|---:|
| structural | key-00 | 0.9707 | 0.8553 | 1500 | 0.0061 | 4.8281 |
| structural | key-01 | 0.9741 | 0.8473 | 1500 | 0.0013 | 4.7969 |
| structural | key-02 | 0.9431 | 0.8327 | 1500 | -0.0069 | 7.6562 |
| full | key-00 | 0.9069 | 0.6047 | 1500 | 0.0492 | 15.5625 |
| full | key-01 | 0.9036 | 0.6393 | 1500 | 0.0497 | 16.7852 |
| full | key-02 | 0.8711 | 0.6500 | 1500 | 0.0748 | 10.5273 |

### 跨 key 平均

| 模式 | 平均 top1 一致率 | 平均 greedy 整段一致 | 平均 ΔPPL_rel |
|---|---:|---:|---:|
| structural | 0.9626 | 0.8451 | 0.0002 |
| full | 0.8939 | 0.6313 | 0.0579 |

- **Structural 相对明文 token 分歧率** ≈ **3.74%**（1 − 一致率）。
- **Full 相对明文 token 分歧率** ≈ **10.61%**。

## 3. 门禁与数值诊断

- **ChainLinear 硬门禁**仅使用 FP32 单元恒等式，**不是**端到端 BF16 logit 误差。
- 端到端 logit max|obf−plain|（诊断）：**15.5625**
- ChainLinear 单元 max abs error：**0.000001**
- hard_passed：**False**；soft_passed：**False**

```
FP32 ChainLinear max abs error | PASS | observed=8.940696716308594e-07 threshold=0.0001
Attention rank-flip rate | PASS | observed=None threshold=0.0
Attention top-1 match | PASS | observed=None threshold=1.0
Attention top-k overlap | PASS | observed=None threshold=1.0
MoE expert set match | PASS | observed=None threshold=1.0
Causal mask match | PASS | observed=None threshold=1.0
Cache vs no-cache identical | PASS | observed=None threshold=1.0
LM Head inverse-permuted argmax | FAIL | observed=0.9069128611851209 threshold=1.0
Greedy release-validation sequences | FAIL | observed=0.6046666666666667 threshold=1.0
Accuracy drop ≤ 0.5 pp | FAIL | observed=0.7497366627424229 threshold=0.5
PPL relative increase ≤ 1% | FAIL | observed=0.04920313008082933 threshold=0.01
```

## 4. 结论（面向工程）

1. 大规模真实场景下，当前 fastProve 相对明文存在可测量的 next-token 不一致；structural（无噪声注入）通常优于 full（完整噪声链）。
2. 若 greedy 整段一致率显著低于 teacher-forced 一致率，说明自回归会放大早期分歧。
3. 硬门禁（argmax/greedy 100% 等）未通过时，**不得**宣称「零精度损失」或「已达发布级数学等价」。
4. 本结果**不能**外推为 MMLU/C-Eval 等任务榜单；亦**不能**推出密码学安全保证。

## 5. 威胁模型与非声明（AGENTS.md）

- 原型针对诚实但好奇的观察者（持久化张量/常规框架输出），假定指定融合算子不回传内部干净临时量——这是实现假设，不是密码学保证。
- 不声称：标准 LWE 密文、全同态/端到端加密推理、仅因 ML-KEM/LWE 风格种子即具备 LWE 安全性、抵御可任意改内核/挂钩/转储寄存器的服务器。

## 6. 原始产物路径

- 机器可读结果：见本报告生成时输入的 JSON 路径（`results/raw/llama_compare_large.json`）
- 英文/混合摘要：`results/REPORT.md`（若存在）
- 本中文报告：`results/REPORT_zh.md`

---

*报告由 `scripts/write_chinese_report.py` 从原始 JSON 自动生成，数字可追溯。*
