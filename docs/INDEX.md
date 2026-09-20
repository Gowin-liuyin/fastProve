# fastProve 文档索引

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/CLEANUP_PLAN.md §8
适用范围：仓库全部 Markdown 文档的唯一入口：生命周期状态、归档清单与阅读顺序

本索引由 P8a（2026-09-19）建立：每份现行文档顶部都带
`状态 / 日期 / 上游依据 / 适用范围` 四行生命周期头；历史文档已移入
`docs/archive/`（移动不删除），并在文首带 HISTORICAL banner。

---

## 1. 现行文档

状态取值：`CURRENT`、`CONTRACT-未实现`、`HISTORICAL-已归档`。带括号的说明是
对状态的限定，不改变状态本身。

| 文档 | 状态 | 日期 | 用途 |
|---|---|---|---|
| `AGENTS.md` | CURRENT（上游规则，中文译本） | 2026-09-19 | 项目使命、数学规范、威胁模型、验收门的最高约束 |
| `README.md` | CURRENT | 2026-09-20 | 项目入口：诚实状态、真实模块树、执行命令与限制摘要 |
| `docs/INDEX.md` | CURRENT | 2026-09-19 | 本文档：文档唯一入口与归档清单 |
| `docs/CLEANUP_PLAN.md` | CURRENT（草案，待决策项见 §10） | 2026-09-19 | 代码结构梳理与文档重建计划；P8a/P8b 已执行，P9 待做 |
| `docs/architecture.md` | CURRENT | 2026-09-20 | 生产路径 / 参考路径 / 侧路实验三分表、as-built 模块地图与依赖方向 |
| `docs/config_reference.md` | CURRENT | 2026-09-20 | 四个 YAML 的键语义、取值域与校验来源（P8b 新建） |
| `docs/scripts_index.md` | CURRENT | 2026-09-20 | `scripts/` 29 个脚本与归档脚本的用途、调用、产物、引用（P8b 新建，P8/P9 后续更新） |
| `results/PROVENANCE.md` | CURRENT | 2026-09-20 | claim → raw 工件 → 生成命令映射，标注 current/superseded（P8b 新建） |
| `docs/runbook.md` | CURRENT | 2026-09-20 | 一页式验证阶梯与阻塞步骤（P8b 新建） |
| `docs/STATUS.md` | CURRENT | 2026-09-20 | P0–P7 重构进度、未决问题、阻塞与下一步（P8b 新建） |
| `docs/IMPLEMENTATION_PROMPT_dual_channel.md` | CURRENT（S0–S5 已执行） | 2026-09-19 | 双通道噪声预算重构提示词与阶段计划；文末有「S1 之后的增量」 |
| `docs/technical_specification.md` | CURRENT | 2026-09-16 | 系统技术架构与实现规格（v2.0） |
| `docs/PROBLEMS.md` | CURRENT | 2026-09-19 | 全部已知问题：A 构造泄漏 / B 实现缺陷 / C 证据缺口 |
| `docs/threat_model.md` | CURRENT | 2026-09-16 | 威胁模型与安全限制的正式记录 |
| `docs/work_factor_security.md` | CURRENT | 2026-09-16 | 攻击工作因子与成本曲线合同 |
| `docs/mathematics.md` | CURRENT | 2026-09-20 | as-built 数学规格：当前构造逐条对应 `file.py:line` |
| `docs/augmented_noisy_obfuscation_plan.md` | CURRENT（阶段 A 已执行；B/D 未开始） | 2026-09-14 | 增广带噪核实验方案（侧路 A 套实现） |
| `docs/implementation_notes.md` | CURRENT | 2026-09-19 | 环境审计与分阶段实现记录（进度接续见 `docs/STATUS.md`） |
| `docs/literature-review/LITERATURE_INDEX.md` | CURRENT | 2026-09-19 | 文献清单（PDF 已移出跟踪）与精读笔记索引 |
| `docs/literature-review/reports/2026-09-09_five_papers_inspiration.md` | CURRENT | 2026-09-09 | 五篇论文对 fastProve 的启发分析 |
| `docs/literature-review/reports/2026-09-09_nine_obfuscation_defenses.md` | CURRENT | 2026-09-09 | 九种混淆式 LLM 推理防御的机制对照 |
| `docs/literature-review/reports/2026-09-09_session_memory.md` | CURRENT | 2026-09-09 | 文献精读会话记录 |
| `docs/literature-review/reports/evaluation_protocol_fastProve.md` | CURRENT | 2026-08-01 | 实验验收的权威评测方案 |
| `docs/literature-review/reports/plaintext_vs_obfuscated_accuracy_survey.md` | CURRENT（第 8 节通用骨架已被上述评测方案取代） | 2026-08-01 | 明文/混淆精度文献综述 |
| `evals/README.md` | CURRENT | 2026-09-19 | `evals/` 第二套评测 harness 的使用说明与数据口径 |
| `results/REPORT.md` | CURRENT | 2026-09-20 | 阶段 A–D 实测结果报告（随机 tiny 正确性证据）；§2.3 ρ=0 重跑说明与 §6.1 replica 边际代价为 2026-09-20 增补 |
| `results/raw/README.md` | CURRENT | 2026-09-20 | `results/raw/` 实际目录盘点与逐文件溯源 |
| `results/figures/README.md` | CURRENT | 2026-09-20 | `results/figures/` 已跟踪图形清单与口径警告 |
| `results/tables/README.md` | CURRENT | 2026-09-19 | `results/tables/` 的生成规则 |
| `scripts/archive/README.md` | CURRENT | 2026-09-19 | `scripts/archive/` 中归档报告脚本的冻结原因 |

---

## 2. 归档文档（`docs/archive/`）

归档 = 移动不删除，保留研究记录；这些文件不再作为数学、工程或验收依据。
文件名格式为 `<日期>_<原名>.md`，文首均有 HISTORICAL banner。完整清单与
归档原因见 [`archive/README.md`](archive/README.md)。

| 归档文档 | 日期 | 原因 |
|---|---|---|
| `docs/archive/2026-07-31_fastprove-prototype.md` | 2026-07-31 | 初版 TDD 实施计划，原型已完成、tests/ 已重组 |
| `docs/archive/2026-08-06_REPORT_zh.md` | 2026-08-06 | 自标 SUPERSEDED 的中文报告，数字来自有缺陷 harness |
| `docs/archive/2026-09-13_problem_analysis.md` | 2026-09-13 | 早期问题分析，正式记录以 `docs/PROBLEMS.md` 为准 |
| `docs/archive/2026-09-14_augmented_noise_stage_a_review.md` | 2026-09-14 | 阶段 A 复核记录，缺陷已修复或被后续交接覆盖 |
| `docs/archive/2026-09-14_scheme_vs_impl_verdict.md` | 2026-09-14 | 早期方案/实现判定，正式口径以 PROBLEMS/threat_model 为准 |
| `docs/archive/2026-09-15_AUGMENTED_NOISE_HANDOFF_20260915.md` | 2026-09-15 | 2026-09-15 交接快照，后续 S0–S5、P0–P7 已推进 |
| `docs/archive/2026-09-16_DERIVATION_PACKAGE.md` | 2026-09-16 | 旧框架（expanded_concat/expanded_full），禁止复活 |
| `docs/archive/2026-09-16_PROOF_PACKAGE.md` | 2026-09-16 | 旧框架的数学证明包，禁止复活 |
| `docs/archive/2026-09-16_aloepri_augmented_noise_architecture.md` | 2026-09-16 | 旧框架 AloePri 扩维方案，禁止复活 |
| `docs/archive/2026-09-16_augmented_covariant_implementation_v2.md` | 2026-09-16 | 旧框架工程实施合同，禁止复活 |
| `docs/archive/2026-09-19_START_PROMPT.md` | 2026-09-19 | 项目启动提示词，已被 AGENTS.md 与当前计划取代 |
| `docs/archive/2026-09-19_IMPLEMENTATION_PLAN.md` | 2026-09-19 | 阶段 A–D 执行计划，REPORT 状态判断与今日事实相反 |
| `docs/archive/2026-09-19_experiment_plan.md` | 2026-09-19 | 旧实验执行方案，现行口径以 README/评测方案为准 |
| `docs/archive/2026-09-19_mathematics_july_construction.md` | 2026-09-19 | 2026-07 构造时代的数学说明；已由 as-built 版 `docs/mathematics.md` 取代 |

---

## 3. 先读这个（推荐顺序）

1. `AGENTS.md` —— 规则与验收门，高于一切文档；
2. `docs/INDEX.md` —— 本文档，确认每份文档的状态与新鲜度；
3. `docs/architecture.md` —— 哪些代码是生产路径、参考路径、侧路实验；
4. `README.md` —— 怎么装、怎么跑、结果口径；
5. `docs/runbook.md` —— 一页式验证阶梯与当前阻塞步骤；
6. `docs/STATUS.md` —— P0–P7 重构进度、未决问题与下一步。

需要配置或复现时再读：`docs/config_reference.md`（YAML 键）、`docs/scripts_index.md`
（脚本与产物）、`results/PROVENANCE.md`（claim→工件→命令）。

从事当前研究，再按需读 `docs/IMPLEMENTATION_PROMPT_dual_channel.md`（双通道设计与
阶段计划）、`docs/PROBLEMS.md`（已知问题与不得声明的内容）、
`docs/threat_model.md`、`docs/work_factor_security.md`。

---

## 4. 已知失效引用与孤儿配置（P8a 登记，未改代码）

- `scripts/run_augmented_noise_kernel.py`（模块 docstring 第 12-13 行）指向不存在的
  `scripts.postprocess_augmented_noise_kernel`；该缺陷最初记录于
  `docs/archive/2026-09-14_augmented_noise_stage_a_review.md`。按 P8a 约束不改代码，
  在此登记：CSV 派生目前由 `scripts/build_report.py` 一类的等价 pass 承担。
- `configs/augmented_noise_kernel.yaml` 是**孤儿配置**：没有 `.py` 读取它，
  `scripts/run_augmented_noise_kernel.py` 在代码内构造 `KernelConfig`。
  本文件仅作文档保留（documentation-only），接线属功能变更，另立项。
  不要把它当作已生效的实验配置。
- `configs/eval_sweep.yaml` 的 `execution.run_in_current_prototype_pass: false` 是过期
  门标志；`base_config: configs/tiny_approx.yaml` 会让 sweep 的 `plaintext`/`exact`
  点继承 `rho_noise_bound: 1e-5` 与 `per_request`。详见 `docs/PROBLEMS.md` §3.5
  （含 72 条 spec 的实测继承表与复现命令）。本次不修改任何 config。
- 4 个从未实现的脚本引用已在原文处加注：`verify_token_recovery_candidate_rescoring.py`
  与 `probe_embedding_norm_equalisation.py`（`docs/IMPLEMENTATION_PROMPT_dual_channel.md`
  S6/S7），`run_augmented_coa_eval.py`（`docs/augmented_noisy_obfuscation_plan.md`），
  `verify_aloepri_subspace_attack.py`（`docs/archive/2026-09-16_aloepri_augmented_noise_architecture.md`）。
- `docs/noise_placement_explained.pdf`/`.tex` 与 `docs/verification/`、
  `docs/handoff_evidence/` 属于推导与核验材料，不属于 Markdown 文档生命周期标注范围。

---

## 5. 计划中文档的落地状态（2026-09-20）

`docs/CLEANUP_PLAN.md` §8.2–§8.3 计划的三份文档已由 P8b 创建：

- [`docs/config_reference.md`](config_reference.md)；
- [`docs/scripts_index.md`](scripts_index.md)；
- [`results/PROVENANCE.md`](../results/PROVENANCE.md)。

本索引第 1 节已登记它们。P8 剩余未落地项：`docs/STATUS.md` 已在 P8b 建立；
`docs/runbook.md`、`results/raw/README.md`、`results/figures/README.md` 已重写。

---

## 6. 维护规则

- 新增现行文档：在文件顶部加生命周期头，并登记到本文档第 1 节；
- 文档失去效力：`git mv` 到 `docs/archive/<日期>_<原名>.md`，文首加
  `> HISTORICAL-已归档：<原因>；当前口径见 docs/INDEX.md。`，并更新
  `docs/archive/README.md` 与本文档第 2 节；
- 旧框架文档（`expanded_concat`/`expanded_full`/`P̂`/`a`/`B_comp`）归档时额外加
  `> 旧框架，禁止复活：生产路径是 D = d；不得引用 expanded_concat/expanded_full/P̂/a/B_comp。`
- 文档内相对链接用 `python scripts/check_doc_links.py` 检查（纯文档工具，不参与
  等价性门禁）。
