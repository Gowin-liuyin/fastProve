# docs/archive 归档清单

状态：CURRENT
日期：2026-09-19
上游依据：docs/INDEX.md、docs/CLEANUP_PLAN.md §8.1
适用范围：`docs/archive/` 内历史文档的列表、归档日期与原因

本目录由 P8a（2026-09-19）建立。**归档 = 移动不删除**：这些文件是研究记录，
保留在仓库中以供追溯，但不再作为数学、工程或验收依据。每份文件顶部都有：

- 一般历史文档：`> HISTORICAL-已归档：<原因>；当前口径见 docs/INDEX.md。`
- 旧框架文档（`expanded_concat`/`expanded_full`/`P̂`/`a`/`B_comp`）：
  `> 旧框架，禁止复活：生产路径是 D = d；不得引用 expanded_concat/expanded_full/P̂/a/B_comp。`

文件名格式：`<日期>_<原名>.md`。现行文档清单与阅读顺序见
[`docs/INDEX.md`](../INDEX.md)。

| 归档文件 | 日期 | 归档原因 |
|---|---|---|
| `2026-07-31_fastprove-prototype.md`（原 docs/superpowers/plans/2026-07-31-fastprove-prototype.md，已移动） | 2026-07-31 | 初版 TDD 实施计划；原型早已完成，测试目录已按 P7 重组 |
| `2026-08-06_REPORT_zh.md`（原 results/REPORT_zh.md，已移动） | 2026-08-06 | 自标 SUPERSEDED 的中文评测报告；数字来自有缺陷 harness，不得引用 |
| `2026-09-13_problem_analysis.md`（原 docs/problem_analysis.md，已移动） | 2026-09-13 | 早期问题分析；正式问题记录以 `docs/PROBLEMS.md` 为准 |
| `2026-09-14_augmented_noise_stage_a_review.md`（原 docs/augmented_noise_stage_a_review.md，已移动） | 2026-09-14 | 阶段 A 复核记录；所述缺陷已修复或被后续交接覆盖 |
| `2026-09-14_scheme_vs_impl_verdict.md`（原 docs/scheme_vs_impl_verdict.md，已移动） | 2026-09-14 | 早期方案/实现判定；正式口径以 `docs/PROBLEMS.md` 与 `docs/threat_model.md` 为准 |
| `2026-09-15_AUGMENTED_NOISE_HANDOFF_20260915.md`（原 docs/AUGMENTED_NOISE_HANDOFF_20260915.md，已移动） | 2026-09-15 | 2026-09-15 交接快照；后续 S0–S5、P0–P7 已推进 |
| `2026-09-16_DERIVATION_PACKAGE.md`（原仓库根 DERIVATION_PACKAGE.md，已移动） | 2026-09-16 | 旧框架方案（`expanded_concat`/`expanded_full`），禁止复活 |
| `2026-09-16_PROOF_PACKAGE.md`（原仓库根 PROOF_PACKAGE.md，已移动） | 2026-09-16 | 旧框架的数学证明包，禁止复活 |
| `2026-09-16_aloepri_augmented_noise_architecture.md`（原 docs/aloepri_augmented_noise_architecture.md，已移动） | 2026-09-16 | 旧框架 AloePri 扩维方案，禁止复活 |
| `2026-09-16_augmented_covariant_implementation_v2.md`（原 docs/augmented_covariant_implementation_v2.md，已移动） | 2026-09-16 | 旧框架工程实施合同，禁止复活 |
| `2026-09-19_START_PROMPT.md`（原仓库根 START_PROMPT.md，已移动） | 2026-09-19 | 项目启动提示词；已被 `AGENTS.md` 与当前双通道计划取代 |
| `2026-09-19_IMPLEMENTATION_PLAN.md`（原 docs/IMPLEMENTATION_PLAN.md，已移动） | 2026-09-19 | 阶段 A–D 执行计划；其对 `results/REPORT.md` 的 SUPERSEDED/FAIL 判断与今日事实相反 |
| `2026-09-19_experiment_plan.md`（原 docs/experiment_plan.md，已移动） | 2026-09-19 | 旧实验执行方案；现行口径以 `README.md`、`docs/literature-review/reports/evaluation_protocol_fastProve.md` 为准 |
| `2026-09-19_mathematics_july_construction.md`（原 docs/mathematics.md，已移动） | 2026-09-19 | 2026-07 构造时代的数学说明；已由 as-built 版 `docs/mathematics.md`（2026-09-20）取代 |

## 使用规则

1. 不要从归档文档复制命令、数字或声明进入报告；现行口径见 `docs/INDEX.md`。
2. 归档文档内的相对链接已按新位置修正；指向其他归档文档的同目录文件即其
   归档名，指向现行文档用 `../`。
3. 旧框架文档中出现的 `expanded_concat`、`expanded_full`、`P̂`、`a`、`B_comp`
   在 `src/` 中不存在，生产路径是 `D = d`；不得据此恢复实现。
4. 如需引用归档材料，必须同时说明它已归档及其日期。
