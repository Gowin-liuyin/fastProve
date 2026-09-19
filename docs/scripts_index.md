# 脚本索引（scripts/ 与 scripts/archive/）

状态：CURRENT
日期：2026-09-20
上游依据：docs/CLEANUP_PLAN.md §P5、§8.3；docs/implementation_notes.md
适用范围：`scripts/` 全部 28 个脚本与 `scripts/archive/` 2 个归档脚本的用途、类别、调用、产物与引用

所有「典型调用」中的 `--help` 形式都在 2026-09-20 实测（26 个现行脚本 + 2 个归档脚本
均接受 `-h/--help`）。带参数示例取自 `README.md`、`docs/runbook.md`、`results/raw/README.md`
或脚本 docstring；无法考证确切 argv 的标「待考」。脚本名在本次清理中**未改动**
（CLEANUP_PLAN §4-⑧：`evaluation/report.py` 内嵌三条命令名，重命名会静默改报告正文）。

## 0. 自包含单文件约定（P5）

`docs/IMPLEMENTATION_PROMPT_dual_channel.md:81` 规定攻击/探针脚本是**自包含单文件、无共享
util**；`docs/CLEANUP_PLAN.md:180` 据此在 P5 不动这些脚本，只在本文件归类。

- 下列 **7 个 `verify_*.py` 攻击探针**是该约定的原始成员，每个脚本自带 `--output`、
  自写 JSON（`claim` / `requires_*` 布尔前提位 / `by_<axis>` 分轴结果）：
  `verify_free_noise_cone.py`、`verify_gram_leakage.py`、
  `verify_known_plaintext_decoder_recovery.py`、`verify_norm_leak_by_depth.py`、
  `verify_recoverability_bound.py`、`verify_rho_recoverable_from_fused_output.py`、
  `verify_token_recovery_from_rho.py`。
- 4 个 `record_*.py` 与其余 `verify_*.py` 沿用同一「脚本之间零 import」的写法：
  2026-09-20 用 `rg "from scripts|import scripts" scripts/*.py` 核对，**全部 28 个脚本
  之间无互相 import**。它们可以 import `fastprove` 库，但不共享脚本级 util。
- 与 CLEANUP_PLAN:180「7 个 `verify_*.py` 与 `record_*.py`」的计数差异：当时把
  `verify_*.py` 与 `record_*.py` 合并表述为 7 个；当前树上 `verify_*` 9 个、`record_*` 4 个。
  计数漂移不改变约定本身，本文件以当前实际清单为准（差异照实登记，不做平滑）。

## 1. 流水线入口（可写正式 `results/raw` 记录）

| 脚本 | 用途（docstring 首句） | 典型调用 | 主要产物 |
|---|---|---|---|
| `run_correctness.py` | Plan the tiny correctness run（默认只读，`--execute` 才跑模型） | `python scripts/run_correctness.py --execute`（README.md:92） | `results/raw/correctness.jsonl`（`--output` 默认，`run_correctness.py:37-39`） |
| `run_accuracy_sweep.py` | Plan or explicitly execute the complete 72-point aligned sweep | `python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --output results/raw/eval_sweep.jsonl --execute-deferred`（runbook.md:78-79） | `eval_sweep*.jsonl`、可选 preflight manifest |
| `build_report.py` | Plan or explicitly derive tables, figures, and REPORT.md from raw JSONL | `python scripts/build_report.py --raw results/raw/eval_sweep_dtype_fixed.jsonl --output results --execute`（runbook.md:93） | `results/tables/`、`results/figures/`、`results/REPORT.md` |
| `run_pretrained_compare.py` | Plaintext vs current-fastProve obfuscated comparison for Llama (or tiny) | `python scripts/run_pretrained_compare.py --model-path <ckpt> --output <json>`（READY 前置见 runbook.md §8；外部 CUDA argv 待考） | `--output` 默认写 llama_compare.json；本次落盘实例为 `results/raw/llama_compare_P.json`、`results/raw/llama_compare_large.json` |
| `run_osnip_style_benchmarks.py` | OSNIP-style utility benchmarks for plain vs fastProve | `python scripts/run_osnip_style_benchmarks.py --model-path <ckpt> --data-dir results/raw/osnip_bench_data ...`（外部 CUDA argv 待考） | `--output` 默认写 osnip_style_bench.json；本次落盘实例为 `results/raw/osnip_style_bench_fixed.json` 等 |
| `run_augmented_noise_kernel.py` | Stage A kernel: augmented encode → converted linear → legal decode | `python scripts/run_augmented_noise_kernel.py --output-dir results/raw/augmented_noise_kernel` | `kernel_records.jsonl`（未跟踪）+ `kernel_records.json` |
| `audit_pretrained_assets.py` | Audit local pretrained assets and write the external-dependency manifest | `python scripts/audit_pretrained_assets.py --output results/raw/pretrained_evaluation_status.json` | 默认 `results/raw/pretrained_evaluation_status.json`（`:59`） |
| `preflight_experiment.py` | Fail-fast preflight for the real-model fastProve experiment | `python scripts/preflight_experiment.py --model-path <ckpt> --dataset-cache <pt> --device cpu --output <json>` | 就绪 manifest（默认无；阻塞中） |
| `prepare_token_cache.py` | Tokenize a local text/JSONL corpus into an aligned fastProve cache | `python scripts/prepare_token_cache.py --model-path <tok> --input-jsonl <jsonl> --output results/raw/qwen_token_cache_s32.pt --sequence-length 32` | 对齐 token cache（`*.pt`，被 ignore） |
| `extract_flickr_captions.py` | Extract deterministic Flickr30k caption JSONL from a Karpathy JSON file | `python scripts/extract_flickr_captions.py --input <karpathy.json> --output <caption.jsonl>` | caption JSONL（本地数据准备） |
| `build_real_scenario_prompts.py` | Build ≥1500 real-scenario prompts for Llama Instruct comparison | `python scripts/build_real_scenario_prompts.py` | 默认 `results/raw/real_scenario_prompts_1500.jsonl`（`:default`） |
| `build_attack_cost_records.py` | Build `results/raw/attack_costs.jsonl` from recorded attack artifacts | `python scripts/build_attack_cost_records.py --output results/raw/attack_costs.jsonl` | `attack_costs.jsonl`（不跑攻击、不造数） |
| `select_sweep_candidates.py` | Select a traceable full-evaluation subset from calibration JSONL | `python scripts/select_sweep_candidates.py --raw <calibration.jsonl> --output <manifest.json>` | 候选 manifest JSON（不跑推理、不改输入） |
| `measure_overhead.py` | Measure reference-implementation overhead versus the plaintext model | `python scripts/measure_overhead.py --hidden-size 1024 --output results/raw/overhead_B7_d1024.json` | `overhead_B7_d*.json` |

引用（节选）：`run_correctness.py` ← `README.md:92`、`tests/cli/test_cli_plan.py`、
`src/fastprove/evaluation/report_markdown.py`（内嵌命令名）；`run_accuracy_sweep.py` ←
`results/REPORT.md`、`docs/runbook.md`、`tests/cli/test_cli_plan.py`、`select_sweep_candidates.py`
的产物；`build_report.py` ← `AGENTS.md`、`docs/runbook.md`、`results/tables/README.md`、
`tests/cli/test_cli_plan.py`；`measure_overhead.py` ← `results/REPORT.md`、
`docs/implementation_notes.md`；`preflight_experiment.py` ← `src/fastprove/evaluation/pretrained.py`、
`README.md`、`tests/cli/test_cli_plan.py`；`run_pretrained_compare.py` ← `results/REPORT.md`；
`run_osnip_style_benchmarks.py` ← `docs/PROBLEMS.md`（§3.4 的 `_wilson_interval` 历史副本）；
`run_augmented_noise_kernel.py` ← `docs/augmented_noisy_obfuscation_plan.md`、
`docs/IMPLEMENTATION_PROMPT_dual_channel.md`；`build_real_scenario_prompts.py` ←
`src/fastprove/evaluation/prompts.py`（语料说明）；`extract_flickr_captions.py` ←
`results/raw/README.md`；`audit_pretrained_assets.py` ← `docs/runbook.md` §8、
`results/raw/blocked_stages_s6_s7.json`；`build_attack_cost_records.py`、
`record_noise_budget.py`、`record_noise_scale_equivalence.py`、
`verify_free_noise_cone.py` 等探针的消费者主要是 `results/raw/README.md` 的溯源表。

## 2. 一次性探针（生产结果只读；写入 `results/raw` 的论证记录）

### 2.1 安全/攻击探针（自包含单文件，见 §0）

| 脚本 | 用途 | 典型调用 | 产物 |
|---|---|---|---|
| `verify_free_noise_cone.py` | Record the free noise cone `ker(L)` as a first-class object（S0） | `PYTHONPATH=src python scripts/verify_free_noise_cone.py --output results/raw/free_noise_cone_check.json` | `free_noise_cone_check.json` |
| `verify_gram_leakage.py` | Quantify what the server-side Gram blocks reveal about the hidden state | `PYTHONPATH=src python scripts/verify_gram_leakage.py --output results/raw/gram_leakage.json` | `gram_leakage.json` |
| `verify_known_plaintext_decoder_recovery.py` | Tier-0 attack: closed-form semantic-decoder recovery from known plaintext | `PYTHONPATH=src python scripts/verify_known_plaintext_decoder_recovery.py --output results/raw/known_plaintext_decoder_recovery.json` | `known_plaintext_decoder_recovery.json` |
| `verify_norm_leak_by_depth.py` | Measure whether the norm leak decays with depth | `PYTHONPATH=src python scripts/verify_norm_leak_by_depth.py --model-path <ckpt> --prompt-file results/raw/real_scenario_prompts_1500.jsonl --output results/raw/norm_leak_by_depth.json` | `norm_leak_by_depth.json` |
| `verify_recoverability_bound.py` | Reproduce the recoverability upper bounds recorded in threat_model §5bis | `PYTHONPATH=src python scripts/verify_recoverability_bound.py --signal-dim 512 --output results/raw/recoverability_bound.json` | `recoverability_bound.json` |
| `verify_rho_recoverable_from_fused_output.py` | Verify that the RMSNorm scale `rho` is recoverable from a fused kernel's output | `PYTHONPATH=src python scripts/verify_rho_recoverable_from_fused_output.py --output results/raw/rho_recoverable_from_fused_output.json` | `rho_recoverable_from_fused_output.json` |
| `verify_token_recovery_from_rho.py` | Recover input tokens from the RMSNorm scale alone（layer 0） | `PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py --model-path <ckpt> --output results/raw/token_recovery_from_rho.json` | `token_recovery_from_rho{,_llama}.json` |

引用：`verify_gram_leakage.py` ← `docs/threat_model.md`、`docs/PROBLEMS.md`；
`verify_token_recovery_from_rho.py` ← `docs/PROBLEMS.md` §1.1、`docs/threat_model.md` 5bis.9、
`scripts/verify_norm_leak_by_depth.py`；`verify_norm_leak_by_depth.py` ← `docs/PROBLEMS.md` §1.3；
`verify_recoverability_bound.py` / `verify_rho_recoverable_from_fused_output.py` ←
`docs/threat_model.md` §5bis、`docs/PROBLEMS.md` §1.2；`verify_known_plaintext_decoder_recovery.py`
← `tests/security/test_known_plaintext_recovery.py`、`docs/PROBLEMS.md` §1.4。
`verify_free_noise_cone.py` 的消费主要在 `docs/mathematics.md` 与
`results/raw/noise_ordering_analysis_20260919.json` 的通道论证。

### 2.2 记录脚本（S0–S2 与性能）

| 脚本 | 用途 | 典型调用 | 产物 |
|---|---|---|---|
| `record_exact_tolerance.py` | Record exact-mode tolerance for the current checkpoint arithmetic dtype | `PYTHONPATH=src python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_B3.json` | `exact_tolerance_{A4,A4_before,A5,B2,B3}.json` |
| `record_noise_budget.py` | Record the S2 dual-channel ledger produced by a real conversion | `PYTHONPATH=src python scripts/record_noise_budget.py --config configs/tiny_approx.yaml --output results/raw/noise_budget_tiny_approx.json` | `noise_budget_tiny_approx.json` |
| `record_noise_scale_equivalence.py` | S1 bit-exact equivalence recorder for the noise-scale configuration refactor | `PYTHONPATH=src python scripts/record_noise_scale_equivalence.py --config configs/tiny_approx.yaml --output results/raw/post_s1_tiny_approx.json --compare results/raw/pre_s1_baseline_tiny_approx.json` | `pre_s1_*` / `post_s1_*` 哈希记录 |
| `record_performance_D2.py` | Record the task D2 performance metric set for plaintext vs obfuscated | `PYTHONPATH=src python scripts/record_performance_D2.py --output results/raw/performance_D2.json` | `performance_D2.json` |

引用：`record_exact_tolerance.py` ← `results/REPORT.md` §2.2、`docs/PROBLEMS.md`；
`record_performance_D2.py` ← `results/REPORT.md` §2.4、`docs/threat_model.md` 条目 5；
`record_noise_budget.py` / `record_noise_scale_equivalence.py` ← `results/raw/README.md`
（S1/S2 验收记录）。

### 2.3 基础设施验证脚本

| 脚本 | 用途 | 典型调用 | 产物 |
|---|---|---|---|
| `verify_refactor_equivalence.py` | Freeze and verify the pre-refactor numerical baseline | `python scripts/verify_refactor_equivalence.py --check`（runbook.md:29） | `results/raw/refactor_baseline_<sha>.json`；`--check` 只读 |
| `verify_pretrained_exact_mode.py` | Exact-mode verification on a real pretrained checkpoint（PROBLEMS §4.1） | `PYTHONPATH=src python scripts/verify_pretrained_exact_mode.py --model-path <ckpt> --output <json> --layers 4` | 预训练 exact 检查 JSON（checkpoint 阻塞，当前不可跑） |

引用：`verify_refactor_equivalence.py` ← `AGENTS.md`、`README.md:82`、`docs/runbook.md` §2、
`docs/STATUS.md` §2、`docs/implementation_notes.md` P0 节；`verify_pretrained_exact_mode.py`
← 仅 `docs/handoff_evidence/20260915/evidence_manifest.json`（核验材料）。

## 3. 文档工具

| 脚本 | 用途 | 典型调用 | 说明 |
|---|---|---|---|
| `check_doc_links.py` | 扫描 Markdown 中的相对链接与反引号路径是否存在（只读） | `python scripts/check_doc_links.py --skip-archive`（runbook.md:121） | 纯文档工具，不参与等价性门禁；退出码 1 表示有断链 |

引用：`AGENTS.md`（§P8 门）、`docs/runbook.md` §9、`docs/INDEX.md` §6、
`docs/CLEANUP_PLAN.md` P8 门。

## 4. 归档脚本（`scripts/archive/`）

| 脚本 | 状态 / 原因 | 替代 |
|---|---|---|
| `postprocess_llama_report.py` | 无 argparse、硬编码 `results/raw/llama_compare_P.json`、会整体覆盖 `results/REPORT.md`、零引用、跨包 import 私有符号 | `scripts/build_report.py`（`scripts/archive/README.md:15`） |
| `write_chinese_report.py` | 第三套独立报告写手，从旧 compare JSON 生成 `docs/archive/2026-08-06_REPORT_zh.md` | `scripts/build_report.py`（`scripts/archive/README.md:16`） |

两者仍可 `-h/--help`（归档不改内容），但**不要**在现行流水线中调用；
`scripts/archive/README.md:21` 的冻结检索命令可确认除归档说明外无现行引用。

## 5. 产物与跟踪状态速查

- 正式运行记录写入 `results/raw/`；`*.jsonl`、`*.pt` 默认被 `.gitignore` 排除，
  白名单仅 `eval_sweep.jsonl`、`eval_sweep_dtype_fixed.jsonl`、`attack_costs.jsonl`、
  `real_scenario_prompts_1500.jsonl`（`.gitignore` 尾部白名单）。
- 逐文件的 claim→命令映射见 `results/PROVENANCE.md`；目录盘点见 `results/raw/README.md`。
- 预训练相关脚本（`preflight_experiment.py`、`prepare_token_cache.py`、
  `run_accuracy_sweep.py --model-path ...`、`run_pretrained_compare.py`、
  `verify_token_recovery_from_rho.py --model-path ...`）当前被缺失的 checkpoint 阻塞，
  见 `docs/runbook.md` §8 与 `results/raw/blocked_stages_s6_s7.json`。
