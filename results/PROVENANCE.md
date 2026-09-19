# results/ 溯源：claim → 原始工件 → 生成命令

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/INDEX.md、docs/PROBLEMS.md、results/raw/README.md
适用范围：`results/REPORT.md` 与安全/门禁结论所引用的原始工件；逐文件目录盘点的完整版见 `results/raw/README.md`

本文件把「报告里的数字/结论」映射到「仓库内可复核的原始文件」与「生成命令」。规则：

- **current**：仍可作为当前口径引用；**superseded**：保留原样、不改数字，引用必须带 caveat；
- `跟踪` = `git ls-files` 命中；`忽略` = 被 `.gitignore` 排除（`.gitignore` 尾部白名单：
  `eval_sweep.jsonl`、`eval_sweep_dtype_fixed.jsonl`、`attack_costs.jsonl`、
  `real_scenario_prompts_1500.jsonl` 四个 `*.jsonl`）；
- 体积为 2026-09-20 工作树的字节数；
- **口径**：除预训练相关记录外，全部数字来自随机 tiny 模型或小张量探针
  （`meaningful_lm_evidence: False`），只作数学/工程正确性证据，不是语言模型能力证据。

> 行号说明：CLEANUP_PLAN §10-⑥ 与本次任务引用的「PROBLEMS.md 713-714」按当前工作树
> 行号并不指向 dtype 条目（P8b 增删后行号已漂移）。该 superseded 事实的当前实际位置是
> `docs/PROBLEMS.md` §3.2（:425-446）与 :887-888 的工件表。此处照实登记，不做平滑。

## 1. `results/REPORT.md` 头条数字

| claim（REPORT 位置） | 原始工件 | 生成命令 | 状态 / 跟踪 / 体积 |
|---|---|---|---|
| FP32 exact 硬/软门禁全 PASS（§2.1） | `results/raw/exact_gate_D1.json` | `PYTHONPATH=src:evals python -m evals --base-config configs/tiny_exact.yaml --config evals/configs/P2.yaml --precision fp32 --keys 1 --layers 1,2,3,4 --check-gates --output results/raw/exact_gate_D1.json`（REPORT.md:48-50） | current / 跟踪 / 23,381 B |
| BF16 独立 condition（§2.2） | `results/raw/exact_gate_D1_bf16.json` | 同上 `--precision bf16`（确切 argv 待考；raw README:24-25） | current（非 exact 门禁）/ 跟踪 / 10,581 B |
| exact logits 偏差 A4/A5/B2/B3（§2.2 表） | `results/raw/exact_tolerance_A4.json`、`A4_before`、`A5`、`B2`、`B3` | `PYTHONPATH=src python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_<阶段>.json`（REPORT.md:180、raw README:26-29） | A4_before 为修复前对照；其余 current / 跟踪 / 各约 400 B |
| 72/72 点 sweep 与 §2.3 两张精度表 | `results/raw/eval_sweep_dtype_fixed.jsonl` | `PYTHONPATH=src python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --execute-deferred`（修 `structured.reduction_compute_dtype_name()` 后重跑） | **current**（72 个 spec 精度指标与旧份逐位相同）/ 跟踪（白名单）/ 923,805 B |
| 同上（REPORT.md:79、:189 实际引用的文件） | `results/raw/eval_sweep.jsonl` | 同命令（修复前一次运行） | **superseded**：`checkpoint_compute_dtype` 标签错误，见 `docs/PROBLEMS.md` §3.2（:425-446）；按用户决定 ⑥ 不改 `REPORT.md` 的引用 / 跟踪（白名单）/ 923,785 B |
| 固定 seed 两次运行 72 个 metric 树逐位相同（§2.3） | 第二遍对照在 `/tmp`，**未入库**（REPORT.md:189） | 同 sweep 命令重跑一遍后逐位比对 | 结论无仓库内原始对照，只有第一遍记录；照实标注 |
| 性能开销表（§2.4） | `results/raw/overhead_B7_d512.json`、`d1024`、`d2048` | `PYTHONPATH=src python scripts/measure_overhead.py --hidden-size <d> --output results/raw/overhead_B7_d<d>.json`（REPORT.md:181-182） | current / 跟踪 / 1,005–1,300 B 量级 |
| D2 性能记录（§2.4） | `results/raw/performance_D2.json` | `PYTHONPATH=src python scripts/record_performance_D2.py --output results/raw/performance_D2.json`（REPORT.md:183） | current / 跟踪 / 2,775 B |
| 「服务端目录可脱离客户端密钥运行」（§2.1） | 测试：`tests/deployment/test_converter.py` | `python -m pytest tests/deployment/test_converter.py -q` | current（代码级证据，非 raw JSON） |

**块/层（block/level）指标**没有独立文件：它们嵌在每条 sweep run record 的
`metrics.layer.*`（`per_layer`、`qk_scores`、`exact_softmax_probabilities`）里，由
`src/fastprove/evaluation/layer_metrics.py:51` 的 `_layer_metrics` 汇总、经 runner 写入。
复核方式：对 `eval_sweep_dtype_fixed.jsonl` 的任一记录读取 `metrics["layer"]`，
生成命令与上表 sweep 相同。

## 2. 安全/数学探针

| claim | 原始工件 | 生成命令 | 状态 / 跟踪 / 体积 |
|---|---|---|---|
| T1–T9 噪声放置推导数值核验（theorem 2/4/9） | `results/raw/noise_ordering_analysis_20260919.json` | `PYTHONPATH=src python docs/verification/verify_noise_ordering.py --output results/raw/noise_ordering_analysis_20260919.json`（runbook.md:49-50；只读复跑可省略 `--output`） | current / 跟踪 / 23,023 B |
| channel 0（`ker(L)`）隐藏量恰为 0 | `results/raw/free_noise_cone_check.json` | `PYTHONPATH=src python scripts/verify_free_noise_cone.py --output results/raw/free_noise_cone_check.json` | current / 跟踪 / 1,337 B |
| 块对角 Gram 泄漏（threat_model 5bis.8 条目 6） | `results/raw/gram_leakage.json` | `PYTHONPATH=src python scripts/verify_gram_leakage.py --output results/raw/gram_leakage.json` | current / 跟踪 / 1,343 B |
| 已知明文恢复 `M`（PROBLEMS §1.4） | `results/raw/known_plaintext_decoder_recovery.json` | `PYTHONPATH=src python scripts/verify_known_plaintext_decoder_recovery.py --output results/raw/known_plaintext_decoder_recovery.json` | current / 跟踪 / 7,680 B |
| 恢复界复核（threat_model §5bis） | `results/raw/recoverability_bound.json` | `PYTHONPATH=src python scripts/verify_recoverability_bound.py --signal-dim 512 --output results/raw/recoverability_bound.json` | current / 跟踪 / 5,654 B |
| `rho` 可从融合输出恢复（PROBLEMS §1.2） | `results/raw/rho_recoverable_from_fused_output.json` | `PYTHONPATH=src python scripts/verify_rho_recoverable_from_fused_output.py --output results/raw/rho_recoverable_from_fused_output.json` | current / 跟踪 / 1,837 B |
| `rho` 恢复 layer-0 token（92.5% Qwen2 / 93.4% Llama） | `results/raw/token_recovery_from_rho.json`、`token_recovery_from_rho_llama.json` | `PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py --model-path <ckpt> --output results/raw/token_recovery_from_rho.json` | current；**依赖已从本机消失的 checkpoint**，重跑受阻 / 跟踪 / 2,080 B、2,045 B |
| 范数泄漏随深度/语料频率诊断 | `results/raw/norm_leak_by_depth.json` | `PYTHONPATH=src python scripts/verify_norm_leak_by_depth.py --model-path <ckpt> --prompt-file results/raw/real_scenario_prompts_1500.jsonl --output results/raw/norm_leak_by_depth.json` | current，带 `corpus_caveat`（PROBLEMS §1.3）/ 跟踪 / 8,296 B |
| 双通道账本（channel 0/1） | `results/raw/noise_budget_tiny_approx.json` | `PYTHONPATH=src python scripts/record_noise_budget.py --config configs/tiny_approx.yaml --output results/raw/noise_budget_tiny_approx.json` | current / 跟踪 / 1,636 B |
| S1 噪声标度逐位等价 | `results/raw/pre_s1_baseline_tiny_{exact,approx}.json`、`results/raw/post_s1_tiny_{exact,approx}.json` | `PYTHONPATH=src python scripts/record_noise_scale_equivalence.py --config <yaml> --output <out> [--compare <pre>]` | current / 跟踪 / 各 ~1 KB 量级 |
| S2 账本一致性（转换期） | 测试：`tests/math/test_noise_budget.py`（含 `test_conversion_writes_noise_budget_next_to_runtime_config`） | `python -m pytest tests/math/test_noise_budget.py -q` | current |

## 3. 侧路实验与预训练资产

| claim / 用途 | 原始工件 | 生成命令 | 状态 / 跟踪 / 体积 |
|---|---|---|---|
| 阶段 A 增广带噪核（侧路，非推理路径） | `results/raw/augmented_noise_kernel/kernel_records.json`（JSON 版，跟踪）与 `kernel_records.jsonl`（忽略） | `PYTHONPATH=src python scripts/run_augmented_noise_kernel.py --output-dir results/raw/augmented_noise_kernel`（raw README:77-78） | current；**不是**安全结论 / 跟踪 19,139 B；忽略 17,025 B |
| 旧框架 v2 数学检查（compact/expanded_concat/expanded_full） | `results/raw/augmented_covariant_v2_math_check_20260916.json`、`expanded_concat_v2_math_check_20260916.json`、`expanded_augmentation_v2_math_check_20260916.json` | `python docs/verification/verify_augmented_covariant_v2.py --output <out>`（raw README:75-76） | **旧框架，禁止复活**；保留原样 / 跟踪 |
| 预训练资产审计（不加载权重） | `results/raw/pretrained_evaluation_status.json` | `python scripts/audit_pretrained_assets.py --output results/raw/pretrained_evaluation_status.json`（raw README:41） | current；Qwen2 artifact **已从本机消失**，重跑报 `checkpoint_bytes = 0` / 跟踪 / 13,275 B |
| caption-only 候选审计 | `results/raw/pretrained_evaluation_status_caption_candidate.json` | 同审计脚本另一次运行（确切 argv 待考） | current（非标准 causal-LM 证据）/ 跟踪 |
| 1500 条真实场景 prompt | `results/raw/real_scenario_prompts_1500.jsonl` | `python scripts/build_real_scenario_prompts.py`（默认输出即该路径） | current / 跟踪（白名单）/ 293,374 B |
| 攻击成本记录（`fastprove.attack.v1`，6 条含 `not_executed`） | `results/raw/attack_costs.jsonl` | `python scripts/build_attack_cost_records.py --output results/raw/attack_costs.jsonl` | current；派生自上述探针 JSON，不跑攻击、不造数 / 跟踪（白名单）/ 10,952 B |
| S6/S7 未执行与 checkpoint 缺失 | `results/raw/blocked_stages_s6_s7.json` | 手工登记的阻塞记录（**无生成脚本**） | current；τ=1e-5 反证实验未关闭 / 跟踪 / 2,221 B |

## 4. OSNIP 风格与 Llama 对比（外部 CUDA 机，argv 待考）

| 原始工件 | 用途 | 生成命令 | 状态 / 跟踪 / 体积 |
|---|---|---|---|
| `results/raw/osnip_style_bench_fixed.json`（+`.log`） | 修复采样后的 plain/structural/full 任务对照 | `python scripts/run_osnip_style_benchmarks.py --model-path <ckpt> --output ...`（外部 CUDA 机 argv 待考；raw README:55-58） | current **但引用必须带 harness caveat**；无法与 OSNIP 论文基线 0.597 对账 / 跟踪 / 62,378 B |
| `results/raw/osnip_style_bench_fixed_plain.json`、`results/raw/osnip_style_bench_fixed_plain_v2.json` | 只跑 plaintext 基线的对照 | 同上 | `_plain` 无自标 superseded；`results/raw/osnip_style_bench_fixed_plain_v2.json` 为 v2 记录 / 跟踪 / 10,828 B、11,697 B |
| `results/raw/osnip_style_bench_smoke.json`、`osnip_style_bench_n200.json` | smoke / 200 样本对照 | 同上 | **superseded**（mmlu/anli 语料排序等缺陷）/ 跟踪 / 10,791 B、16,557 B |
| `results/raw/llama_compare_P.json`、`llama_compare_large.json`（+`.log`） | Llama-3.2-3B 上 plaintext vs 本方案对比 | `python scripts/run_pretrained_compare.py --model-path <ckpt> --output ...`（外部 CUDA argv 待考；raw README:53-54） | 自标 **superseded**（harness 选择缺陷，不得引用任务数字）/ 跟踪 / 29,268 B、188,023 B |

`results/raw/osnip_bench_data/`（10 个任务数据 + wikitext2）被 `.gitignore` 排除，
来源与整理命令待考；`results/raw/*.log` 为同命令 tty 日志。

## 5. 不在此表内的记录

- `eval_F1_smoke.json`、`eval_P2.json`、`eval_P2_k5.json`：evals harness 条件记录，
  命令见 `evals/README.md` 与 `results/raw/README.md:50-52`；
- `experiment_readiness.json`、`flickr30k_caption_preflight*.json`、
  `flickr30k_caption_eval_inputs.pt`、`qwen_token_cache_s32.pt`、`gates_recompute.log`、
  `handoff_diagnostic_20260915/`：就绪/诊断/缓存类，逐条见 `results/raw/README.md`；
- `results/raw/refactor_baseline_5e19c43ea2b2.json`：P0 等价性基线，命令
  `python scripts/verify_refactor_equivalence.py`，复核用 `--check`（runbook.md §2）。

## 6. 复现前置条件

- 包安装：`python -m pip install -e '.[test]'`（P0/P1 后不再需要脚本内 `sys.path` 引导）；
- `evals` 门禁命令仍需 `PYTHONPATH=src:evals`（历史命令按原样保留，P6a 已同时提供打包路径）；
- 预训练相关命令需要仓库外的 checkpoint 与本机可用算力；当前阻塞状态见
  `docs/runbook.md` §8 与 `results/raw/blocked_stages_s6_s7.json`。**不得**用随机 tiny
  结果替代这些测量（AGENTS.md）。
