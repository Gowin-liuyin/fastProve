# results/raw 目录清单与溯源

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/INDEX.md、docs/PROBLEMS.md
适用范围：`results/raw/` 的实际目录盘点：每条记录的用途、生成命令、跟踪状态与备注

本目录当前共有 **65 条条目**（62 个文件 + 3 个子目录；不含本 README）。
下表的「生成命令」只写仓库内可考证的来源：能找到脚本默认路径/docstring 的写脚本
路径；exact argv 无法考证的标 `待考`。所有 `results/raw/*.jsonl`、`*.pt` 默认被
`.gitignore` 排除，白名单仅 `eval_sweep.jsonl`、`eval_sweep_dtype_fixed.jsonl`、
`eval_sweep_rho0.jsonl`、`eval_sweep_rho0_calibration.jsonl`、
`attack_costs.jsonl`、`real_scenario_prompts_1500.jsonl`。

> 口径声明：除预训练相关记录外，绝大部分精度/安全数字来自随机 tiny 模型或小张量
> 探针（`meaningful_lm_evidence: False`），只能作为数学/工程正确性证据，不得描述为
> 语言模型能力证据（AGENTS.md）。

## 1. 当前扫描、门禁与性能记录

| 文件 | 记录内容 | 生成命令 | 跟踪 | 备注 |
|---|---|---|---|---|
| `eval_sweep.jsonl` | 72/72 点噪声 sweep（2 基线 + 63 topk_preserving + 7 free_bounded） | `PYTHONPATH=src python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --execute-deferred` | 跟踪（白名单） | **superseded**：`checkpoint_compute_dtype` 标签错误，见 `docs/PROBLEMS.md` §3.2；`results/REPORT.md` 的引用按用户决定 ⑥ 不改，仅在此标注 |
| `eval_sweep_dtype_fixed.jsonl` | 同上，dtype 溯源字段已修正 | 同上（修 `structured.reduction_compute_dtype_name` 后重跑） | 跟踪（白名单） | **superseded（2026-09-20）**：被 ρ=0 重跑取代；72 个 spec 的精度指标仍逐位相同，只有被修标签与墙钟/内存类差异 |
| `eval_sweep_rho0.jsonl` | 72/72 点 sweep，ρ=0 基础口径（**当前规范**） | `python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --output results/raw/eval_sweep_rho0.jsonl --execute-deferred` | 跟踪（白名单） | 记录自带生效 `obfuscation`（ρ=0、`per_request`）与修正后的 dtype 标签；125 个非性能指标与旧两份逐位相同；见 `docs/PROBLEMS.md` §3.5、`results/PROVENANCE.md` §7 |
| `eval_sweep_rho0_calibration.jsonl` | 同上，`--calibration-only`（sample_count=8）冒烟 | 同命令加 `--calibration-only` | 跟踪（白名单） | 全量重跑前的口径/schema 检查：72/72 成功、exact 门禁通过 |
| `exact_gate_D1.json` | FP32 exact 硬/软门禁明细 | `PYTHONPATH=src:evals python -m evals --base-config configs/tiny_exact.yaml --config evals/configs/P2.yaml --precision fp32 --keys 1 --layers 1,2,3,4 --check-gates --output results/raw/exact_gate_D1.json` | 跟踪 | `results/REPORT.md` §2.1 的头条门禁 |
| `exact_gate_D1_bf16.json` | BF16 独立条件（非 exact 门禁） | 同上，`--precision bf16`（确切 argv 待考） | 跟踪 | BF16 只作独立 condition 记录 |
| `exact_tolerance_A4.json` / `exact_tolerance_A4_before.json` | A4 阶段前后 exact logits 偏差 | `PYTHONPATH=src python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_A4.json` | 跟踪 | `A4_before` 是修复前对照 |
| `exact_tolerance_A5.json` | A5（结构化基）exact logits 偏差 | 同上，输出改 A5 | 跟踪 | 见 `results/REPORT.md` §2.2 表 |
| `exact_tolerance_B2.json` | B2（部署 attention）exact logits 偏差 | 同上，输出改 B2 | 跟踪 | 同上 |
| `exact_tolerance_B3.json` | B3（部署 FFN）exact logits 偏差 | 同上，输出改 B3 | 跟踪 | 同上；docstring 的示例命令 |
| `overhead_B7_d512.json`、`overhead_B7_d1024.json`、`overhead_B7_d2048.json` | eager 参考实现 prefill/decode 开销（d=512/1024/2048） | `PYTHONPATH=src python scripts/measure_overhead.py --hidden-size <d> --output results/raw/overhead_B7_d<d>.json` | 跟踪 | 全部带 `implementation: eager reference, not a fused kernel` |
| `performance_D2.json` | D2 性能记录（TTFT/TPOT/kernel 数/KV cache/tied-embedding 投影） | `PYTHONPATH=src python scripts/record_performance_D2.py --output results/raw/performance_D2.json` | 跟踪 | 见 `results/REPORT.md` §2.4 |
| `refactor_baseline_5e19c43ea2b2.json` | P0 冻结的 12 section 等价性基线（2395 摘要叶子） | `python scripts/verify_refactor_equivalence.py`（自写 `results/raw/refactor_baseline_<sha>.json`） | 跟踪 | 必须被 `--check` 逐位复现；P0 验收证据 |
| `noise_budget_tiny_approx.json` | tiny_approx 转换产出的双通道噪声账本 + runtime obfuscation 段 | `PYTHONPATH=src python scripts/record_noise_budget.py --config configs/tiny_approx.yaml --output results/raw/noise_budget_tiny_approx.json` | 跟踪 | S2 账本；channel 0/1 语义见 `src/fastprove/noise_budget.py` |
| `pre_s1_baseline_tiny_exact.json`、`pre_s1_baseline_tiny_approx.json` | S1 重构前的 state_dict/logits 逐位哈希 | `PYTHONPATH=src python scripts/record_noise_scale_equivalence.py --config <yaml> --output results/raw/pre_s1_baseline_tiny_<mode>.json` | 跟踪 | S1 以逐位相等为验收 |
| `post_s1_tiny_exact.json`、`post_s1_tiny_approx.json` | S1 重构后的哈希 + `comparison` 结论 | 同上，输出改 `post_s1_*`（可用 `--compare pre_s1_baseline_*`） | 跟踪 | 与 pre 文件逐位相等 |

## 2. 预训练资产、语料与外部 GPU 记录

| 文件 | 记录内容 | 生成命令 | 跟踪 | 备注 |
|---|---|---|---|---|
| `pretrained_evaluation_status.json` | 本地资产审计（不加载权重）：`skipped_external_dependency`，Qwen2 checkpoint 曾找到并哈希 | `python scripts/audit_pretrained_assets.py --output results/raw/pretrained_evaluation_status.json` | 跟踪 | **该 Qwen2 checkpoint 已从本机消失**，见 `blocked_stages_s6_s7.json` |
| `pretrained_evaluation_status_caption_candidate.json` | 同上，限定 `caption_only_nonstandard_lm_candidate` 范围 | 同上（另一次审计，确切 argv 待考） | 跟踪 | 不等价于标准 causal-LM 证据 |
| `flickr30k_caption_preflight.json`、`flickr30k_caption_preflight_bf16.json` | 旧 generic caption preflight（历史诊断） | `python scripts/preflight_experiment.py ... --output ...`（确切 argv 待考） | 跟踪 | 不得替代显式 device/范围 opt-in 的 preflight |
| `flickr30k_caption_preflight_cpu_fp32.json`、`flickr30k_caption_preflight_mps_fp32.json`、`flickr30k_caption_preflight_mps_bf16.json` | 显式 device/dtype 的无权重资源/身份预检 | `python scripts/preflight_experiment.py --device <dev> --activation-dtype <dt> --accept-nonstandard-caption --output ...`（确切 argv 待考） | 跟踪 | 不含 PPL/accuracy/性能结果 |
| `flickr30k_caption_eval_inputs.pt` | 405 条 caption-only token cache（长度 24） | `python scripts/prepare_token_cache.py ...`（caption 由 `scripts/extract_flickr_captions.py` 抽出；确切 argv 待考） | **未跟踪（被忽略）** | 只有显式 `--accept-nonstandard-caption` 才会被视为可执行候选 |
| `qwen_token_cache_s32.pt` | Qwen 对齐 token cache（seq=32） | `python scripts/prepare_token_cache.py --output results/raw/qwen_token_cache_s32.pt ...`（确切 argv 待考） | **未跟踪（被忽略）** | 与 `real_scenario_prompts_1500.jsonl` 同源 |
| `real_scenario_prompts_1500.jsonl` | 1500 条真实场景 prompt（`{id, text}`） | `python scripts/build_real_scenario_prompts.py` | 跟踪（白名单） | 被多个探针脚本作为 prompt 输入 |
| `experiment_readiness.json` | 就绪交接清单：可执行范围/资源警告/第一条校准命令 | 疑似 `scripts/preflight_experiment.py`/就绪审计落盘（确切命令待考） | 跟踪 | 不是实验结果 |
| `gates_recompute.log` | 历史门禁重算输出：一次 `HARD: FAIL / SOFT: FAIL` + `wrote results/REPORT.md` | 待考（生成脚本已不存在） | 跟踪 | 历史诊断，勿引用为当前门禁状态（当前 exact 门禁见 `exact_gate_D1.json`） |
| `eval_F1_smoke.json` | evals harness F1 smoke 条件 JSON | `PYTHONPATH=src:. python -m evals --config evals/configs/F1.yaml --keys 1 --layers 1,2,3,4 --sample-count 4 --check-gates --output ...`（依 `evals/README.md`，确切 argv 待考） | 跟踪 | 随机 tiny，仅正确性 |
| `eval_P2.json` | evals P2 条件 JSON（keys=3） | `evals/README.md` 的 output 示例（`--config evals/configs/P2.yaml --keys 3 --check-gates`） | 跟踪 | `protocol` 指向评测方案 |
| `eval_P2_k5.json` | evals P2 条件 JSON（keys=5） | 同上，`--keys 5`（确切 argv 待考） | 跟踪 | `top1_token_agreement_ci95` 用 `evals.stats` 定义（`docs/PROBLEMS.md` §3.4） |
| `llama_compare_P.json` / `llama_compare_large.json` | Llama-3.2-3B 上 plaintext vs 本方案对比（含 legacy arm 字段与 superseded 标记） | `python scripts/run_pretrained_compare.py ... --output ...`（外部 CUDA 机 argv 待考） | 跟踪 | 自标 **superseded**（harness 选择缺陷，不得引用任务数字） |
| `llama_compare_run.log` / `llama_compare_large.log` | 上述两次运行的 tty 日志 | 重定向自同命令 | 跟踪 | 同上 |
| `osnip_style_bench_smoke.json`、`osnip_style_bench_n200.json` | OSNIP 风格 smoke / 200 样本对照 | `python scripts/run_osnip_style_benchmarks.py ... --output ...`（外部 CUDA 机 argv 待考） | 跟踪 | 自标 **superseded**（mmlu/anli 语料排序等缺陷，不得引用任务数字） |
| `osnip_style_bench_fixed.json` / `_fixed.log` | 修复采样后的 plain/structural/full 任务对照 | 同上（外部 CUDA 机 argv 待考） | 跟踪 | 仍是随机化前 harness 的后续修复版；引用前必须连同其 caveat |
| `osnip_style_bench_fixed_plain.json` / `_plain.log` | 只跑 plaintext 基线的对照 | 同上 | 跟踪 | 无法与 OSNIP 论文基线（0.597）对账 |
| `osnip_style_bench_fixed_plain_v2.json` | plaintext 基线的 v2 记录 | 同上 | 跟踪 | 无自标 superseded |
| `osnip_bench_data/` | 10 个本地任务数据文件 + wikitext2（未跟踪） | 外部下载/整理（来源与命令待考） | **未跟踪（被忽略）** | 被 `run_osnip_style_benchmarks.py` 读取 |

## 3. 数学/安全探针与侧路实验

| 文件 | 记录内容 | 生成命令 | 跟踪 | 备注 |
|---|---|---|---|---|
| `free_noise_cone_check.json` | `ker(L)` 维数、主角度、pure-concat vs generic 噪声的最大语义泄漏 | `PYTHONPATH=src python scripts/verify_free_noise_cone.py --output results/raw/free_noise_cone_check.json` | 跟踪 | channel 0 隐藏量恰为 0 bit 的核验 |
| `gram_leakage.json` | 块对角 Gram 恢复行范数/内积的最大误差 | `PYTHONPATH=src python scripts/verify_gram_leakage.py --output results/raw/gram_leakage.json` | 跟踪 | 威胁模型 5bis.8 条目 6 |
| `known_plaintext_decoder_recovery.json` | 已知明文恢复 `M` 的样本代价/几何 | `PYTHONPATH=src python scripts/verify_known_plaintext_decoder_recovery.py --output results/raw/known_plaintext_decoder_recovery.json` | 跟踪 | PROBLEMS 1.4 |
| `recoverability_bound.json` | 多攻击/扫描下的恢复界复核 | `PYTHONPATH=src python scripts/verify_recoverability_bound.py --signal-dim 512 --output results/raw/recoverability_bound.json` | 跟踪 | 5bis 复核 |
| `rho_recoverable_from_fused_output.json` | `rho` 从融合输出可恢复（无需改 kernel） | `PYTHONPATH=src python scripts/verify_rho_recoverable_from_fused_output.py --output results/raw/rho_recoverable_from_fused_output.json` | 跟踪 | PROBLEMS 1.2 |
| `token_recovery_from_rho.json`、`token_recovery_from_rho_llama.json` | 用 `rho` 恢复 layer-0 输入 token 的恢复率（Qwen2 / Llama-3.2-3B，n=4000） | `PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py --model-path <ckpt> --output results/raw/token_recovery_from_rho.json` | 跟踪 | 依赖已从本机消失的 Qwen2 checkpoint；Llama JSON 同 |
| `norm_leak_by_depth.json` | 按层/语料频率的范数泄漏诊断 | `PYTHONPATH=src python scripts/verify_norm_leak_by_depth.py --output results/raw/norm_leak_by_depth.json` | 跟踪 | 有语料 caveat（`corpus_caveat` 字段） |
| `attack_costs.jsonl` | `attack.v1` 成本记录（6 条：claim/coa_view/cost/status/recovery） | `python scripts/build_attack_cost_records.py --output results/raw/attack_costs.jsonl` | 跟踪（白名单） | 派生自上述多个探针 JSON+脚本耗时 |
| `noise_ordering_analysis_20260919.json` | T1–T9 噪声放置推导数值核验 | `PYTHONPATH=src python docs/verification/verify_noise_ordering.py --output results/raw/noise_ordering_analysis_20260919.json` | 跟踪 | 定理 2/4/9 的数值依据 |
| `blocked_stages_s6_s7.json` | S6/S7 未执行及原因（checkpoint 不存在，`checkpoint_bytes = 0`） | 手工登记的阻塞记录（无生成脚本） | 跟踪 | τ=1e-5 反证实验仍未关闭；见 `docs/STATUS.md` |
| `augmented_covariant_v2_math_check_20260916.json` | v2 数学方案 FP64 检查（compact-primary） | `python docs/verification/verify_augmented_covariant_v2.py --output results/raw/augmented_covariant_v2_math_check_20260916.json` | 跟踪 | `meaningful_lm_evidence: False`；旧框架（已归档） |
| `expanded_concat_v2_math_check_20260916.json` / `expanded_augmentation_v2_math_check_20260916.json` | `expanded_concat` / `expanded_full` 候选的检查 | 同上，不同模式/输出 | 跟踪 | **旧框架，禁止复活**（生产路径 D = d） |
| `augmented_noise_kernel/kernel_records.json` | 阶段 A 增广带噪核实验记录（JSON 版） | `PYTHONPATH=src python scripts/run_augmented_noise_kernel.py` | 跟踪 | 侧路实验；无 Transformer 接入 |
| `augmented_noise_kernel/kernel_records.jsonl` | 同上（JSONL 版，逐行） | 同上 | **未跟踪（被忽略）** | `.gitignore` 的 `results/raw/**/*.jsonl` 生效 |
| `handoff_diagnostic_20260915/public_query_projection.json` | 2026-09-15 交接诊断：public query 投影探针 | `python docs/handoff_evidence/20260915/probe_public_query_projection.py --output results/raw/handoff_diagnostic_20260915/public_query_projection.json` | 跟踪 | 核验材料，不属于 Markdown 文档生命周期 |

## 4. 记录规范

- 新的 sweep/评测 attempt 使用独立子目录与全新空 JSONL；原型不支持 partial resume；
- 随机 Tiny 记录必须带 `correctness_only_random_tiny` / `meaningful_lm_evidence: False`
  口径，不能作为真实语言模型精度证据；
- 已标 superseded 的记录保留原样、不改数字、不删文件；引用时必须说明 superseded；
- 派生表/图只允许读取指定 attempt 下可验证的成功记录（`results/REPORT.md` §「结果口径」）；
- 不得提交 checkpoint、数据集、缓存或大体积生成物；确需入库的小记录走 `.gitignore` 白名单。
