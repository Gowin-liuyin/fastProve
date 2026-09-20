# 配置参考（configs/*.yaml）

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、`src/fastprove/config.py`、`src/fastprove/noise_budget.py`
适用范围：`configs/` 四个 YAML 的每个键：语义、默认值、取值域与校验来源；不含 `evals/configs/*.yaml`

本文件把目前只存在于 YAML 注释、`config.py` 与 `noise_budget.py` docstring 里的配置词汇
集中成一份参考。所有结论以工作树源码为准；校验行号指向 `src/fastprove/config.py` 等文件。

## 0. 四个配置的定位

| 文件 | 谁读它 | 状态 |
|---|---|---|
| `configs/tiny_exact.yaml` | `fastprove.config.load_config`（`config.py:233`） | CURRENT：exact 正确性路径（ρ=0、`fixed_debug`） |
| `configs/tiny_approx.yaml` | `load_config` | CURRENT：部署折衷路径（ρ=1e-5、`per_request`） |
| `configs/eval_sweep.yaml` | `scripts/run_accuracy_sweep.py:153`（`_load_sweep`）与 `evaluation/sweep.py:44`（枚举） | CURRENT：72 点 sweep 驱动；**不是** `PrototypeConfig` 文件 |
| `configs/augmented_noise_kernel.yaml` | 无 Python reader | **documentation-only**，见 §7 |

`tiny_*` 两个文件由 `PrototypeConfig` 五段（model/obfuscation/attention/runtime/evaluation）
组成；`load_config` 要求五段齐全（`config.py:238-241`），未知键会在 dataclass 构造时以
`TypeError` 失败（没有静默忽略）。

## 1. `model` 段（tiny_exact / tiny_approx 共用）

| 键 | 两个 YAML 的值 | dataclass 默认 | 取值域 / 校验 |
|---|---|---|---|
| `vocab_size` | 64 | 必填 | >0；`config.py:31-42` |
| `hidden_size` | 32 | 必填 | >0，且 `% num_attention_heads == 0`；`config.py:43-46` |
| `intermediate_size` | 64 | 必填 | >0；`config.py:31-42` |
| `num_layers` | 2 | 必填 | >0；`config.py:31-42` |
| `num_attention_heads` | 4 | 必填 | >0，且 `% num_key_value_heads == 0`；`config.py:47-50` |
| `num_key_value_heads` | 2 | 必填 | >0；GQA 比 `query_heads_per_kv_head`（`config.py:62-66`） |
| `max_sequence_length` | 32 | 必填 | >0；与 evaluation 段交叉校验（见 §6） |
| `rms_epsilon` | 0.00001 | `1e-5`（`config.py:27`） | >0；`config.py:53` |
| `rope_theta` | 10000.0 | `10000.0`（`config.py:28`） | >0；`config.py:53`；实现为单一 theta 原始 RoPE，**不含** llama3 `rope_scaling`（`docs/PROBLEMS.md` §4.2） |
| `qkv_bias` | **未出现**（默认 False） | `False`（`config.py:29`） | bool；Qwen2 适配器在代码内设 `True`（`pretrained/qwen2.py:96`），Llama 由 checkpoint `attention_bias` 决定（`pretrained/llama.py:128`） |

head_dim = hidden_size / num_attention_heads（`config.py:56-60`），必须为偶数以支持 RoPE
（`config.py:51-52`）。

## 2. `obfuscation` 段

| 键 | tiny_exact | tiny_approx | dataclass 默认 | 取值域 / 校验 |
|---|---|---|---|---|
| `hidden_noise_dim` | 8 | 8 | 必填 | >0；`config.py:94-95` |
| `value_noise_dim_per_head` | 2 | 2 | 必填 | >0；`config.py:94-95` |
| `max_condition_number` | 10.0 | 10.0 | 必填 | 有限且 ≥1；`config.py:100-101` |
| `noise_propagation_gamma` | 0.5 | 0.5 | 必填 | 有限且 `0 < γ < 1`；`config.py:102-103` |
| `refresh_mode` | `fixed_debug` | `per_request` | 必填 | `{fixed_debug, per_request}`；`config.py:104-105`，见 §3 |
| `basis_block_size` | 8 | 8 | `16`（`config.py:78`） | >0，且必须整除 `hidden_size + hidden_noise_dim`；`config.py:96-97`、`213-220`，见 §5 |
| `rho_noise_bound` | `0.0` | `0.00001` | `0.0`（`config.py:80`） | 有限且 ≥0；`config.py:106-107`，见 §4 |
| `free_noise_energy_ratio` | 0.02 | 0.02 | `0.02`（`config.py:86`） | 有限且 ≥0，且 ≤ `AUXILIARY_MAGNITUDE_BOUND = 30.0`（`policy.py:21`、`config.py:118-123`）；见 §4 |
| `signal_noise_coupling_scale` | 0.02 | 0.02 | **`0.0`**（`config.py:91`） | 有限且 ≥0；`config.py:108-113`；见 §4 |
| `lm_head_mode` | **未出现**（默认 `untied_deployed`） | 同左 | `untied_deployed`（`config.py:79`） | `{untied_deployed, fused_norm_head}`；`config.py:98-99`；`fused_norm_head` 目前抛 `NotImplementedError`，不静默回退（`results/REPORT.md` §2.4） |

注意两处「文件值 ≠ dataclass 默认」的历史原因：`free_noise_energy_ratio: 0.02` 是为与
S1 之前的硬编码 `scale=0.02` 逐位一致（`augmented_noise.py:65-74` 的归一化约定使部署
σ 恰等于 energy_ratio）；`signal_noise_coupling_scale` 的 dataclass 默认是保守的 `0.0`，
两个 YAML 显式写 `0.02` 以保持 pre-S1 行为（`config.py:87-91` 注释与
`docs/IMPLEMENTATION_PROMPT_dual_channel.md` S1 增量节）。

## 3. `refresh_mode`：fixed_debug vs per_request

- `fixed_debug`：固定的 ξ 在转换期写入转换后偏置；按 AGENTS.md 只用于正确性测试。
  运行期约束：`debug_enabled=false` 时构造即报错
  （`models/obfuscated.py:87-89`、`:776-778`）；`ChainLinear` 要求提供
  `fixed_refresh`（`layers/linear.py:163-164`）。
- `per_request`：ξ 在融合操作内部由可复现的逐请求、逐层种子生成
  （`augmented_noise.record_generator` 的域分离；`config.py` 不存种子材料）。
  `ChainLinear` 要求传入 `request_context`（`layers/linear.py:222`），且不得带
  `fixed_refresh`（`layers/linear.py:165`）。
- **exact 模式要求**：正确性路径用 `fixed_debug` + `rho_noise_bound: 0`
  （`configs/tiny_exact.yaml:16-18`）。sweep 的 `plaintext`/`exact` 点当前**继承**
  `tiny_approx` 的 `per_request` + ρ=1e-5，不等价于 `tiny_exact`，见 §8。
- 不得把确定性的固定刷新描述为强隐私（AGENTS.md）。

## 4. 三个噪声旋钮：`rho_noise_bound` / `free_noise_energy_ratio` / `signal_noise_coupling_scale`

### 4.1 通道词汇（channel 0 / channel 1）

`noise_budget.py:1-29` 定义 `R^N = ker(L) + row_space(L)` 的显式切分：

- **channel 0**（`ker(L)`，free channel）：与非线性检查点精确交换，精度代价恰为 0，
  但信息论隐藏量也**恰为 0**（theorem 4）；`DualChannelBudget` 在构造时强制
  `accuracy_cost_bound == 0` 且 `information_theoretic_hiding_bits == 0`
  （`noise_budget.py:88-104`）。**永远不得描述为隐私来源。**
- **channel 1**（semantic direction）：每个单位的噪声都精确扰动 `h`（theorem 2），必须
  携带可计算误差界，其来源被白名单限制为 `BOUND_SOURCES = ("margin_delta_k",
  "value_diameter_D_V", "rho")`（`noise_budget.py:42`、`:105-110`）。

账本在转换期由 `DualChannelBudget.from_config`（`noise_budget.py:169`）构造，并在
`converter/save.py:103-104` 校验一致性、在 `:187-201` 写出运行时文件
noise_budget.json。更完整的通道说明见 `docs/INDEX.md`、`docs/mathematics.md` 与
`docs/IMPLEMENTATION_PROMPT_dual_channel.md`。

### 4.2 各旋钮的归属

| 键 | 通道 | 语义 |
|---|---|---|
| `free_noise_energy_ratio` | channel 0 | 部署路径归一化标定下刷新 ξ 的逐分量 σ；只影响数值条件，不影响隐私（`config.py:81-86`；`augmented_noise.deployed_free_noise_sigma`，`augmented_noise.py:65`） |
| `rho_noise_bound` | channel 1 | RMS 标量 ρ 上的有界扰动上界；exact 合同要求 0（`results/REPORT.md` §6） |
| `signal_noise_coupling_scale` | channel 1 邻接 | `e' = h C + e G + xi` 的线性耦合 C；非零使辅助态成为信号的线性像（`docs/PROBLEMS.md` §1.7 记录的泄漏），dataclass 默认 0.0 |

`sources` 字典把 `rho_noise_bound`、`attention.tau_max`、`attention.tau_error`（以及可选的
router tau）折叠为 channel-1 的 `accuracy_cost_bound = max(...)`（`noise_budget.py:199-216`）；
exact 家族若携带非零 channel-1 噪声，账本会改标为
`exact_attention_with_semantic_noise` 而不是 `exact`（`noise_budget.py:210-212`、`:154-167`）。

## 5. `basis_block_size` 整除规则

`PrototypeConfig.__post_init__`（`config.py:213-220`）要求：

```text
(hidden_size + hidden_noise_dim) % basis_block_size == 0
```

tiny 配置：(32 + 8) % 8 == 0 通过；报错信息提示调整 hidden_noise_dim 或
basis_block_size。Value 基不使用块划分：`head_dim + value_noise_dim_per_head` 恒为单个
稠密正交块，因此**不受**该整除约束（`config.py:221-224` 注释）。

## 6. `attention` / `runtime` / `evaluation` 段

### 6.1 `attention`

| 键 | tiny_exact | tiny_approx | 取值域 / 校验 |
|---|---|---|---|
| `mode` | `exact` | `topk_preserving` | `{plaintext, exact, topk_preserving, free_bounded}`（`modes.py:12-18`）；`config.py:136-137` |
| `tau_max` | 0.0 | 0.03 | 有限且 ≥0（`config.py:138-141`）；plaintext/exact 必须为 0（`config.py:146-149`） |
| `tau_error` | 0.0 | 0.10 | 同上；提供固定误差预算那一项 |
| `alpha` | 0.8 | 0.8 | `(0, 1)`（`config.py:142-143`） |
| `preserve_top_k` | 4 | 4 | ≥1（`config.py:144-145`） |

预算公式在 `layers/attention.py:95` 的 `compute_topk_budget`：对有效 logits 排序后
`τ = min(τ_max, α·Δ_k/2, τ_error)`（`attention.py:126-133`）；`free_bounded` 不使用 margin
项。被掩码位置保持精确 `-inf`（`safe_masked_softmax_fp32`，`attention.py:73`）。
`plaintext`/`exact` 强制零噪声界的检查在 `config.py:146-149`。

### 6.2 `runtime`

| 键 | tiny_exact | tiny_approx | 取值域 / 校验 |
|---|---|---|---|
| `seed` | 20260731 | 20260731 | int；决定所有域分离种子 |
| `request_id` | `tiny-exact` | `tiny-approx` | 非空（`config.py:167-168`） |
| `device` | cpu | cpu | `{cpu, mps, cuda}`（`config.py:163-164`） |
| `activation_dtype` | float32 | float32 | `{float32, bfloat16}`（`config.py:165-166`）；BF16 是独立 condition，不替代 FP32 exact 门禁（`results/REPORT.md` §3） |
| `debug_enabled` | `true` | `false` | bool；`fixed_debug` 需要 true（见 §3）；生产模式必须 false |

### 6.3 `evaluation`

| 键 | tiny_exact | tiny_approx | 默认 | 校验 |
|---|---|---|---|---|
| `batch_size` | 2 | 2 | 必填 | >0（`config.py:183-192`） |
| `sequence_length` | 16 | 16 | 必填 | >0；与 model 段交叉校验（`config.py:207-212`） |
| `sample_count` | 8 | 8 | 必填 | >0 |
| `generation_tokens` | 4 | 4 | 必填 | >0；`sequence_length + generation_tokens ≤ max_sequence_length` |
| `warmup_runs` | 1 | 1 | 必填 | ≥0 |
| `timed_runs` | 3 | 3 | 必填 | >0 |
| `bootstrap_replicates` | —（默认） | —（默认） | `1000`（`config.py:181`） | ≥0（`config.py:193-194`） |

## 7. `eval_sweep.yaml` 的键

顶层与 `experiment` 段由 `scripts/run_accuracy_sweep.py:153-164` 读取
（必填：`base_config`、`experiment`、`sweep`、`execution`）：

| 键 | 语义 | 消费点 |
|---|---|---|
| `base_config` | 全部 spec 的基座 `PrototypeConfig` 路径；当前 `configs/tiny_approx.yaml` | `run_accuracy_sweep.py:258-259`、`:374-375` 记录 sha256 |
| `experiment.name` | 实验名 | 计划/record 元数据 |
| `experiment.seed` | sweep 种子声明 | 描述性；当前无代码读取，运行期用 base 的 `runtime.seed`（`run_accuracy_sweep.py` 的 run record `seed` 字段来自 config） |
| `experiment.model_id` / `dataset_id` / `source_revision` | 模型与数据身份 | `evaluation/runner.py:342-375` 的 run record |
| `experiment.output_jsonl` | 默认输出路径（`eval_sweep.jsonl`） | `run_accuracy_sweep.py:321-324` || `experiment.cached_inputs` | 对齐缓存默认路径 | `--dataset-cache` 的默认来源 |
| `experiment.calibration_sample_count` | 校准阶段的样本数（默认 8） | `run_accuracy_sweep.py:261`、`:278-281` |
| `experiment.full_sample_count` | 全量阶段样本数（默认取 base 的 `evaluation.sample_count`） | `run_accuracy_sweep.py:262`、`:278-281` |
| `experiment.selection_rule` | 选择规则说明文本 | 描述性；当前无代码读取该 YAML 值（`select_sweep_candidates` 的 manifest 自带 `selection_rule`，`evaluation/selection.py:328`） |
| `evaluation_override` | 覆盖 `EvaluationConfig` 的字段（未知字段报错） | `evaluation/spec_config.py:52-62` |
| `obfuscation_override` | 覆盖 `ObfuscationConfig` 的字段（未知字段报错）；基础 sweep 显式声明 `rho_noise_bound: 0.0` | `evaluation/spec_config.py:69-79`、`run_accuracy_sweep.py:155-156` |
| `evaluation_notes` | 长度 24 的动机与「valid_count ≤ k 收零噪声」口径说明 | 描述性 |
| `sweep.modes` | 四种模式清单（枚举器实际只固定产 plaintext/exact + topk/free） | `evaluation/sweep.py:44-97` |
| `sweep.tau_max` | 7 个 τ 值 | 同上；幂积出 63 个 topk 点与 7 个 free 点 |
| `sweep.alpha` | 3 个 α 值（topk 专有） | 同上 |
| `sweep.preserve_top_k` | 3 个 k 值（topk 专有） | 同上 |
| `sweep.tau_error` | 统一误差预算（0.10） | 同上 |
| `execution.run_in_current_prototype_pass` | 基础 sweep 完成标记（2026-09-19 完成后置 `true`）；不影响枚举，只在计划元数据与 `deferred_by_config` 里反映 | `run_accuracy_sweep.py:368-369`；`docs/PROBLEMS.md` §3.5 |
| `execution.reason` | 上述标志的原因文本 | 描述性 |

`_config_for_spec`（`evaluation/spec_config.py:28-89`）替换 `attention`/`evaluation`/
`runtime` 三块，并**显式重置 `obfuscation.rho_noise_bound` 为
`BASE_SWEEP_RHO_NOISE_BOUND = 0.0`**（其余 obfuscation 字段继承 base，除非
`obfuscation_override` 另行指定）。这是 §8 的决策落地。

## 8. `eval_sweep.yaml` 的基线口径（PROBLEMS §3.5，已决策 2026-09-20）

1. `execution.run_in_current_prototype_pass: true` 表示基础 sweep 已于 2026-09-19
   完成；枚举器不读 `execution`，重跑仍需显式 `--execute-deferred`。
2. 72 个基础点全部 `rho_noise_bound = 0`：由 `_config_for_spec` 的默认契约强制，
   并在 `eval_sweep.yaml` 的 `obfuscation_override` 中显式声明（两者由
   `tests/evaluation/test_sweep_rho_policy.py` 锁定一致）。`refresh_mode` 保持
   `per_request`（实测不破坏 exact 性，且 `fixed_debug` 不得当隐私路径）。
   部署折衷 ρ=1e-5 的复制点由重跑后的 selection 候选单独生成，不属于这 72 个基础点。
3. 已归档 `eval_sweep*.jsonl` 产生于该口径之前（继承 ρ=1e-5），**重跑后不可与之
   逐点比较**；见 `results/PROVENANCE.md` §7。

## 9. `augmented_noise_kernel.yaml`（documentation-only）

**没有任何 Python 代码读取该文件。** `scripts/run_augmented_noise_kernel.py:53` 在代码内
构造 `KernelConfig`（字段：`signal_dim`、`noise_dim`、`max_condition_number`、`gamma`、
`energy_ratio`、`basis_family`、`seed`），脚本的 `--help` 也只有 `--output-dir`。
该缺陷最初记录于 `docs/archive/2026-09-14_augmented_noise_stage_a_review.md`，
`docs/INDEX.md` §4 与 `docs/CLEANUP_PLAN.md` §8.4 已登记；接线属功能变更，另立项。

其键只作文档语义参考：`basis_family`（`dense_two_sided`/`dense_baseline`）、`hidden_size`
（脚本内对应 `signal_dim`，且脚本用 64 而非 32）、`hidden_noise_dim`（脚本内 `noise_dim`）、
`condition_target`、`max_condition_number`、`activation_dtype`、`refresh_mode`、
`refresh_granularity`、`noise_energy_ratio`、`noise_propagation_gamma`、
`signal_noise_coupling_scale`、`rho_noise_bound`、`attention_mode`、`attacker_views`、
`sweep.{energy_ratios,noise_dims,condition_numbers,basis_families}`、`runtime.{seed,device}`。
**不要把该 YAML 当作已生效的实验配置。**

## 10. 校验来源速查

| 关注点 | 位置 |
|---|---|
| YAML → dataclass 装载 | `config.py:233-254` |
| model 段校验 | `config.py:31-54` |
| obfuscation 段校验 | `config.py:93-123` |
| attention 段校验（含 exact 零界） | `config.py:136-149` |
| runtime 段校验 | `config.py:162-168` |
| evaluation 段校验 | `config.py:183-194` |
| 跨段约束（上下文长度、块整除） | `config.py:207-224` |
| 通道账本一致性 | `noise_budget.py:274-347` |
| sweep spec → config（含 ρ=0 契约） | `evaluation/spec_config.py:28-89` |
| sweep 枚举校验 | `evaluation/sweep.py:44-97` |
| Wilson 区间的权威实现 | `src/fastprove/evaluation/stats.py:43`；`evals/stats.py:19` 仅重导出（PROBLEMS §3.4） |
