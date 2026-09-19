# 架构：生产路径 / 参考路径 / 侧路实验

状态：CURRENT
日期：2026-09-20
上游依据：docs/CLEANUP_PLAN.md §P2、§8.3、AGENTS.md
适用范围：src/fastprove 的路径归类、as-built 模块地图、依赖方向与包边界；脚本索引见 docs/scripts_index.md

本文件回答一个新人最容易踩的问题：**哪些代码是推理时真正执行的，哪些只是验收门与数学
等价性测试的参考实现，哪些是 forward 从不调用的侧路实验。** 第二、三节给出 P8 补齐的
as-built 模块地图与依赖方向；第四节登记三处「看起来可以合并、实际不能合并」的结构。

## 1. 三分表

| 分类 | 模块 | 说明 |
|---|---|---|
| 生产路径（forward 实际执行） | `models/obfuscated.py` | `ObfuscatedDecoderBlock` / `ObfuscatedTinyCausalLM` 的无解码 forward；调用部署权重、SecureEmbedding、Secure LM Head、attention 与 rmsnorm 的混合态算子 |
| | `layers/deployed.py` | `build_deployed_attention` / `build_deployed_feed_forward` 及其 Gram 产物（rho 统计的因子张量） |
| | `layers/embedding.py`、`layers/head.py` | 预混词表嵌入与混淆 LM head（不物化明文 embedding / logits 列） |
| | `layers/attention.py` | `ObfuscatedAttention`、`compute_topk_budget`、噪声采样、`safe_masked_softmax_fp32` |
| | `layers/rmsnorm.py` | RoPE、Q/K 后正交变换、`perturb_rms_scale`、`rms_norm_fp32` |
| | `layers/swiglu.py::generate_swiglu_transform` | 生产 SwiGLU 的神经元置换与缩放生成 |
| | `augmented_noise.py::deployed_free_noise_sigma` | 唯一的生产用噪声标度函数（逐位等于 energy_ratio） |
| | `structured.py` | `StructuredBasis` 因子、Gram 产物、`free_noise_basis`（ker(L) 账本） |
| | `codec.py`、`state.py`、`seed.py`、`config.py` | 词表置换、混合态容器、域名分离种子、配置校验 |
| | `converter/`、`noise_budget.py` | 部署打包与转换期双通道预算账本；`noise_budget` 同时参与转换门 |
| 参考路径（验收门 / 数学等价性测试，forward 不调用） | `layers/linear.py`（`ChainLinear`） | 一般增广仿射恒等式；被 `tests/math/test_chain_linear.py`、`tests/layers/test_deployed_matches_chain_linear.py` 使用 |
| | `conversion.py` | `convert_affine_chain` 的离线参考转换（生产融合见 `layers/deployed.py`） |
| | `transforms.py` 的 dense 族 | `generate_transform` / `generate_two_sided_transform` 仅用于验收门、等价性测试与侧路核实验；生产使用 `generate_orthogonal` / `generate_signed_permutation` |
| | `layers/moe.py`（`ObfuscatedMoE`）、`layers/router.py`（`StableRouter`） | 专家路由与 MoE 的正确性参考；当前 tiny LM 生产 forward 未接入 |
| | `layers/swiglu.py::convert_swiglu_weights`、`refresh_swiglu_noise` | SwiGLU 协变与刷新参考；测试专用 |
| 侧路实验（独立脚本，不接入推理） | `augmented_noise.py` 其余函数 | 采样、传播子、`record_generator`、覆盖度指标；服务侧路核实验与单元测试 |
| | `scripts/run_augmented_noise_kernel.py` 及 `results/raw/augmented_noise_kernel/` | 噪声链平稳性/覆盖度实验，不是推理路径 |
| | `scripts/verify_*.py`、`scripts/record_*.py` | 一次性探针与记录脚本（自包含单文件约定，见 docs/implementation_notes.md 与 docs/scripts_index.md） |

## 2. as-built 模块地图

以 2026-09-20 的工作树为准（`converter` 已拆包；setup.py 已由 P1 移除，元数据只在
`pyproject.toml`）。每个文件的一行职责来自其模块 docstring。

```text
src/fastprove/
├── __init__.py                  # 包 docstring：不做 re-export，避免传递性依赖
├── seed.py                      # 域分离种子：derive_seed / make_generator / RequestContext
├── hashing.py                   # 分块 sha256_file（7 处重复实现统一到此处）
├── io.py                        # resolve_path / write_json_file
├── metrics.py                   # tensor_error_metrics（无依赖，修 P3 前的 models→evaluation 倒置）
├── policy.py                    # AUXILIARY_MAGNITUDE_BOUND=30.0、check_auxiliary_magnitude
├── modes.py                     # AttentionMode 四值枚举
├── basis_common.py              # BasisDescriptor、dtype 校验、指纹原语
├── bounded_noise.py             # 位混合原语：mix_coordinates / mantissa_to_signed_unit
├── coa_views.py                 # COAViewContract 字符串契约（不是隔离机制）
├── transforms.py                # 变换生成：正交/带符号置换（生产）与 dense 族（参考）
├── structured.py                # StructuredBasis M = Π1 D B Π2、Gram、ker(L)
├── state.py                     # MixedState 与显式门禁的 encode_debug/decode_debug
├── codec.py                     # TokenCodec 词表置换
├── config.py                    # 配置 dataclass 与 YAML 校验（load_config）
├── conversion.py                # 行向量数学布局 ↔ PyTorch 布局；convert_affine_chain（参考）
├── augmented_noise.py           # A 套噪声链；仅 deployed_free_noise_sigma 属生产
├── noise_budget.py              # DualChannelBudget / ChannelBudget（S2 账本）
├── converter/                   # 部署打包包（原单文件 converter.py，P3e 拆分）
│   ├── __init__.py              # 兼容层：保留旧单文件的顶层 re-export
│   ├── hygiene.py               # 服务端 bundle 安全闸门与 ConversionManifest
│   ├── load.py                  # load_server_model：仅凭服务端 bundle 重建可运行模块
│   ├── client.py                # ClientBundle / load_client（客户端秘密侧）
│   ├── save.py                  # ModelConverter：写 server/client 两个目录
│   └── basis_io.py              # 基因子序列化与确定性重建
├── layers/
│   ├── attention.py             # 精确/近似注意力 FP32 核；compute_topk_budget；有界噪声
│   ├── rmsnorm.py               # FP32 RMS 统计、RoPE、Q/K 后正交变换、rho 扰动
│   ├── deployed.py              # 部署权重离线融合（生产）
│   ├── linear.py                # ChainLinear（参考路径）
│   ├── swiglu.py                # SwiGLU 置换/缩放协变与刷新（生成器属生产）
│   ├── router.py                # StableRouter 与 margin-bounded 路由（参考）
│   ├── moe.py                   # ObfuscatedMoE（参考）
│   ├── embedding.py             # SecureEmbedding：预混词表查找
│   └── head.py                  # Secure LM head：final-norm 吸收 + 词表列置换
├── models/
│   ├── plain.py                 # PlainDecoderBlock / PlainTinyCausalLM（明文参考）
│   ├── obfuscated_types.py      # 共享 dataclass：KV cache、debug 容器、Converted*
│   ├── client.py                # ObfuscatedBlockClient：隐藏混合变换持有者
│   ├── block_builder.py         # BlockDeploymentMixin：离线融合权重装配
│   └── obfuscated.py            # ObfuscatedDecoderBlock / ObfuscatedTinyCausalLM（生产）
├── pretrained/
│   ├── _checkpoint_io.py        # 共享 checkpoint I/O：require_file/load_json/target_tensors
│   ├── llama.py                 # LlamaForCausalLM 适配器（llama3 rope_scaling 未实现）
│   └── qwen2.py                 # Qwen2ForCausalLM 适配器（含 Q/K/V bias 与 GQA）
└── evaluation/
    ├── environment.py           # 运行环境元数据
    ├── runtime.py               # 设备选择与显存释放
    ├── diagnostics.py           # exact 比对的首个失配诊断
    ├── exact_gate.py            # exact 门禁与容差包络
    ├── layer_metrics.py         # 层/块级注意力与 Softmax 误差指标（写入 run record.metrics）
    ├── performance.py           # 延迟/吞吐/内存/KV cache 测量
    ├── spec_config.py           # sweep spec → PrototypeConfig 解析
    ├── runner.py                # tiny 评测执行原语（run_tiny_sweep_spec）
    ├── metrics.py               # 显式数值正确性/Softmax 诊断
    ├── accuracy.py              # teacher-forced / greedy 对齐精度
    ├── artifacts.py             # run record schema 与 JSONL 写入
    ├── sweep.py                 # 72 点枚举（enumerate_sweep_specs）
    ├── selection.py             # 校准 JSONL → 候选 manifest
    ├── token_cache.py           # 对齐 token cache 的构建/校验
    ├── pretrained.py            # 本地预训练资产审计
    ├── attack_records.py        # fastprove.attack.v1 成本记录
    ├── correctness.py           # `run_correctness` 入口
    ├── prompts.py               # prompt 语料装载
    ├── benchmarks/{tasks,scoring,bundle}.py  # OSNIP 风格 harness（P5 从脚本迁入）
    └── report*.py               # report/build_report 及其 format/rows/tables/figures/markdown/tradeoff 拆分
```

## 3. 模块职责与允许的导入方向

依赖方向是单向的：**上层的模块可以 import 下层，下层不得反向 import**。下表是每个模块
的归属与主要被依赖面；结论均在 2026-09-20 用 `rg` 对 `src/fastprove` 全树核对过。

### 3.1 依赖分层

```text
L0 叶子（无 fastprove 内部 import）
  seed  hashing  io  metrics  policy  modes  basis_common  bounded_noise  coa_views
L1 基础构件
  transforms → basis_common, seed
  state      → transforms
  codec      → seed
  config     → modes, policy
  conversion → transforms
  augmented_noise → seed, transforms
  structured → basis_common, policy, seed
  noise_budget → config, modes, structured
L2 层与模型
  layers/*  → L0/L1（含 layers/* 内部互引；不 import models）
  models/*  → layers/, L0/L1（不 import evaluation、不 import pretrained）
L3 适配与评测
  pretrained/* → config, hashing, models.plain, pretrained/_checkpoint_io
  evaluation/* → pretrained/, models/, layers/, L0/L1（单向下沉）
L4 部署写手与第二套 harness
  converter/* → layers/, models/, noise_budget, structured, config（models/layers 不反向）
  evals/*     → fastprove（反向仅 2 个现行脚本 + 1 个归档脚本，见 §3.3）
```

核对命令与结论（原样保留）：

- 叶子模块：`rg -n "from \.|import fastprove" src/fastprove/{seed,hashing,io,metrics,policy,modes,basis_common,bounded_noise,coa_views}.py` → 无命中。
- `layers` 不得 import `models`：`rg -n "from (\.\.|fastprove)\.?models|import models" src/fastprove/layers` → 无命中。
- `models` 不得 import `evaluation`：`rg -n "from (\.\.|fastprove)\.?evaluation|import evaluation" src/fastprove/models` → 无命中。
- `models` 不得 import `pretrained`：`rg -n "pretrained" src/fastprove/models` → 无命中。

### 3.2 各模块 owns 什么

| 模块 | owns（主要公共对象） |
|---|---|
| `seed.py` | `derive_seed`、`make_generator`、`RequestContext`；所有种子派生的唯一来源 |
| `hashing.py` | `sha256_file`（分块）；文件身份记录 |
| `io.py` | `resolve_path`、`write_json_file`；脚本 JSON 落盘惯例 |
| `metrics.py` | `tensor_error_metrics`；无第三方依赖的误差指标，供 models/scripts 复用 |
| `policy.py` | `AUXILIARY_MAGNITUDE_BOUND = 30.0`（`policy.py:21`）、`check_auxiliary_magnitude`、`REDUCTION_COMPUTE_DTYPE` |
| `modes.py` | `AttentionMode` 四模式（`modes.py:12`） |
| `basis_common.py` | `BasisDescriptor`、`validate_basis_dtype`、指纹原语；所有基族的共同契约 |
| `bounded_noise.py` | 坐标稳定位混合原语；三种 centering 策略保留在调用方（attention/rmsnorm/router） |
| `coa_views.py` | `COAViewContract` 的攻击者输入白名单**字符串契约**与 manifest 落盘 |
| `transforms.py` | `generate_orthogonal`、`generate_signed_permutation`（生产）；`generate_transform`、`generate_two_sided_transform`（参考） |
| `structured.py` | `StructuredBasis`（`M = Π1 D B Π2`）、`gram_rms_scale`、`free_noise_basis`（`structured.py:254`）、`channel_decomposition`（`structured.py:287`） |
| `state.py` | `MixedState`、显式 `enabled=True` 的 `encode_debug`/`decode_debug` |
| `codec.py` | `TokenCodec` 与词表置换生成 |
| `config.py` | `PrototypeConfig` 及各段 dataclass、`load_config`（`config.py:233`） |
| `conversion.py` | 数学布局 ↔ PyTorch 布局转换、`convert_affine_chain`（参考路径） |
| `augmented_noise.py` | A 套噪声链；生产只调用 `deployed_free_noise_sigma`（`augmented_noise.py:65`） |
| `noise_budget.py` | `ChannelBudget`/`DualChannelBudget`、`BOUND_SOURCES`（`noise_budget.py:42`）；转换期通道账本 |
| `converter/` | `ModelConverter`（`converter/save.py`）、`load_server_model`（`converter/load.py`）、`ClientBundle`（`converter/client.py`）、`ConversionManifest`（`converter/hygiene.py`）、`basis_io` |
| `layers/` | P2/P3 生产算子与参考实现（见 §1 三分表） |
| `models/` | 明文与混淆模型、共享 dataclass、客户端侧 block、离线融合 mixin |
| `pretrained/` | Llama/Qwen2 本地 checkpoint 适配与共享 I/O |
| `evaluation/` | 从 run record schema 到报告、sweep、门禁、性能、攻击成本记录的完整评测面 |

各文件对应的 `REFERENCE PATH` / `侧路实验` banner 与 §1 三分表一一对应；删除这些文件会
丢失验收门，因此只标注、不删除。

### 3.3 `evals/` 与 `src/fastprove/evaluation/` 的关系（P6）

- **两套并存是既定决策**（CLEANUP_PLAN §10-①，选项 A+C）：`src/fastprove/evaluation/` 是
  原型自带评测与报告流水线；`evals/` 是第二套独立 harness（P2 协议门禁
  `python -m evals --check-gates` 走它）。
- **依赖方向单向**：`evals/*` 依赖 `fastprove`（例：`evals/model_factory.py:25-33` 从
  `fastprove.config` / `fastprove.models.obfuscated` / `fastprove.pretrained.qwen2` 取件）；
  反向伸手的现行脚本只有 2 个：`scripts/run_pretrained_compare.py:127-129,362`、
  `scripts/run_osnip_style_benchmarks.py:168-169`，另有归档脚本
  `scripts/archive/postprocess_llama_report.py:9-10`。
- **打包**：P6a 已把 `evals` 纳入 `pyproject.toml:30-35`（`include = ["fastprove*", "evals*"]`
  与 `evals = ["configs/*.yaml"]`），`PYTHONPATH=src:.` 的隐式要求不再是唯一路径。
- **指标定义**：P6b 新增逐位一致测试 `tests/evaluation/test_evals_metric_parity.py`，让
  `evals/metrics_common.py` 单向复用 `fastprove.evaluation.metrics`。
- **已知例外（未统一，见 `docs/PROBLEMS.md` §3.4）**：Wilson 区间仍有两份实现——
  `evals/stats.py:47`（`z = 1.959963984540054`，查表于 `evals/stats.py:112`）与
  `src/fastprove/evaluation/benchmarks/scoring.py:18`（默认 `z = 1.959963985`）。
  在显式决策前**禁止**假设两者逐位一致，也禁止在重构里静默统一。

## 4. 三处「平行但不合并」的结构

这些结构看起来重复，实际语义不同；CLEANUP_PLAN P4 已明确**不合并**，只在此说明。

### 4.1 greedy 解码循环（plain vs obfuscated）

- 明文：`models/plain.py:395-439` 的 `PlainTinyCausalLM.generate_greedy`，cache 是
  `PlainKVCache`（`plain.py:30`），只存原始 K/V、position 与 valid mask（`plain.py:194-198`
  的 `torch.cat` 增量拼接）。
- 混淆：`models/obfuscated.py:1133-1180` 的 `ObfuscatedTinyCausalLM.generate_greedy`，
  多一个必填的 `request_context`，cache 是 `ObfuscatedKVCache`，携带
  conversion/model/layer 身份与 request seed/id，并在续算前校验
  （`obfuscated.py:504-513`），Value 侧缓存的是混合 Value 而非明文 V。

两者循环骨架相同是有意的：任何"提取共同 helper"的尝试都会把 request 身份与混合 Value
语义塞进明文路径，或把明文缓存契约泄漏给混淆路径。保持平行结构、分别演进。
出处：CLEANUP_PLAN P4 表末项；`docs/implementation_notes.md:530`。

### 4.2 `coa_views.py` 是字符串契约，不是隔离机制

`COAViewContract`（`coa_views.py:1`）只声明每个威胁视图允许/禁止的输入清单并落盘
manifest；它**不执行**任何访问控制。真正的隔离来自部署形态（服务端只有融合权重，见
`converter/` 的两个目录约定）与调试 API 门禁（`state.py` 的 `enabled=True`）。
把契约文件当成安全边界是误读。

### 4.3 三份 bounded-noise 采样器与两份 top-k margin 预算

- 采样器：`layers/attention.py`（按 valid mask 去均值）、`layers/rmsnorm.py`（故意不去
  均值，见其文件内注释）、`layers/router.py`（按全部 expert 去均值，逐坐标 `seed_for`）。
  只共享 `bounded_noise.py` 的位混合原语，三种 centering 策略保持显式。
- top-k 预算：`layers/attention.py:95` 的 `compute_topk_budget`（只看 valid positions、
  用第 k/(k+1) 个 valid margin、返回 `min(tau_max, alpha*margin/2, tau_error)`）与
  `layers/router.py` 的 `_margin_and_tau`（对未掩码 canonical expert logits 排序、margin
  截断到 0、非正 margin 给零预算）语义不同，其 docstring 已注明不得合并。

## 5. 相关文档

- 脚本分类、调用与产物：`docs/scripts_index.md`；
- 每个 YAML 键与校验来源：`docs/config_reference.md`；
- claim → raw 工件 → 命令：`results/PROVENANCE.md`；
- 验证阶梯与阻塞：`docs/runbook.md`；重构进度：`docs/STATUS.md`。
