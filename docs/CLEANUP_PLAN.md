# 代码结构梳理与文档重建计划

状态：草案，待决策项见 §10
日期：2026-09-19
适用范围：整个仓库（`src/`、`scripts/`、`evals/`、`tests/`、`docs/`、`configs/`、`results/`）
上游依据：`AGENTS.md`（规则与验收门）、`docs/IMPLEMENTATION_PROMPT_dual_channel.md`（当前设计与 T1–T10 结论）

---

## 1. 目标与非目标

**目标**：产出一份结构清晰、分层正确、可被新人读懂的代码库，以及一套与代码一致、带生命周期标注的文档。**数值与行为逐位不变。**

**非目标（本次一律不做）**：

- 不改任何数学定义、噪声构造、可交换族、门限或 τ/α/top-k 语义；
- 不新增或删除推理模式；
- 不做性能优化（AGENTS.md：正确性通过前不优化）；
- 不重跑实验、不下载 checkpoint、不生成新的 `results/raw` 数值记录（唯一例外是 §3 的等价性基线文件）；
- 不修改任何对外 claim（`docs/threat_model.md`、`work_factor_security.md`、`PROBLEMS.md` 的结论原样保留）；
- 不"顺手修 bug"。发现的缺陷只记录进 `docs/PROBLEMS.md` 候选清单，单独立项。

---

## 2. 现状事实基线（实测，不是估计）

| 维度 | 事实 |
|---|---|
| `src/fastprove` | 40 个 `.py`，13,284 行。最大 5 个：`evaluation/runner.py` 1549、`models/obfuscated.py` 1476、`evaluation/report.py` 1348、`converter.py` 607、`structured.py` 560 |
| `scripts/` | 28 个 `.py`。最大 3 个共 ~3300 行，其中约 1900 行是可复用库逻辑、1290 行是 `main`+argparse |
| `evals/` | 独立的第二套评测 harness，15 模块 4399 行，**未被打包**（`pyproject` 只打包 `src`），靠 `PYTHONPATH=src:.` 运行，但确实在用（`results/REPORT.md` 的 exact-gate 命令走它） |
| `tests/` | 45 个文件、382 个测试、11.9s 全绿；无 `conftest.py`、无 marker |
| 包安装状态 | `python -c "import fastprove"` **失败**。测试只在 `python -m pytest`（cwd 进 `sys.path`）下能跑；裸 `pytest` 收集报错 |
| 环境 | Python 3.13.12 / torch 2.11.0 / macOS arm64 / CPU+MPS，无 CUDA |
| Git | 工作区干净。`__pycache__`/`egg-info`/`.DS_Store` 全部未被跟踪（只需 `rm`，不需 `git rm`）；**被跟踪的垃圾只有 11 个 LaTeX 中间文件** |
| 跟踪体积 | 75 MB，其中 72 MB 是 43 个 PDF（38 篇文献）；`.git` 59 MB |
| 文档 | 18+ 份重叠设计文档，仅 2 份带 supersession banner；`README.md` 严重过期；`docs/mathematics.md` 描述的是 7 月的构造 |
| 结果 | `results/raw/` 58 个文件被跟踪，但 `*.jsonl` 被 `.gitignore` 排除——而 `REPORT.md` 正引用其中的 `eval_sweep.jsonl`，新克隆无法复核头条数字 |

---

## 3. 等价性保障机制（本计划的地基，必须先做）

"实质没变"不能靠人眼 review 保证。重构前先建立冻结基线，之后**每一步提交都必须通过**。

### 3.1 两级等价

- **L1 数值等价（强制，逐位）**：所有张量以 `sha256(tensor.cpu().contiguous().numpy().tobytes())` 比对，必须完全相同。
- **L2 产物等价（强制，结构级）**：生成的 JSON/JSONL 记录的字段集合、类型、数值必须相同；允许字段顺序差异，不允许键增删。

### 3.2 新增 `scripts/verify_refactor_equivalence.py`

固定种子、固定 `configs/tiny_exact.yaml` + `configs/tiny_approx.yaml`，覆盖并输出 digest：

1. `StructuredBasis` 矩阵、descriptor、condition number、`ker(L)` 基；
2. `transforms` 两族生成结果（`generate_transform` / `generate_two_sided_transform`）；
3. `build_deployed_attention` / `build_deployed_feed_forward` 全部输出权重与 Gram 产物；
4. 单块 `plaintext` / `exact` / `topk_preserving` / `free_bounded` 输出 + ρ 场 + 每层刷新噪声样本；
5. tiny LM logits、greedy token 序列、KV-cache 单步 vs 全序列；
6. `StableRouter` / `ObfuscatedMoE` 决策与门控权重；
7. `ChainLinear` 仿射恒等式残差；
8. `ModelConverter.convert` 产出的 manifest、`noise_budget.json`、server state_dict 键集合；
9. `evaluation/metrics`、`accuracy`、`artifacts`、`report._rows` 在固定输入上的输出。

输出写 `results/raw/refactor_baseline_<git_sha>.json`（**纳入跟踪**，它是本次重构的验收证据）。

### 3.3 每步门（gate）

```
python -m pytest tests -q                              # 必须 382 passed（或与基线一致）
python scripts/verify_refactor_equivalence.py --check   # 必须 L1/L2 全等
```

任一失败 → 回滚该步，不允许"调整基线让它通过"（AGENTS.md：不得为让测试通过而改数学定义）。

---

## 4. 高危清单：会静默改数的操作

这些是本次重构最可能翻车的地方，实施时逐条确认。

1. **RNG 抽样顺序 / 形状 / 设备**。把 `torch.rand`、`make_generator` 调用移位、合并、改形状或改 device，都会改变后续全部噪声。拆分函数时必须保持抽样的**次序、次数、形状**完全一致。
2. **`derive_seed` 的 domain 元组字符串**。它是密钥派生输入，改名即改数。重命名模块/函数时不得连带改 domain 字面量。
3. **三个 bounded-noise 采样器不可合并**：`layers/attention.py:157`（按 valid mask 去均值）、`layers/rmsnorm.py:46`（**故意不去均值**，见其 L28-31 注释）、`layers/router.py:142`（按全部 expert 去均值，且逐坐标 `seed_for`）。只能抽出共享的位混合原语，三种 centering 策略必须保留为显式命名策略。
4. **`compute_topk_budget`（`attention.py:99`）与 `StableRouter._margin_and_tau`（`router.py:118`）掩码语义不同**，不可直接统一。
5. **Gram `rms_scale` 三处复制**（`structured.py:199/239`、`models/obfuscated.py:161`、`layers/moe.py:397`）合并时，`einsum` 表达式、`clamp_min`、`/d`、`+eps`、`sqrt` 的顺序必须逐字保持，否则 FP32 归约结果会变。
6. **FP64 生成 → 存储 dtype 的 cast 位置**不可挪动。
7. **被测试断言的错误字符串**。拆分校验逻辑时保留原文。
8. **`src/fastprove/evaluation/report.py:1337-1339` 把三条命令名内嵌进生成的 `REPORT.md`**。重命名 `run_correctness.py` / `run_accuracy_sweep.py` / `build_report.py` 会静默改变报告正文 → 本次**不重命名任何脚本**。
9. **安全闸门原样搬运**：`converter.py:53-108`（`_FORBIDDEN_SERVER_KEY_LEAF_TOKENS` + `_assert_server_bundle_is_clean`）、`state.py:49-52`（`_require_debug`）、`models/obfuscated.py:635-645`（`_debug_unmix`）、各 `forward_debug` 的 `debug_enabled` 守卫。另外 `converter.py:277` `ClientBundle(debug_enabled=True)` 默认开启是**现有设计**，本次不动（要动就是行为变更，单独立项）。
10. **测试内的路径常量**：`tests/test_cli_plan.py:25`、`tests/test_evaluation_runner.py:27` 用 `parents[1]`，移入子目录后需改 `parents[2]`。
11. **跨包私有符号引用**：`tests/test_selection.py:6` 从 `tests.test_report_integrity` import；`tests/test_cli_plan.py:10-15` 从 `scripts.run_accuracy_sweep` import 私有符号；`tests/test_artifacts.py:15` 从 `evaluation.report` import `_series_groups`。移动前必须先建立替代来源。

---

## 5. 阶段计划

每阶段结束都跑 §3.3 的门。提交纪律见 §9。

### P0 建立基线（前置，不动任何现有代码）

- [ ] 修包可导入性：`pip install -e .`（或在 `pyproject` 的 `[tool.pytest.ini_options]` 加 `pythonpath = ["src", "."]` 作为兜底）。这是 13 处 `sys.path` 补丁和裸 `pytest` 失败的根因。
- [ ] 新增 `scripts/verify_refactor_equivalence.py`，冻结 `results/raw/refactor_baseline_<sha>.json`。
- [ ] 记录基线：`pytest -q --durations=25` 输出、382 计数、当前 skip 情况（本机 MPS 可用 → MPS 测试实跑；Linux/CPU 上会静默少跑 4 个）。

**产出**：等价性工具 + 基线文件。**门**：基线自比对通过。

### P1 卫生与打包（零代码移动）

- [ ] 删除未跟踪垃圾：根/`results/`/`docs/literature-review/` 的 `.DS_Store`、`UNKNOWN.egg-info/`、`src/fastprove.egg-info/`、各 `__pycache__/`、`.mypy_cache/`、`.pytest_cache/`。
- [ ] `git rm --cached` 11 个 LaTeX 中间文件（`results/figures/*.{aux,log,out}`），`.gitignore` 补规则。
- [ ] `.gitignore` 决策落地（见 §10-②）：`results/raw/*.jsonl` 的排除与 AGENTS.md「报告结论必须可从仓库产物复现」直接冲突。
- [ ] `setup.py` 处置（见 §10-⑦）。
- [ ] `results/figures/测评结果.pdf` 改 ASCII 名。
- [ ] `.workbuddy/memory/` 决定 ignore 还是跟踪。

**门**：`pytest` 与等价性双绿；`git status` 干净。

### P2 死代码清理与分层标注（低风险）

确认死代码，删除：

- `layers/attention.py:137` `_coordinate_value`（零调用点）
- `evaluation/selection.py:20` `REQUIRED_BASELINE_MODES`（定义未读）
- `structured.py:510` `auxiliary_magnitude_bound`（仅出现在 docstring）
- `scripts/record_exact_tolerance.py:31` 对 `_checkpoint_compute_dtype` 的失效 import（该符号全仓不存在，被 `except ImportError` 吞掉）
- 待复核：`layers/swiglu.py:127` `refresh_swiglu_noise`

**不删但必须打标签**（它们支撑验收门，删了会丢门）：`layers/linear.py` 的 `ChainLinear`、`conversion.py`、`transforms.py` 的 dense 族、`augmented_noise.py`（除 `deployed_free_noise_sigma`）、`layers/moe.py` 的 `ObfuscatedMoE`、`layers/swiglu.py` 的 `convert_swiglu_weights`。做法：模块顶部加统一 banner，例如

```
REFERENCE PATH — 参与验收门与数学等价性测试，部署 forward 不调用本模块。
生产路径见 layers/deployed.py。
```

并在 `docs/architecture.md` 给出「生产路径 / 参考路径 / 侧路实验」三分表。

- [ ] 5 个从未被 import 的 `__init__.py` re-export facade（根、`layers`、`models`、`evaluation`、`pretrained`，共 84 行）：二选一——要么正式定为公共 API 并让消费者改用，要么清空 re-export 以消除强制的传递性 import。推荐后者（更小改动），但需同步修 §P3 的分层违规。

**门**：双绿。

### P3 大文件按既有缝拆分（纯移动，不重写）

规则：**只移动，不改写**；旧模块保留 re-export 以免破坏 `tests`/`scripts` 的 import；每拆一个文件跑一次门。

| 源 | 目标 | 依据行段 |
|---|---|---|
| `evaluation/runner.py` 1549 | `environment.py`（54-102）、`runtime.py`（103-168）、`diagnostics.py`（261-322）、`exact_gate.py`（323-472，零 torch/model 耦合，最干净的缝）、`layer_metrics.py`（473-598）、`performance.py`（599-977）、`spec_config.py`（978-1017）；`runner.py` 只留 `_execute_tiny`+`run_tiny_sweep_spec` | 6 个互不耦合的关注点 |
| `evaluation/report.py` 1348 | `report_format.py`（28-78，5 个纯函数服务 120+ 调用点）、`report_rows.py`（79-574，496 行纯 schema 映射）、`report_tradeoff.py`（600-719）、`report_figures.py`（786-917，唯一 matplotlib 耦合）、`report_tables.py`（978-1248）、`report_markdown.py`（1249-1348）；`build_report` 收缩到 ~40 行编排 | 已实测的内部分段 |
| `models/obfuscated.py` 1476 | `obfuscated_types.py`（84-134 的 dataclass）、`client.py`（50-83）、`block_builder.py`（416-590，源码已有 `# -- deployed attention weights (offline fusion)` 预标缝）、`obfuscated.py` 留 runtime forward | 三个关注点被熔在一起 |
| `converter.py` 607 | `converter/` 包：`hygiene.py`（53-109 安全闸门）、`load.py`（110-268）、`client.py`（269-317）、`save.py`（318-544，L498 已有 `# -- client bundle` 预标缝）、`basis_io.py`（545-607） | 命名与 `conversion.py` 冲突，拆包同时消歧 |
| `structured.py` 560 | 抽出 `policy.py`（507-560 的预算/dtype 常量）——**这一步同时解掉 `config.py → structured.py` 的分层倒置**；可选 `structured_generate.py`（377-506） | |

顺带修分层违规：

- [ ] `models/obfuscated.py:17` `from ..evaluation.metrics import tensor_error_metrics` → 把 `tensor_error_metrics` 移到无依赖的 `fastprove/metrics.py`，或让 `forward_debug` 返回原始张量、由调用方算指标。
- [ ] `config.py:12-13`、`noise_budget.py:36-38` 对 `layers/attention` 与 `structured` 的反向依赖 → `AttentionMode` 与 `AUXILIARY_MAGNITUDE_BOUND` 迁入中立模块（`policy.py` / `modes.py`）。
- [ ] `converter.py` 内 8 处函数内重复 import（`:279/:303/:330/:331/:573/:590/:178/:184`）提到模块级。

**门**：双绿（这是全计划最需要频繁跑门的阶段）。

### P4 去重——只做可证等价的部分

| 重复 | 处置 |
|---|---|
| `_sha256_file` **7 份**（`pretrained/llama.py:42`、`pretrained/qwen2.py:25`、`evaluation/pretrained.py:89`、`evaluation/runner.py:176`、`scripts/run_accuracy_sweep.py:157`、`scripts/extract_flickr_captions.py:19`、`converter.py:110` 非分块变体） | 统一到 `fastprove/hashing.py`，保留分块/非分块两个语义 |
| `transforms.py` vs `structured.py` 的 `_validate_dtype`/`_fingerprint`/`descriptor`/`total_dim`/`validate_integrity` | 抽 `basis_common.py`；错误信息逐字保留 |
| `generate_transform` vs `generate_two_sided_transform`（同文件 69% 相同），且后者把 `generate_orthogonal` 的体内联了两次（`:272-279`） | 抽 log-对角生成与 condition 检查助手；**抽样调用次序与形状不得变** |
| Gram `rms_scale` **3 份** | 合并到 `structured.py` 的单一实现，宽度断言以参数保留；见 §4-⑤ |
| `pretrained/llama.py` vs `qwen2.py`（57.5% 相同，9 段 ≥8 行逐字相同） | 抽 `pretrained/_checkpoint_io.py` |
| `runner.py:624-785` vs `:786-886`（plain/obf 性能测量 57.8% 相同） | 参数化共同骨架，两个特化保留 |
| bounded-noise 采样器 3 份 | **只抽位混合原语**，三种 centering 保留（§4-③） |
| top-k margin 预算 2 份 | 本次**不合并**，只补注释说明语义差异（§4-④） |
| `plain.py:395-439` vs `obfuscated.py:1429-1476` greedy 循环 | 本次**不合并**（cache 语义不同），只在文档里说明平行结构 |

**门**：双绿 + 每项合并附带一个「新旧实现逐位相同」的临时断言（合并落地后删除临时断言，保留基线比对）。

### P5 scripts 库化（保留脚本文件名与 CLI 表面）

约束：脚本名不改（§4-⑧）；`docs/IMPLEMENTATION_PROMPT_dual_channel.md:76-78` 规定攻击/探针脚本为**自包含单文件、无共享 util** → 7 个 `verify_*.py` 与 `record_*.py` 本次**不动**，只在 `docs/scripts_index.md` 归类说明。

| 源 | 迁入 |
|---|---|
| `run_osnip_style_benchmarks.py:184-1021`（8 个数据集 loader + 打分引擎，约 75% 是可复用 harness） | 新建 `fastprove/evaluation/benchmarks/{tasks.py,scoring.py,bundle.py}` |
| `run_pretrained_compare.py:190-513`（`greedy_unpadded_pair`、215 行流式指标引擎 `_run_pair`） | `evaluation/accuracy.py` 邻位；prompt 装载 → `evaluation/prompts.py` |
| `run_accuracy_sweep.py:291-481`（两个 run-record 构造器）、`_spec_payload`/`_order_specs_for_exact_gate`/`_load_selected_run_ids` | 前者 → `evaluation/artifacts.py`；后者 → `evaluation/sweep.py`（现仅 94 行） |
| `_env_meta` 2 份（`run_osnip:96`、`run_pretrained_compare:65`） | 统一到 `evaluation/environment.py`（P3 已建） |
| `_resolve` 3 份、JSON/JSONL 写入 ~20 处同一惯用法 | `fastprove/io.py`：`write_json_record` / `append_jsonl` / `resolve_path` |
| 13 处 `sys.path` 引导（两种不兼容写法） | P0 的可编辑安装后全部删除，只在 README 留一条 fallback 说明 |
| `_wilson_interval`（`run_osnip:322`）与 `evals/stats.py:wilson_interval` 重复，而同文件已 import `evals.stats` | 删本地副本，用 `evals.stats` |

同步处理 `tests/test_cli_plan.py` 与 `tests/test_pretrained_compare_cli.py` 对脚本私有符号的依赖：**同一提交内**改为从新位置 import。

**门**：双绿。另需人工确认 `README.md`/`docs/experiment_plan.md`/`results/REPORT.md` 里的命令行仍逐字有效。

### P6 `evals/` 归位（需决策，见 §10-①）

事实：`evals` 依赖 `fastprove`，反向无依赖；`scripts/run_pretrained_compare.py:59-62`、`run_osnip_style_benchmarks.py:58-60`、`postprocess_llama_report.py:9-10` 反向伸手进 `evals` 取共享件；`evals/` 内**零 `def test_`**，`testpaths=["tests"]` 使 `evals/conftest.py` 的 6 个 fixture 完全不被加载——而 `tests/test_eval_boundaries.py:22-28`、`tests/test_pretrained_compare_cli.py:20-21` 正在手工重建同样的对象。

推荐方案（不改行为的最小动作）：

- [ ] 把 `evals` 纳入打包或加显式路径 shim，消除 `PYTHONPATH=src:.` 的隐式要求；
- [ ] 先加「两套指标实现逐位一致」测试（`evals/metrics_common.py` 的 `kl_divergence`/`js_divergence`/`topk_set_overlap`/`max_abs_error`/`relative_l2_error` vs `fastprove/evaluation/metrics.py`），**通过后**再让 `evals` 单向复用 `fastprove` 的定义。未通过则两者都保留并在文档中写明差异（那意味着历史结果基于不同定义，属实质问题，需升级为 PROBLEMS 条目）；
- [ ] `evals/conftest.py` 的 fixture 迁到可被 `tests/` 使用的位置；
- [ ] `scripts/postprocess_llama_report.py`（无 argparse、硬编码路径、会整体覆盖 `results/REPORT.md`、零引用、跨包 import 私有符号且无路径引导）→ 归档到 `scripts/archive/` 或删除（见 §10-④）。`scripts/write_chinese_report.py`（第三套独立报告写手）同理。

**门**：双绿。

### P7 测试重组

顺序很重要：先补 `conftest.py`，再移动。

- [ ] 新建 `tests/conftest.py`，收纳当前被复制的 fixture：
  - run-record metric payload（4 份：`test_report_integrity.py:19`、`test_report_cli_coverage.py:20`、`test_cli_plan.py:28`、`test_run_record_schema.py:15`，其中两份逐字相同）
  - run-record 构造器（5 个变体）——**必须先把 `_success_record` 提到 conftest**，否则 `tests/test_selection.py:6` 一移就断
  - `ModelConfig`/`ObfuscationConfig` 工厂（重复 8 次，魔法数 `hidden_noise_dim=8`/`value_noise_dim_per_head=2`/`max_condition_number=10.0`/`noise_propagation_gamma=0.5`/`basis_block_size=8`）
  - `plain → obfuscated` 转换助手（6 份近似相同）
  - `generate_structured_basis` 包装（4+3 份）
  - 脚本装载助手（两种不兼容写法：`importlib.util.spec_from_file_location` vs 直接 `from scripts... import`）
- [ ] 注册 marker：`slow`、`subprocess`、`mps`、`bf16`、`safetensors`。现状零 marker，Linux/CPU 上 4 个测试静默消失且无人察觉。
- [ ] 按下述分组移入子目录（文件名不变）：`math/`、`layers/`、`models/`、`deployment/`、`security/`、`evaluation/`、`cli/`、`pretrained/`。判断分歧处：`test_deployed_weights.py` 归 `layers/`（按 import）、`test_no_decode_in_forward.py` 与 `test_key_material_isolation.py` 归 `deployment/`、`test_report_cli_coverage.py` 归 `evaluation/`（进程内断言报告语义，不算 CLI）。
- [ ] 每目录加 `__init__.py` 或设 `importmode = "importlib"`（防同名 basename 冲突——当前 45 个 basename 唯一，但移动后要保持安全）。
- [ ] 修 `PROJECT_ROOT` 的 `parents[1]` → `parents[2]`。
- [ ] `tests/test_eval_boundaries.py` 测的是 `evals` 而非 `fastprove`，随 P6 决策定位置。

**门**：382 passed 计数不变、等价性绿、且**裸 `pytest` 也能收集**（这是本阶段的新增验收点）。

### P8 文档重建（工作量最大的一段）

#### 8.1 建立生命周期标注

每份文档顶部加统一头：`状态`（CURRENT / CONTRACT-未实现 / HISTORICAL-已归档）、`日期`、`上游依据`、`适用范围`。新建 `docs/INDEX.md` 作为唯一入口，列出全部文档及其状态。

归档（**移动不删除**，它们是研究记录）到 `docs/archive/<日期>_<原名>.md`，并在文件头加 supersession banner：`START_PROMPT.md`、`docs/IMPLEMENTATION_PLAN.md`（127 KB，其 ~3046 行仍称 REPORT 为 `SUPERSEDED`/`HARD: FAIL`，与今日事实相反）、`augmented_noise_stage_a_review.md`、`AUGMENTED_NOISE_HANDOFF_20260915.md`、`problem_analysis.md`、`scheme_vs_impl_verdict.md`、`experiment_plan.md`、`superpowers/plans/`、`results/REPORT_zh.md`（已自标 SUPERSEDED）。`aloepri_augmented_noise_architecture.md` 已有正确 banner，作为模板。

#### 8.2 必须重写的文档

| 文档 | 问题 | 动作 |
|---|---|---|
| `README.md` | 称「噪声扫描尚未执行、REPORT 只有 pending」——而 REPORT §2.3 有完整 72/72 表；模块树缺 ~10 个现有模块；仍声称本地有 Qwen2 checkpoint（实际已不存在） | 重写：诚实状态、正确模块树、真实 runbook、与证据一致的限制摘要 |
| `docs/mathematics.md` | 563 行，零次提及 `StructuredBasis`、deployed 权重、`ker(L)`、双通道、`SecureEmbedding`、`TokenCodec`；描述的是 7 月构造。而 README 把它指为**唯一**数学文档 | 归档原文 + 新写 as-built 数学文档 |
| `results/raw/README.md` | 写 `not_run`，实际目录有 60+ 条记录 | 重写为 provenance 表 |
| `results/figures/README.md` | 写 `not_generated`，实际有跟踪的 tex/pdf | 重写 |
| `AGENTS.md` 的「Expected Repository Layout」 | 缺 `structured.py`、`converter.py`、`codec.py`、`coa_views.py`、`noise_budget.py`、`layers/{deployed,embedding,head,moe}.py`、`pretrained/`、`evals/` | 只改这一节，规则条文不动 |

#### 8.3 新增文档（填补新人读不到的内容）

- `docs/architecture.md`——as-built 模块地图：16 根模块 + 10 layers + 12 evaluation + `pretrained/`；分层与依赖方向；**生产路径 / 参考路径 / 侧路实验三分表**（说清 `conversion.py`+`layers/linear.py` 是参考、`layers/deployed.py` 是生产、`augmented_noise.py` 是 forward 从不调用的侧路、`coa_views.py` 是字符串契约而非隔离机制、`evals/` 与 `src/fastprove/evaluation/` 的关系）。
- `docs/runbook.md`——把散落的验证阶梯按序列成一页：`pytest` → `docs/design_identity_check.py` → `python -m evals.report --check-gates` → `docs/verification/verify_noise_ordering.py` → `scripts/run_accuracy_sweep.py` → `scripts/build_report.py`；并标出**当前被阻塞的环节**。
- `docs/config_reference.md`——每个 YAML 键的语义、默认值、取值域，标注哪些是 exact 模式强制（`rho_noise_bound: 0`、`refresh_mode`）；解释新词汇：channel 0（`ker(L)`，hiding = 0 bit）vs channel 1（语义通道，需要来自 Δ_k / D_V / ρ 的误差界）、`free_noise_energy_ratio`、`signal_noise_coupling_scale`、`basis_block_size`。目前这些只存在于 YAML 注释与 `noise_budget.py`/`structured.py` 的 docstring 里。
- `docs/scripts_index.md`——28 个脚本的索引：分类（流水线入口 / 一次性探针 / 疑似废弃）、命令、产物路径、被谁引用。
- `docs/STATUS.md`（或 CHANGELOG）——接续 `docs/implementation_notes.md`（它停在 2026-09-16，完全没覆盖 09-19 的 S0–S5 重构）。必须写入 `results/raw/blocked_stages_s6_s7.json` 的事实：S6/S7 未执行，pretrained checkpoint 已从本机消失（`checkpoint_bytes = 0`），**τ=1e-5 操作点的反证实验仍未关闭**——这条当前只存在于一个 JSON 里，任何 prose 文档都没提。
- `results/PROVENANCE.md`——claim → raw 文件 → 生成命令的映射表；标注 current / superseded。

#### 8.4 文档一致性修正（只改文档，不改数字）

- 修正 4 处引用了**不存在**的脚本：`verify_token_recovery_candidate_rescoring.py`、`verify_aloepri_subspace_attack.py`、`run_augmented_coa_eval.py`、`probe_embedding_norm_equalisation.py`。
- `scripts/run_augmented_noise_kernel.py:12-13` docstring 指向不存在的 `scripts.postprocess_augmented_noise_kernel`。
- `docs/IMPLEMENTATION_PROMPT_dual_channel.md` 的「现状必读」是 pre-S1 快照，已部分失真（`0.02` 硬编码 13 处已收敛为 `config.py:86` 的具名默认；`ker(L)` 已在 `structured.py:278` 构造；`noise_budget.py` 已落地并在 `converter.py:392-393/476-490` 接线；「40 个测试文件」现为 45）。加一节「S1 之后的增量」而非改写原文。
- **禁止复活的旧框架**：`aloepri_*.md`、`DERIVATION_PACKAGE.md`、`augmented_covariant_implementation_v2.md` 描述的 `expanded_concat`/`expanded_full`/`P̂`/`a`/`B_comp` 在 `src/` 中**不存在**（生产是 `D = d`）；T3/T4/T5 已证 channel-0 噪声的信息论隐藏量**恰为 0**，填满它只把已知明文样本需求从 `d+r+1` 抬到 `N+1`。文档重建必须统一采用后者的口径。
- `configs/eval_sweep.yaml` 的两个待查项：`execution.run_in_current_prototype_pass: false` 是过期门标志；`base_config: configs/tiny_approx.yaml`（带 `rho_noise_bound: 1e-5`）与同文件「exact 正确性用 tiny_exact」的注释表面冲突——`evaluation/sweep.py` 里查不到 `rho_noise_bound` 处理，**需实测 spec→config 构造链确认 sweep 的 plaintext/exact 点是否继承了非零 ρ 噪声**。若确实继承，属实质问题，立 PROBLEMS 条目，不在本次静默修改。
- `configs/augmented_noise_kernel.yaml` 是孤儿（无 `.py` 读它，`run_augmented_noise_kernel.py` 在代码里构造 `KernelConfig`）。该缺陷 `augmented_noise_stage_a_review.md:72` 已记录。本次只在文档标注「documentation-only」，接线属功能变更，另立项。
- 语言政策统一（见 §10-⑤）。

**门**：文档内所有相对路径、命令、文件引用做一次自动化链接检查（新增 `scripts/check_doc_links.py`，纯文档工具，不参与等价性）。

### P9 收尾验证

- [ ] 全量等价性比对 + `pytest`（含裸 `pytest`）。
- [ ] **新克隆冒烟**：`git clone` 到临时目录 → `pip install -e .` → `pytest` → 跑 runbook 里的只读检查。这是验证「结论可从仓库产物复现」的唯一硬证据。
- [ ] 输出交接摘要：改动文件清单、命令清单、通过/失败/未执行三分列表、`results/raw/refactor_baseline_*.json` 路径、下一阶段建议。

---

## 6. 目标结构（P3–P7 完成后）

```
src/fastprove/
├── seed.py  hashing.py  io.py  metrics.py  policy.py        # 无依赖底层
├── basis_common.py  transforms.py  structured.py            # 基变换两族
│   structured_generate.py
├── state.py  codec.py  coa_views.py  config.py
├── conversion.py            # 参考路径：一般增广仿射
├── noise_budget.py  augmented_noise.py                      # 侧路：核实验
├── converter/               # hygiene / load / client / save / basis_io
├── layers/                  # attention rmsnorm deployed(生产) linear(参考)
│                            # swiglu router moe embedding head
├── models/                  # plain / obfuscated / obfuscated_types / client
│                            # block_builder
├── pretrained/              # _checkpoint_io / llama / qwen2
└── evaluation/
    ├── environment.py runtime.py diagnostics.py exact_gate.py
    ├── layer_metrics.py performance.py spec_config.py runner.py
    ├── metrics.py accuracy.py artifacts.py sweep.py selection.py
    ├── token_cache.py pretrained.py attack_records.py correctness.py
    ├── prompts.py  benchmarks/{tasks,scoring,bundle}.py
    └── report.py report_format.py report_rows.py report_tradeoff.py
        report_figures.py report_tables.py report_markdown.py
tests/{math,layers,models,deployment,security,evaluation,cli,pretrained}/
  + conftest.py
docs/{INDEX,architecture,runbook,config_reference,scripts_index,STATUS}.md
  + archive/
```

---

## 7. 已知的测试缺口（本次不实现，仅登记）

AGENTS.md 验收门逐条映射后，两处无对应测试：

1. **固定种子复现相同 metrics**——只有组件级复现测试，没有任何测试把 metrics/run-record 流水线跑两遍并断言 payload 相等。`tests/test_evaluation_runner.py:263` 写了一条可追溯记录但没有重跑比对。这是全套里最弱的一门。
2. **MoE 生产输出不得暴露 router logits**——`tests/test_moe.py:276` 测的是正确性而非暴露面。

两者是补测试而非重构，建议本计划完成后单独立项（也可在 P7 顺带补，需你确认）。

---

## 8. 工作量估算

| 阶段 | 估时 | 风险 |
|---|---|---|
| P0 基线 | 0.5 天 | 低，但决定后续全部可验证性 |
| P1 卫生打包 | 0.5 天 | 低 |
| P2 死代码+标注 | 0.5 天 | 低 |
| P3 大文件拆分 | 2 天 | **中高**（移动量最大，门要跑得最勤） |
| P4 去重 | 1.5 天 | **高**（§4 全部高危项集中在此） |
| P5 scripts 库化 | 2 天 | 中（测试对私有符号的依赖） |
| P6 evals 归位 | 1 天 | 中（依赖决策①，可能牵出指标定义不一致） |
| P7 测试重组 | 1 天 | 中（fixture 提取顺序） |
| P8 文档重建 | 2.5 天 | 低技术风险，高判断成本 |
| P9 收尾 | 0.5 天 | 低 |
| 合计 | **≈12 工作日** | |

P1/P2/P8 可与其他阶段并行；P3 → P4 → P5 必须串行。

---

## 9. 提交纪律

1. **一次提交只做一类事**：移动 / 删除 / 抽取 / 文档，四者不混。
2. **纯移动提交不得含任何字符改动**（除 import 路径）。
3. 每次提交的 commit message 记录：阶段编号、等价性结果（`L1 OK / L2 OK`）、`pytest` 计数。
4. 等价性失败 → 回滚，不调基线。
5. 不动 `git config`、不 force push、不 `--amend` 已推送的提交。
6. 遇到「必须改行为才能继续」的情况 → 停下来记录并询问，不自行决定。

---

## 10. 需要你决策的事项

① **`evals/` 去向**：(A) 保留为独立顶层 harness，只补打包与文档；(B) 并入 `src/fastprove/protocol_eval/`；(C) 只统一指标定义、其余不动。推荐 (A)+(C)，因为 `results/REPORT.md` 的 exact-gate 命令依赖它，动它等于动已发表结果的复现路径。

② **`results/raw/*.jsonl` 是否纳入版本控制**：现状被 `.gitignore` 排除，但 `REPORT.md` 正引用 `eval_sweep.jsonl`，新克隆无法复核头条数字，与 AGENTS.md「所有报告数字必须可从仓库内命令与产物复现」冲突。建议改为白名单跟踪小体积 raw 记录。

③ **38 篇文献 PDF（72 MB，占跟踪体积 96%）是否移出仓库**（改为清单 + DOI/URL）。`.git` 已 59 MB。

④ **历史文档与疑似废弃脚本**：归档（保留研究记录，仓库更胖）还是删除（更干净，靠 git history 找回）？我倾向归档。

⑤ **文档语言政策**：现状混杂（`AGENTS.md`、`evals/README.md` 英文，其余中文，部分段落句内混用）。建议：`docs/` 与 `README.md` 用中文，代码 docstring 与 `AGENTS.md` 用英文。

⑥ **`REPORT.md` 引用了带 dtype 标签 bug 的 `eval_sweep.jsonl`**（`PROBLEMS.md:713-714` 已记录它被 `eval_sweep_dtype_fixed.jsonl` 取代）。改引用可能牵动正文数字 → **这超出「实质不变」的边界**，需你明确授权后单独处理。

⑦ **`setup.py` 是否删除**：它的注释说是为 Python 3.9 系统 pip 保留的 legacy shim，但本机环境是 3.13.12，`pyproject.toml` 足够。删除可消除元数据双份维护（`UNKNOWN.egg-info` 就是它的产物）。

⑧ **§7 的两个测试缺口**是否在 P7 顺带补齐，还是完成本计划后单独立项。
