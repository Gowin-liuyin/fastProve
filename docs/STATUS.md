# fastProve 项目状态（P8b 之后）

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/CLEANUP_PLAN.md、docs/PROBLEMS.md
适用范围：接续 `docs/implementation_notes.md` 的进度快照：P0–P7 重构、当前门禁、未决问题与下一步

本文接续 [`docs/implementation_notes.md`](implementation_notes.md)（其最后一条记录停在
P4，2026-09-19）。P8a 已建立文档生命周期与归档；本次 P8b 完成 README / 数学 /
raw / figures 重写，并新建本文件与 [`docs/runbook.md`](runbook.md)。

## 1. 当前事实（2026-09-20）

| 维度 | 事实 |
|---|---|
| 测试 | `python -m pytest tests -q` → **399 passed**（P7 重组后计数） |
| 等价性 | `python scripts/verify_refactor_equivalence.py --check` → 12 section 与 `results/raw/refactor_baseline_5e19c43ea2b2.json` 逐位一致 |
| 数学预言机 | `python docs/design_identity_check.py` → 13 项 FP64 恒等式全过 |
| 环境 | Python 3.13.12 / PyTorch 2.11.0 / NumPy 2.4.4；macOS arm64；CPU + MPS，无 CUDA |
| 包 | `pip install -e .` 可用；裸 `pytest` 可收集 |
| 模式 | `plaintext` / `exact` / `topk_preserving` / `free_bounded` 四模式均实现 |
| 扫描 | 72/72 点已落盘（随机 tiny，仅正确性），见 `results/REPORT.md` §2.3 |
| 预训练 | 仓库不含 checkpoint；Qwen2 1.5B 本机 artifact 已消失（`checkpoint_bytes = 0`） |

## 2. P0–P7 重构提交（`git log --oneline`）

| 提交 | 阶段 | 内容 |
|---|---|---|
| `ba586c5` | P0 | 冻结重构等价性基线（`refactor_baseline_5e19c43ea2b2.json`） |
| `fbba79c` | P1 | 卫生、打包与仓库瘦身（LaTeX 中间物、白名单） |
| `8d0717e` | P2 | 死代码清理与分层标注 |
| `a347a30` | P3a | 抽出 `policy`/`modes`/`metrics`，修分层倒置 |
| `a6a6f72` | P3b | 按既有缝拆分 `evaluation/runner.py` |
| `86f470d` | P3c | 拆分 `evaluation/report.py` |
| `b7b3aa6` | P3d | `models/obfuscated.py` → client/types/builder |
| `d584691` | P3e | 把 `converter` 单文件拆成 `converter/` 包 |
| `bb4d255`–`b9a1f38` | P4a–P4f | 去重（hash、checkpoint I/O、basis 原语、dense 生成、Gram rms_scale、位混合原语） |
| `f4a1c96` | P4 | 记录去重结果与性能测量延期 |
| `57548fc`–`ee4e038` | P5a–P5c | scripts 库化（sweep record builders、pretrained compare、osnip benchmarks） |
| `165b1f4`–`2bae50a` | P6a–P6b | `evals/` 打包化、指标定义对齐（逐位一致测试） |
| `cd6cc04` | P7 | 测试重组（8 个子目录 + `conftest.py`）、补齐两个验收缺口 |

更早的 S0–S5 双通道重构为 `5e19c43`（等价性基线的对照点）。

## 3. 门与纪律

- 每步提交门：`python -m pytest tests -q` + `python scripts/verify_refactor_equivalence.py --check`；
- 等价性失败只允许回滚，不允许调整基线；
- 本次文档重建**未改任何 Python 代码、未重跑任何数值实验**（唯一新写入的运行时
  文件是验证过程产生的临时副本，见 `git status`）。

## 4. 未决问题（`docs/PROBLEMS.md`）

- **§3.4 待决策**：Wilson 区间两份定义 z 常数不同（`evals/stats.py` 的
  `1.959963984540054` vs `benchmarks/scoring.py` 的 `1.959963985`），最大差
  ~1.2e-10；两份实现都保留，未统一。
- **§3.5 待决策**：`configs/eval_sweep.yaml` 的 `base_config: configs/tiny_approx.yaml`
  使全部 72 个 spec（含 plaintext/exact 基线点）继承 `rho_noise_bound: 1e-5` 与
  `per_request`；exact 正确性仍只对 `configs/tiny_exact.yaml`（ρ=0、`fixed_debug`）成立。
  在决策前不得把重跑的 sweep 基线点描述为 ρ=0 基线。
- **A 类构造固有泄漏（7 条，不可修）**：最严重的是 `ρ` 泄漏 `‖h‖` 并在 FP32 下
  恢复 layer-0 输入 token（实测 92.5% Qwen2 / 93.4% Llama）；公开基座 embedding 时
  词表置换挡不住明文；Gram 几何、`e ≈ hC` 通道、融合内核假设挡不住 `ρ`。
- **C 类证据缺口**：所有精度结论仍跑在随机 tiny 模型上；Llama `llama3`
  rope_scaling 未实现；τ=1e-5 下真实任务精度退化未测；深层泄漏语料有局限。

## 5. 阻塞阶段（S6/S7）

`results/raw/blocked_stages_s6_s7.json`（2026-09-19）记录：

- **S6** 候选重排反证实验：`not_executed`。阻塞原因：`token_recovery_from_rho.json`
  引用的 Qwen2 1.5B base checkpoint 已不在本机（整个 `model_artifacts/raw_remote_pull/models`
  目录消失，审计报 `checkpoint_bytes = 0`）。τ=1e-5 操作点的反证实验**仍未关闭**。
- **S7** embedding 范数均衡可行性探针：`not_executed`，同一阻塞。
- 用户决定：跳过并登记阻塞，不下载 checkpoint。

## 6. 下一步

1. **P9 新克隆冒烟**（唯一硬证据）：`git clone` 到临时目录 → `pip install -e .` →
   `pytest` → 按 `docs/runbook.md` 执行只读检查；
2. P8 剩余文档：config_reference（配置参考）、scripts_index（脚本索引）、
   results/PROVENANCE（claim→工件映射）（CLEANUP_PLAN §8.3）；
3. 两个 §3 待决策项按显式授权单独处理（任一选择都会改动已记录工件，不得静默统一）；
4. 若获得 checkpoint 与授权：执行 S6/S7 与标准 causal-LM 评测，替换随机 tiny 证据。

## 7. 本次文档重建发现的代码/文档不一致（登记，未改代码）

1. `scripts/build_report.py --raw results/raw`（目录形式）当前会报
   `run record missing fields`：目录里的 `attack_costs.jsonl` 与
   `real_scenario_prompts_1500.jsonl` 不是 run record，而报告器对目录下所有
   `*.jsonl` 做 schema 校验。可用的形式是显式指向某个 run-record JSONL
   （如 `--raw results/raw/eval_sweep_dtype_fixed.jsonl`）。README 已按可用形式改写。
2. `docs/mathematics.md`（旧版）与 AGENTS.md 的 \(M=\Pi DB\) 描述单侧排列，而
   as-built 实现是 \(M=\Pi_1DB\Pi_2\)（`src/fastprove/structured.py:1`、`:153`）；
   新数学文档已按实现描述并登记该差异。
3. `results/REPORT.md` 中的旧测试路径（形如 tests/test_rho_noise.py）已按 P7 重组后的
   真实路径修正；runtime_config.json 标注为转换产物的运行时文件名（不入库）。
