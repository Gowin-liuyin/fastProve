# fastProve

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/INDEX.md、docs/STATUS.md
适用范围：项目入口：诚实状态、真实模块树、可执行命令、证据口径与限制

`fastProve` 是一个 PyTorch 研究原型，用于实现并比较小型 Llama-like
Transformer 的四种推理模式：

- `plaintext`：明文基线；
- `exact`：RoPE 后共同正交 Q/K、混合 Value、Softmax 无噪声，数学上应与明文
  只差浮点误差；
- `topk_preserving`：按 clean Top-k 边界 margin 限幅 logit 噪声，保持 Top-k 集合；
- `free_bounded`：固定有界 logit 噪声，允许排序变化。

项目验证的是数学与工程可行性，不是生产密码系统。

## 当前状态（诚实口径）

- **四种模式均已实现**（`src/fastprove/modes.py:12`），同一套基础权重下可比较。
- **72/72 点噪声扫描已落盘**：`results/REPORT.md` §2.3 给出完整表格，
  原始记录 `results/raw/eval_sweep.jsonl`（72 条，随机 tiny 模型，仅正确性证据）；
  dtype 溯源标签修正版为 `results/raw/eval_sweep_dtype_fixed.jsonl`（精度指标逐位
  相同；按用户决定 ⑥，`REPORT.md` 的引用保持不变，旧文件仅标记 superseded）。
- **exact 数学测试全绿**：`python -m pytest tests -q` → **399 passed**；
  FP64 恒等式预言机 `python docs/design_identity_check.py` 13 项全过。
- **重构等价性已冻结**：`python scripts/verify_refactor_equivalence.py --check`
  必须复现 `results/raw/refactor_baseline_5e19c43ea2b2.json`（12 section、2395 摘要叶子）。
- **包可安装**：`python -m pip install -e .` 后 `import fastprove` 与裸
  `pytest` 均可工作（P0/P1 修复）。
- **仓库不包含任何 checkpoint**。`results/raw/pretrained_evaluation_status.json`
  曾记录本机找到的 Qwen2 1.5B artifact，但该目录**现已从本机消失**；
  `results/raw/blocked_stages_s6_s7.json` 记录 `checkpoint_bytes = 0`，S6/S7
  两阶段未执行。一个 Llama-3.2-3B checkout 只存在于本机仓库外路径
  （见 `docs/PROBLEMS.md` §4.1），不入库、不随仓库分发。
- **`results/REPORT.md` 的端到端数字全部来自随机 tiny 模型**，只能作为正确性证据，
  不能描述为有意义的语言模型精度；预训练评测尚未执行（见 §5 限制）。
- 已知未决问题：`docs/PROBLEMS.md` §3.4（Wilson 区间两份定义 z 常数不同）、
  §3.5（sweep 的 plaintext/exact 点继承 `rho_noise_bound: 1e-5` 与
  `per_request`，不是 ρ=0 基线）。

## 代码结构

```text
src/fastprove/
├── seed.py  hashing.py  io.py  metrics.py  policy.py  modes.py
├── basis_common.py  transforms.py  structured.py
├── state.py  codec.py  coa_views.py  config.py
├── conversion.py            # 参考路径：一般增广仿射
├── noise_budget.py  augmented_noise.py   # 侧路：双通道账本 / 核实验
├── bounded_noise.py
├── converter/               # hygiene / load / client / save / basis_io
├── layers/
│   ├── attention.py  rmsnorm.py  deployed.py  linear.py
│   ├── swiglu.py  router.py  moe.py  embedding.py  head.py
├── models/
│   ├── plain.py  obfuscated.py  obfuscated_types.py
│   ├── client.py  block_builder.py
├── pretrained/              # _checkpoint_io.py / llama.py / qwen2.py
└── evaluation/
    ├── environment.py runtime.py diagnostics.py exact_gate.py
    ├── layer_metrics.py performance.py spec_config.py runner.py
    ├── metrics.py accuracy.py artifacts.py sweep.py selection.py
    ├── token_cache.py pretrained.py attack_records.py correctness.py
    ├── prompts.py
    ├── benchmarks/{tasks,scoring,bundle}.py
    └── report.py report_format.py report_rows.py report_tradeoff.py
        report_figures.py report_tables.py report_markdown.py
```

另有独立的第二套评测 harness `evals/`（`python -m evals`，含 F0–P3 条件矩阵与
`--check-gates`），以及顶层 `scripts/`、`tests/{math,layers,models,deployment,
security,evaluation,cli,pretrained}/`。生产路径、参考路径、侧路实验的三分表见
[`docs/architecture.md`](docs/architecture.md)。

## 安装与快速验证

```bash
python -m pip install -e '.[test]'          # 可编辑安装；3.9 系统 pip 过旧时用 --user --no-build-isolation
python -m pytest tests -q                   # 399 passed
python scripts/verify_refactor_equivalence.py --check
python docs/design_identity_check.py        # 13 项 FP64 恒等式
```

一键验证阶梯（含各步骤阻塞状态）见 [`docs/runbook.md`](docs/runbook.md)。

## 显式执行（默认只读/不跑模型）

```bash
# tiny exact 正确性运行（写 results/raw/correctness.jsonl；随机 tiny，仅正确性）
python scripts/run_correctness.py --execute

# 72 点扫描：默认只列出配置，不跑推理；显式执行需 --execute-deferred
python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml
python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml \
    --output results/raw/eval_sweep.jsonl --execute-deferred

# 报告器：默认只读检查；--execute 才写表/图/REPORT.md（不跑模型）
# 注意：当前 --raw 传目录会因 attack_costs.jsonl 等非 run-record JSONL 报错，
# 请显式指向某个 run-record JSONL。
python scripts/build_report.py --raw results/raw/eval_sweep_dtype_fixed.jsonl --output results
python scripts/build_report.py --raw results/raw/eval_sweep_dtype_fixed.jsonl --output results --execute

# 第二套 harness 的门禁（随机 tiny）
PYTHONPATH=src:. python -m evals --check-gates
```

预训练/数据类命令（`scripts/run_accuracy_sweep.py --model-path ... --dataset-cache ...`、
`scripts/preflight_experiment.py`、`scripts/prepare_token_cache.py`）目前因本机
checkpoint 缺失而**不可执行**，详见 `results/raw/blocked_stages_s6_s7.json` 与
[`docs/runbook.md`](docs/runbook.md)。

## 结果口径

只有指定 attempt 下可验证的成功 run record 才能进入表、图和结论；随机 Tiny 记录
必须标注 `correctness_only_random_tiny` / `meaningful_lm_evidence: False`。
已标 superseded 的记录（`eval_sweep.jsonl`、`llama_compare_*`、
`osnip_style_bench_{smoke,n200}.json`）保留原样、不删数字；引用必须带 caveat。
`results/raw/` 的逐文件溯源见 [`results/raw/README.md`](results/raw/README.md)。

## 限制与不声明

完整威胁模型见 [`docs/threat_model.md`](docs/threat_model.md)，摘要如下：

- 原型最多评估**诚实但好奇**（honest-but-curious）的服务方：它按既定代码执行，
  但会观察持久化张量与普通框架输出；
- 允许假设“指定融合操作不返回其内部干净临时值”——这是**实现假设，不是密码学保证**；
- **不提供** FHE/MPC/TEE、标准 LWE 密文或端到端加密；使用 ML-KEM/LWE 派生种子
  也不构成 LWE 安全性；
- 若攻击者能修改 CUDA/Triton/PyTorch kernel、挂任意 hook、转储寄存器/临时缓冲区
  或读取全部客户端秘密，本原型**不保护** Softmax logits、概率或被提取的信号分量；
- 实测泄漏：`ρ` 在 FP32 下恢复 layer-0 输入 token 的比率约 92.5%（Qwen2）/
  93.4%（Llama）；公开基座 embedding 时词表置换挡不住明文（行范数对行置换不变）；
  融合内核假设也挡不住 `ρ`；Gram 几何暴露范数与内积；`e ≈ hC` 是第二条通道；
- channel 0（`ker(L)`）噪声的信息论隐藏量**恰为 0**，不是隐私来源；
- 未关闭的缺口：τ=1e-5 操作点的候选重排反证实验（S6）未执行；真实任务精度在
  加噪下的退化未测；Llama-3.2-3B 的 `llama3` rope_scaling 未实现，故任何 Llama
  基线本身需要修正后才能用于精度结论；
- eager 参考实现的墙钟开销（+31%…+58%）不是融合 kernel，也不是 ≤5% MAC 预算的证明。

## 文档

- [`docs/INDEX.md`](docs/INDEX.md)：全部文档的状态与阅读顺序（唯一入口）
- [`docs/runbook.md`](docs/runbook.md)：一页式验证阶梯
- [`docs/STATUS.md`](docs/STATUS.md)：P0–P7 重构进度、测试计数、阻塞与下一步
- [`docs/mathematics.md`](docs/mathematics.md)：as-built 数学规格（逐条 `file.py:line`）
- [`docs/architecture.md`](docs/architecture.md)：生产/参考/侧路三分表
- [`docs/PROBLEMS.md`](docs/PROBLEMS.md)：全部已知问题与不得声明的内容
- [`results/REPORT.md`](results/REPORT.md)：阶段 A–D 实测结果（随机 tiny）
