# fastProve

`fastProve` 是一个 PyTorch 研究原型，用于实现并公平比较小型
Llama-like Transformer 的四种推理模式：

- `plaintext`：原始明文基线；
- `exact`：Q/K 精确协变、混合 Value、Softmax 无噪声；
- `topk_preserving`：按 clean Top-k 边界 margin 限制 logit 噪声；
- `free_bounded`：固定有界 logit 噪声，允许排序变化。

项目验证的是数学与工程可行性，不是生产密码系统。它不提供 FHE、标准 LWE
密文、端到端加密或针对任意 hook、内核修改、寄存器/临时缓冲区转储的保护。

## 当前状态

原型、单元/正确性测试、增量 KV cache、机器可读运行记录、72 点扫描枚举、本地
Qwen2 适配、对齐 token cache、资源预检和报告生成器已经实现。按用户当前要求，
噪声扫描和长性能实验尚未执行；因此 [results/REPORT.md](results/REPORT.md) 只
包含 `pending` 状态，不包含困惑度下降、最佳配置或真实语言模型精度结论。

随机 Tiny LM 只用于正确性。环境审计发现本机已有可离线加载的 Qwen2 1.5B
checkpoint，适配入口已完成；仍需提供/获准一个对齐 causal-LM 语料 cache 才能
产生标准 LM 精度证据。当前已准备一个 405 条样本的 Flickr30k
caption-only cache，但它不是通用语言模型基准；相关状态和方案见：

- 本地资产状态由
[`scripts/audit_pretrained_assets.py`](scripts/audit_pretrained_assets.py) 写入
[`results/raw/pretrained_evaluation_status.json`](results/raw/pretrained_evaluation_status.json)。

- 当前可执行范围、资源警告和第一条校准命令见
[`results/raw/experiment_readiness.json`](results/raw/experiment_readiness.json)。

- [数学实现](docs/mathematics.md)
- [威胁模型](docs/threat_model.md)
- [环境与实现记录](docs/implementation_notes.md)
- [分阶段实验方案](docs/experiment_plan.md)
- [TDD 实施计划](docs/superpowers/plans/2026-07-31-fastprove-prototype.md)

当前 16-GiB MPS 主机的 FP32 预检会给出内存警告；预检同时显示 BF16 的估计工作集
约为 10.7 GB。BF16/MPS 可以先做资源和适配器校准，但只能在独立的 FP32 exact
gate 通过后作为近似噪声精度结论，不能用 dtype 标签替代 exact 验收。MPS/CUDA
检查点运行时使用 FP32（离线矩阵生成/求逆仍为 FP64）。

## 代码结构

```text
src/fastprove/
├── config.py, seed.py, state.py, transforms.py, conversion.py
├── layers/
│   ├── linear.py, rmsnorm.py, attention.py, swiglu.py, router.py
├── models/
│   ├── plain.py, obfuscated.py
└── evaluation/
    ├── correctness.py, runner.py, accuracy.py, metrics.py, token_cache.py
    ├── artifacts.py, sweep.py, report.py
```

生产接口不返回 decoded hidden/noise、attention probability 或 router logits；
调试接口需要显式开启。非线性检查点所需的基材料不进入服务模块的
`state_dict`/`named_buffers`，但 Python introspection 或恶意内核仍可读取
进程内临时值，因此这只是参考可信边界。

## 安装与验证

```bash
# With a modern virtualenv/pip, editable mode is fine:
# python3 -m pip install -e '.[test]'
# The stock macOS/Xcode Python 3.9 pip is too old for PEP 660; use a regular
# user install there instead (the dependencies are already audited locally):
python3 -m pip install --user --no-build-isolation '.[test,pretrained-lite]'
# The larger `.[pretrained]` extra additionally installs `datasets` for a
# future downloaded/public corpus; it is not needed for the cached local run.
python3 -m pytest -q
python3 -m compileall -q src evals scripts
```

测试覆盖线性恒等式、布局转换、条件数、域分离、RMSNorm gamma 吸收、RoPE
后共同 Q/K 正交、GQA、mask/tie、四种 Softmax 模式、混合 Value、SwiGLU、
Router、exact Block/Tiny LM、增量 cache、指标和工件协议。

## 计划与显式执行

计划/预检命令默认不运行模型、不写 raw 结果；token cache 命令只写用户指定的
新 cache 文件：

```bash
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml
python3 scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml
python3 scripts/preflight_experiment.py --help
python3 scripts/build_report.py --raw results/raw --output results
```

对本地 Flickr30k caption-only cache 做资源预检时，也必须显式加入
`--accept-nonstandard-caption`；否则预检会拒绝把它标成可执行的标准 LM 输入。

后续获准执行且已有本地模型/数据时，显式命令为：

```bash
python3 scripts/run_correctness.py \
  --config configs/tiny_exact.yaml \
  --output results/raw/correctness.jsonl \
  --execute

python3 scripts/run_accuracy_sweep.py \
  --config configs/eval_sweep.yaml \
  --model-path "/absolute/path/to/qwen2-1.5b" \
  --dataset-cache results/raw/eval_inputs.pt \
  --output results/raw/eval_sweep.jsonl \
  --execute-deferred

# 仅当明确接受 Flickr30k 的非标准 caption-only 范围时，使用：
# --dataset-cache results/raw/flickr30k_caption_eval_inputs.pt \
# --accept-nonstandard-caption

# 本机内存受限时显式指定，不发生隐式 device/dtype fallback；CPU FP32 是最小
# 可复核起点，MPS/BF16 仅作独立资源校准；校准可显式缩短计时：
# --calibration-only --batch-size 1 --device cpu --activation-dtype fp32 \
# --warmup-runs 1 --timed-runs 3
# 校准也可显式关闭 bootstrap（仅用于快速控制面，不用于最终报告）：
# --bootstrap-replicates 0

python3 scripts/build_report.py \
  --raw results/raw \
  --output results \
  --execute
```

本机当前可直接使用的第一条真实模型校准命令（明确接受 Flickr30k
caption-only、非标准 LM 范围）已写入
[`results/raw/experiment_readiness.json`](results/raw/experiment_readiness.json)。
它固定 CPU/FP32、batch 1、8 条样本和短计时重复；执行前仍需用户明确授权，因为
它会加载本地 1.5B checkpoint 并写入新的 raw JSONL。

Sweep 默认使用配置中的完整样本数（当前为 64）；资源受限时显式加
`--calibration-only`（当前 8），不会把校准结果冒充完整评测；统一内存紧张时
可加 `--batch-size 1`。

扫描在进入近似模式前执行 exact 硬门禁；任一数值容差、NaN/Inf、
teacher-forced agreement 或 greedy agreement 失败，后续配置会逐条记录为
`skipped`。Raw JSONL 拒绝重复 `run_id`；显式执行会在任何推理开始前拒绝
非空输出目标。当前原型不实现 partial resume，重试必须使用新的空目标，并把
该次 attempt 放在独立 raw 目录中，报告器只读取该 attempt 的文件。每条记录
携带同一 stage 的 `expected_run_ids`；只有清单与实际 run 完全对账、全部成功且
exact gate 通过，完整 72 点报告状态才可成为 `completed`；使用候选 manifest
的 full 子集会标为 `completed_selected_subset`，不能冒充完整覆盖。
候选 manifest 即使来自乱序 raw，也会在执行前稳定重排为
`plaintext → exact → approximate`，不会因输入顺序提前消耗 exact gate。
模型或 token-cache 预检失败时，也会为 72 个配置写入独立 `failure` 记录，避免
空白 attempt 掩盖输入阻塞。

深层 FP32 checkpoint 的 exact gate 会把实际 device、checkpoint 算术 dtype 和
使用的容差写入 `metrics.exact_gate_tolerances`。当前已用 CPU/MPS smoke 登记
Softmax absolute envelope `2e-3`（relative-L2 仍为 `1e-4`，token agreement
仍必须为 1）；这不是把 BF16 结果标成 FP32，也不是跳过 exact gate。

算力受限时可先运行 `scripts/run_accuracy_sweep.py --calibration-only`，再用
[`scripts/select_sweep_candidates.py`](scripts/select_sweep_candidates.py) 生成
候选 manifest，并通过 `--spec-ids-file` 执行 full 子集；筛选规则和失败点会写入
manifest，不允许手工只保留有利结果。

## 结果口径

只有指定 attempt 下 `results/raw/**/*.jsonl` 中可验证的成功记录才能进入表、图和结论。正式
报告必须同时给出明文绝对值、混淆绝对值、绝对变化和相对变化；基线为零时
相对下降记为 `null/undefined`。失败和跳过记录会进入 CSV 与报告清单，不被
插值；成功点以不连线散点显示。每个表项保留 raw 文件和行号，cohort 同时锁定
checkpoint/config/tokenizer hash、数据 hash、activation dtype、checkpoint 算术
dtype、device、sequence length、generation length 和 batch size。
最佳点分别按 top-k-preserving、free-bounded 和整体给出，并要求
预训练/数据证据标记与 exact gate。随机 Tiny、未执行点和失败点不得被描述为
真实语言能力或安全证据。
