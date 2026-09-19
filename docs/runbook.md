# fastProve 验证阶梯（runbook）

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/STATUS.md、results/raw/blocked_stages_s6_s7.json
适用范围：从零验证当前仓库的一页式命令阶梯；所有命令均在仓库根目录执行

本页只列**真实存在且可执行**的命令。默认设备为 CPU（本机 macOS arm64，无 CUDA），
随机 tiny 模型；需要 checkpoint 的步骤在 §8 明确标为阻塞。命令按顺序执行，前一步
失败就不要继续。

前置（只需一次）：

```bash
python -m pip install -e '.[test]'   # 或 --user --no-build-isolation（旧 pip）
```

## 1. 测试套件

```bash
python -m pytest tests -q
```

通过判据：`399 passed`（P7 重组后计数；目录见 `tests/`）。

## 2. 重构等价性基线

```bash
python scripts/verify_refactor_equivalence.py --check
```

通过判据：`EQUIVALENCE OK: 12 section(s) match refactor_baseline_5e19c43ea2b2.json`。
这是逐位（L1 张量 / L2 产物结构）比对；失败只允许回滚，不允许改基线。

## 3. 数学恒等式预言机（FP64）

```bash
python docs/design_identity_check.py
```

通过判据：13 项恒等式全部 `[OK]`，末尾 `all identities verified in FP64`
（示例误差量级 ≤ 3e-14）。

## 4. 噪声放置推导数值核验

```bash
python docs/verification/verify_noise_ordering.py            # 只读，打印 T1–T9 结果
# 需要落盘时（会重写已跟踪工件，请先确认）：
# python docs/verification/verify_noise_ordering.py \
#     --output results/raw/noise_ordering_analysis_20260919.json
```

通过判据：T1–T9 全部按预期（in-`ker(L)` 噪声语义无损、outside 噪声误差恰为 `nL`）。

## 5. 协议门禁（evals harness）

```bash
python -m evals --check-gates --precision fp32
```

`python -m evals` 是 `python -m evals.report` 的别名（`evals/__main__.py:1`），
默认 `--config evals/configs/P2.yaml --base-config configs/tiny_exact.yaml`。
通过判据：输出 `HARD: PASS   SOFT: PASS   ALL: PASS`。

**必须显式 `--precision fp32`**。默认 bf16 目前会在 debug 路径崩溃
（`RuntimeError: ... float != c10::BFloat16`，`models/obfuscated.py:653`）；该缺陷在
重构前提交 `5e19c43` 上同样复现，见 `docs/PROBLEMS.md` §3.6。P6 打包 `evals` 之后
不再需要 `PYTHONPATH`。

与 `results/REPORT.md` §2.1 逐字对应的门禁命令（写 `results/raw/exact_gate_D1.json`；
若要避免覆盖已跟踪工件，先把 `--output` 改到临时路径）：

```bash
python -m evals --base-config configs/tiny_exact.yaml \
    --config evals/configs/P2.yaml --precision fp32 --keys 1 --layers 1,2,3,4 \
    --check-gates --output results/raw/exact_gate_D1.json
```

## 6. 72 点扫描（tiny 正确性）

```bash
python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml          # 只列 72 点，不跑推理
python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml \
    --output results/raw/eval_sweep.jsonl --execute-deferred                   # 显式授权执行
```

通过判据：72 条 run record 全部 `status: success`，且 exact 门禁通过（无 skipped）。
已知边界：sweep 的 plaintext/exact 点从 `base_config: configs/tiny_approx.yaml`
继承 `rho_noise_bound: 1e-5` 与 `per_request`，不是 ρ=0 的 exact 基线
（`docs/PROBLEMS.md` §3.5）。

## 7. 报告器

```bash
# 只读检查（不写文件）。注意：--raw 传目录会被 attack_costs.jsonl 等非 run-record
# JSONL 拒绝，需显式指向 run-record 文件。
python scripts/build_report.py --raw results/raw/eval_sweep_dtype_fixed.jsonl --output results
python scripts/build_report.py --raw results/raw/eval_sweep_dtype_fixed.jsonl --output results --execute
```

通过判据：只读模式打印 `record_count: 72`、`status_counts.success: 72`；
`--execute` 写 `results/tables/`、`results/figures/`、`results/REPORT.md`（不跑模型）。

## 8. 阻塞步骤（当前不可执行，照实记录）

以下步骤需要**预训练 checkpoint**，而本机已无可用 checkpoint：

- 预训练资产审计：`results/raw/pretrained_evaluation_status.json` 引用的
  Qwen2 1.5B artifact 目录已从本机消失；`scripts/audit_pretrained_assets.py`
  现报 `checkpoint_bytes = 0`。
- S6 候选重排反证实验（τ=1e-5 操作点）与 S7 embedding 范数均衡探针：
  **未执行**，阻塞记录 `results/raw/blocked_stages_s6_s7.json`
  （`status: not_executed`，`decision: user chose to skip and record the blockage
  rather than download a checkpoint`）。
- 标准 causal-LM 评测与 `scripts/run_accuracy_sweep.py --model-path ... --dataset-cache ...`：
  同样被 checkpoint 阻塞；Flickr30k caption-only cache 只是非标准候选。
- Llama-3.2-3B checkout 仅存在于本机仓库外路径（`docs/PROBLEMS.md` §4.1），
  不入库；且 `llama3` rope_scaling 未实现，其基线需修正后才可用于精度结论。

阻塞解除条件与所需材料见上述 JSON 的 `would_require` 字段。**不得**用随机 tiny
结果替代这些测量（AGENTS.md：random tiny 只能作正确性证据）。

## 9. 收尾检查

```bash
python scripts/check_doc_links.py --skip-archive   # 文档链接检查（纯文档工具）
git status --short                                  # 确认没有意外生成物
```
