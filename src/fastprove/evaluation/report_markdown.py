"""Final Markdown assembly for report derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence


def _write_report_markdown(
    *,
    report_markdown: Path,
    experiment_status: str,
    pretrained_status: str,
    status_counts: Dict[str, int],
    raw_source_lines: str,
    provenance_table: str,
    exact_table: str,
    accuracy_table: str,
    softmax_table: str,
    performance_table: str,
    non_success_table: str,
    overall_best: Optional[str],
    topk_best: Optional[str],
    free_best: Optional[str],
    topk_frontier: Sequence[str],
    free_frontier: Sequence[str],
) -> None:
    """Render and write REPORT.md from precomputed sections."""

    report = f"""# fastProve 实验报告

> 状态：`{experiment_status}`。预训练评测：`{pretrained_status}`。
> 记录统计：success={status_counts["success"]}，failure={status_counts["failure"]}，
> skipped={status_counts["skipped"]}。

## 1. 实验目标

公平比较 plaintext、exact、topk_preserving 与 free_bounded 四种模式，并
分别回答 exact 数值误差、精度下降、排序变化、最佳折中和安全非结论。

## 2. 数学实现摘要

原型使用良态增广基、显式非线性检查点、RoPE 后共同正交 Q/K 变换、
混合 Value 路径、SwiGLU 置换/缩放补偿，以及仅作用于有效 logits 的
FP32 有界噪声。

## 3. 环境、模型和数据

所有数值只能来自 raw JSONL。随机 Tiny LM（random tiny model）仅用于
correctness，不能作为有意义的语言模型准确率证据。

Raw 证据：

{raw_source_lines}

{provenance_table}

## 4. 精确模式正确性

{exact_table}

只有 `exact_gate_passed=True` 且 greedy/token agreement 均为 1 的记录
才能支持“仅浮点级误差”的本次实验结论；否则必须先调查。

## 5. 噪声强度—精度曲线

见 `figures/accuracy_vs_tau.png` 与 `tables/summary.csv`。成功记录以
不连线的散点显示；failure/skipped 点不插值，并在下表中直接追溯 raw 行。
若状态为 `completed_selected_subset`，曲线仅覆盖候选 manifest 选中的点，
不能解释为完整 72 点扫描。若记录启用 sample-unit bootstrap，accuracy 表同时
列出 top-1 drop 和 PPL relative increase 的 95% 区间。

{accuracy_table}

## 6. Softmax 排序与 Top-k 变化

见 `figures/softmax_vs_tau.png`；该图同样只画不连线的散点。Top-k 模式
保证的是集合而非集合内顺序；free-bounded 允许但不保证发生排名改变。

{softmax_table}

## 7. 最佳折中配置

透明规则：先筛选 top-1 绝对下降不超过 0.01 且 token agreement 至少
0.99 的非零噪声点，再选实际噪声最大者；若无满足点，则最小化
top-1 drop 与 perplexity 相对增幅之和。只有明确标记为预训练且数据可作为
LM 证据、同一 cohort 且 exact gate 通过的记录才参与选择。

- 整体：`{overall_best or "not_available"}`
- topk_preserving：`{topk_best or "not_available"}`
- free_bounded：`{free_best or "not_available"}`
- topk_preserving Pareto：`{", ".join(topk_frontier) or "not_available"}`
- free_bounded Pareto：`{", ".join(free_frontier) or "not_available"}`

## 8. 性能开销

转换、prefill、decode/TPOT、tokens/s、峰值内存和 KV-cache 内存必须由
正式运行记录；当前状态不得从代码结构推测数值。

{performance_table}

失败与跳过点不插值、不删除，列示如下：

{non_success_table}

## 9. 威胁模型和安全限制

本原型只考察 honest-but-curious 观察者看到的持久张量和普通框架输出。
“API 不返回概率”不是密码学安全。能够修改内核、挂任意 hook、读取融合
检查点内部、dump 寄存器/临时缓冲区或读取客户端秘密的攻击者不受保护。
本项目不是 FHE、端到端加密，也不因任何 KDF 而获得 LWE 安全性。

## 10. 完整复现命令

```bash
python3 -m pytest -q
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml --execute
python3 scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --execute-deferred
python3 scripts/build_report.py --raw results/raw --output results --execute
```
"""
    report_markdown.write_text(report, encoding="utf-8")
