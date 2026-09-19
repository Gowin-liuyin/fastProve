#!/usr/bin/env python3
"""从大规模 compare JSON 生成中文精度报告 results/REPORT_zh.md。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "—"
    return ("%%.%df" % digits) % x


# Runs produced before the benchmark harness was repaired must not be quoted.
# The marker is emitted from the source artifact rather than hand-edited into
# the Markdown, so regenerating the report cannot silently drop the warning.
_HARNESS_FIX_DATE = "2026-08-06"


def _harness_validity_banner(data: Dict[str, Any]) -> List[str]:
    """Return a SUPERSEDED banner when the source run predates the harness fix.

    A run is trustworthy only if its artifact records the repaired selection
    and scoring settings.  Older artifacts have no such keys, which is exactly
    the signal used here.
    """

    config = data.get("config") or {}
    selection = config.get("example_selection")
    scoring = config.get("mc_scoring_rule")
    if selection == "seeded_sample_stratified_mmlu" and scoring == "acc_norm":
        return []

    return [
        "> ## ⚠️ 本报告数字已作废（SUPERSEDED，%s）" % _HARNESS_FIX_DATE,
        ">",
        "> 源工件未记录已修复的采样与打分设置"
        "（`example_selection` / `mc_scoring_rule`），",
        "> 说明它产生于存在缺陷的评测 harness。"
        "**其中的任务精度与 retention 数字不得对外引用。**",
        ">",
        "> 已定位的 disqualifying 缺陷：",
        "> 1. 样本选择为语料前缀截断，而 `mmlu_test.jsonl` 按 subject 排序、",
        ">    `anli_r3_test.jsonl` 按 label 排序；",
        "> 2. `piqa_val.jsonl` 全部 1838 行 gold label 均为 0（退化语料）；",
        "> 3. 多选打分未做长度归一化，系统性偏向短候选；",
        "> 4. 明文基线未与同模型公开参考值对账。",
        ">",
        "> 修复见 `scripts/run_osnip_style_benchmarks.py`；重跑并对账后方可引用。",
        "> 另见 `docs/threat_model.md` 第 5bis 节：精度/retention **不是隐私证据**。",
        "",
    ]


def build_chinese_report(data: Dict[str, Any]) -> str:
    ev = data.get("evaluation") or {}
    pairs = data.get("pairs") or []
    env = data.get("environment") or {}
    base = data.get("plaintext_base") or {}
    gates = data.get("gates") or {}
    diag = (data.get("aggregated_metrics") or {}).get("diagnostics") or {}

    n = int(ev.get("sample_count") or ev.get("prompts_per_key") or 0)
    n_keys = int(ev.get("n_keys") or data.get("n_keys") or 0)
    if not n_keys and pairs:
        n_keys = len({p.get("key", {}).get("label") for p in pairs})

    structural = [p for p in pairs if p.get("mode") == "structural"]
    full = [p for p in pairs if p.get("mode") == "full"]

    def col(items: List[Dict[str, Any]], key: str) -> List[float]:
        out = []
        for p in items:
            v = p.get(key)
            if v is not None:
                out.append(float(v))
        return out

    lines: List[str] = []
    lines.append("# Llama-3.2-3B-Instruct 明文 vs 当前 fastProve 混淆 — 中文评测报告")
    lines.append("")
    for line in _harness_validity_banner(data):
        lines.append(line)
    lines.append("> 本报告仅比较**明文**与**本仓库当前协变混淆方案**（structural / full）。")
    lines.append("> **不包含**旧版 ModelSplit 混淆对照。")
    lines.append(
        "> 术语：混淆态相对明文（实数增广协变混淆）。"
        "**不是** LWE-keyed / LWE-inspired：该构造在实数域上精确可逆，"
        "h 可由 c 恢复，详见 `docs/threat_model.md` 第 5bis 节。"
    )
    lines.append("")
    lines.append("## 1. 测试概览")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    lines.append("| 明文基座 | `%s` |" % base.get("root", "—"))
    lines.append(
        "| 明文校验 | %s |"
        % ("通过" if base.get("is_plaintext_verified") else "未通过/未知")
    )
    lines.append("| weights_sha256 | `%s` |" % (base.get("weights_sha256") or "—"))
    lines.append("| config_sha256 | `%s` |" % (base.get("config_sha256") or "—"))
    lines.append("| 每 key 的 Prompt 数 | **%d** |" % n)
    lines.append("| 独立 master key 数 | **%d** |" % n_keys)
    lines.append("| 模式 | %s |" % ", ".join(ev.get("modes") or ["structural", "full"]))
    lines.append("| 序列长度 / 生成长度 | %s / %s |" % (ev.get("sequence_length"), ev.get("generation_tokens")))
    lines.append("| dtype / device | %s / %s |" % (ev.get("dtype"), ev.get("device")))
    lines.append("| 真实场景 Prompt | %s |" % ("是" if ev.get("real_scenario") else "否"))
    lines.append("| 种子 | %s |" % ev.get("seed"))
    lines.append("| GPU | %s |" % env.get("gpu_name"))
    lines.append("| torch | %s |" % env.get("torch_version"))
    lines.append("| 运行状态 | %s |" % data.get("status", "—"))
    lines.append("| 原始结果 | `results/raw/llama_compare_large.json`（见下文路径） |")
    lines.append("")
    lines.append("### 数据量说明")
    lines.append("")
    lines.append(
        "- 每个 master key 使用**同一组** %d 条真实场景 Prompt（中英混合：知识问答、"
        "指令跟随、代码、推理、办公邮件、摘要、翻译等）。" % n
    )
    lines.append(
        "- 模式 × 密钥单元数：%d（应等于 模式数 × key 数）。" % len(pairs)
    )
    lines.append(
        "- Teacher-forced 有效 next-token 决策数（单 key 示例，取 pairs 中最大值）："
        " **%s**。"
        % max((int(p.get("n_valid_token_decisions") or 0) for p in pairs), default=0)
    )
    lines.append(
        "- Greedy：默认对全部 Prompt 去 padding 后生成对比（`greedy_n_samples` 应等于 %d）。"
        % n
    )
    lines.append("")

    lines.append("## 2. 精度差距总表（相对明文）")
    lines.append("")
    lines.append(
        "指标说明：**一致率**为混淆与明文 next-token argmax 相同的比例（越高越好）；"
        "**greedy 整段一致率**为生成后缀完全相同的样本比例；"
        "**ΔPPL_rel** = PPL_obf/PPL_plain − 1（相对增幅，越接近 0 越好）；"
        "**top1_drop_pp** 为相对标签的准确率百分点差（可与「相对明文一致率」区分）。"
    )
    lines.append("")
    lines.append(
        "| 模式 | key | top1 一致率 | greedy 整段一致 | greedy n | ΔPPL_rel | e2e logit max |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for p in pairs:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                p.get("mode"),
                (p.get("key") or {}).get("label"),
                _fmt(p.get("top1_token_agreement"), 4),
                _fmt(p.get("greedy_sequence_exact_match"), 4),
                p.get("greedy_n_samples"),
                _fmt(p.get("ppl_relative_increase"), 4),
                _fmt(p.get("e2e_logit_max_absolute_error"), 4),
            )
        )
    lines.append("")

    s_agree = _mean(col(structural, "top1_token_agreement"))
    f_agree = _mean(col(full, "top1_token_agreement"))
    s_greedy = _mean(col(structural, "greedy_sequence_exact_match"))
    f_greedy = _mean(col(full, "greedy_sequence_exact_match"))
    s_ppl = _mean(col(structural, "ppl_relative_increase"))
    f_ppl = _mean(col(full, "ppl_relative_increase"))

    lines.append("### 跨 key 平均")
    lines.append("")
    lines.append("| 模式 | 平均 top1 一致率 | 平均 greedy 整段一致 | 平均 ΔPPL_rel |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| structural | %s | %s | %s |"
        % (_fmt(s_agree, 4), _fmt(s_greedy, 4), _fmt(s_ppl, 4))
    )
    lines.append(
        "| full | %s | %s | %s |"
        % (_fmt(f_agree, 4), _fmt(f_greedy, 4), _fmt(f_ppl, 4))
    )
    lines.append("")
    if s_agree is not None:
        lines.append(
            "- **Structural 相对明文 token 分歧率** ≈ **%.2f%%**（1 − 一致率）。"
            % ((1.0 - s_agree) * 100.0)
        )
    if f_agree is not None:
        lines.append(
            "- **Full 相对明文 token 分歧率** ≈ **%.2f%%**。"
            % ((1.0 - f_agree) * 100.0)
        )
    lines.append("")

    lines.append("## 3. 门禁与数值诊断")
    lines.append("")
    lines.append(
        "- **ChainLinear 硬门禁**仅使用 FP32 单元恒等式，"
        "**不是**端到端 BF16 logit 误差。"
    )
    lines.append(
        "- ⚠️ **门禁范围警告**：该门禁测试独立的 `ChainLinear` 模块，"
        "而被评测的 Llama 路径**并不实例化它**"
        "（`models/obfuscated.py` 直接用 `@ self.*_weight_math`，零引用 ChainLinear）。"
        "因此 PASS 只验证了孤立的仿射刷新恒等式，"
        "**不代表**被测模型实际执行的仿射代数。"
    )
    lines.append(
        "- 端到端 logit max|obf−plain|（诊断）：**%s**"
        % _fmt(diag.get("e2e_logit_max_absolute_error"), 4)
    )
    lines.append(
        "- ChainLinear 单元 max abs error：**%s**"
        % _fmt(diag.get("chain_linear_unit_max_absolute_error"), 6)
    )
    if gates:
        lines.append("- hard_passed：**%s**；soft_passed：**%s**"
                     % (gates.get("hard_passed"), gates.get("soft_passed")))
        lines.append("")
        lines.append("```")
        for r in gates.get("results") or []:
            status = "PASS" if r.get("passed") else "FAIL"
            lines.append(
                "%s | %s | observed=%s threshold=%s"
                % (r.get("name"), status, r.get("observed"), r.get("threshold"))
            )
        lines.append("```")
    else:
        lines.append("- 本原始文件未嵌入 gates 表时，以运行日志 `--check-gates` 输出为准。")
    lines.append("")
    lines.append("## 4. 结论（面向工程）")
    lines.append("")
    lines.append(
        "1. 大规模真实场景下，当前 fastProve 相对明文存在可测量的 next-token 不一致；"
        "structural（无噪声注入）通常优于 full（完整噪声链）。"
    )
    lines.append(
        "2. 若 greedy 整段一致率显著低于 teacher-forced 一致率，说明自回归会放大早期分歧。"
    )
    lines.append(
        "3. 硬门禁（argmax/greedy 100% 等）未通过时，**不得**宣称「零精度损失」或「已达发布级数学等价」。"
    )
    lines.append(
        "4. 本结果**不能**外推为 MMLU/C-Eval 等任务榜单；亦**不能**推出密码学安全保证。"
    )
    lines.append("")
    lines.append("## 5. 威胁模型与非声明（AGENTS.md）")
    lines.append("")
    lines.append(
        "- 原型针对诚实但好奇的观察者（持久化张量/常规框架输出），"
        "假定指定融合算子不回传内部干净临时量——这是实现假设，不是密码学保证。"
    )
    lines.append(
        "- 不声称：标准 LWE 密文、全同态/端到端加密推理、"
        "仅因 ML-KEM/LWE 风格种子即具备 LWE 安全性、"
        "抵御可任意改内核/挂钩/转储寄存器的服务器。"
    )
    lines.append(
        "- **已量化上界**（`docs/threat_model.md` 第 5bis 节）：当前实数可逆构造下 "
        "`h = c · (M⁻¹)[:, :d]` 对任意噪声精确成立（实测 ~1e-15），"
        "故噪声强度、刷新档位与密钥派生方式**均不改变** h 的可恢复性；"
        "`structural` 与 `full` 在此意义下同为 0。"
        "复核：`PYTHONPATH=src python3 scripts/verify_recoverability_bound.py`。"
    )
    lines.append(
        "- 因此本报告的精度/一致率/PPL 数字**只度量效用**，不构成隐私证据；"
        "`full` 相对 `structural` 多掉的精度在当前威胁模型下未换到可度量的隐私收益。"
    )
    lines.append("")
    lines.append("## 6. 原始产物路径")
    lines.append("")
    lines.append("- 机器可读结果：见本报告生成时输入的 JSON 路径（通常 `results/raw/llama_compare_large.json`）")
    lines.append("- 英文/混合摘要：`results/REPORT.md`（若存在）")
    lines.append("- 本中文报告：`results/REPORT_zh.md`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告由 `scripts/write_chinese_report.py` 从原始 JSON 自动生成，数字可追溯。*")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw/llama_compare_large.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/REPORT_zh.md",
    )
    args = parser.parse_args()
    inp = Path(args.input)
    if not inp.is_file():
        # fall back to smaller run name
        alt = Path("results/raw/llama_compare_P.json")
        if alt.is_file():
            inp = alt
        else:
            raise SystemExit("input not found: %s" % args.input)
    data = json.loads(inp.read_text(encoding="utf-8"))
    report = build_chinese_report(data)
    # inject concrete path
    report = report.replace(
        "通常 `results/raw/llama_compare_large.json`",
        "`%s`" % inp.as_posix(),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("wrote %s from %s (samples=%s pairs=%s)"
          % (out, inp, (data.get("evaluation") or {}).get("sample_count"), len(data.get("pairs") or [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
