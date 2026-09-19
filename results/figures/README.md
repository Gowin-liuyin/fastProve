# results/figures 目录清单与溯源

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/INDEX.md、results/raw/README.md
适用范围：`results/figures/` 的实际跟踪图形清单、来源与口径警告

本目录的当前目标是保存**已跟踪、已编译**的 LaTeX/PDF 图表；`*.png`、`*.aux`、
`*.log`、`*.out` 均被 `.gitignore` 排除（`.aux/.log/.out` 为编译中间物，PNG 为旧渲染）。
仓库内**没有**生成这些图的 Python 脚本：`.tex` 由外部 GPU 运行的手工整理材料写成，
生成脚本 `待考`。所有图表基于自标 superseded 的 OSNIP/LLaMA 对照记录
（`results/raw/osnip_style_bench_*.json`、`llama_compare_*.json`），
**引用其中的任务精度数字前必须连同该记录的 superseded caveat 一起说明**。

## 1. 已跟踪图形

| 文件 | 内容 | 来源/生成 | 跟踪 | 备注 |
|---|---|---|---|---|
| `osnip_style_table1.tex` / `.pdf` | OSNIP Table-1 风格的主结果表（Llama-3.2-3B-Instruct，10 个闭式任务，plain/structural/full + RP） | 手写 LaTeX（编译：`xelatex osnip_style_table1.tex`），数据取自 `results/raw/osnip_style_bench_fixed.json` 一代记录；无仓库内生成脚本（待考） | 跟踪 | 表内注记为 `n=200, BF16, RTX 5090`；源记录自标 superseded |
| `osnip_vs_fastprove_compare.tex` / `.pdf` | OSNIP 论文 vs fastProve 的对比图（pgfplots） | 手写 LaTeX（编译：`pdflatex osnip_vs_fastprove_compare.tex`）；待考 | 跟踪 | 只作历史对照展示 |
| `osnip_vs_fastprove_table.tex` / `.pdf` | 两栏对比表（OSNIP vs fastProve） | 手写 LaTeX（编译：`xelatex osnip_vs_fastprove_table.tex`）；待考 | 跟踪 | 同上 |
| `osnip_vs_fastprove_tasks.tex` / `.pdf` | 逐任务指标柱状图（OSNIP 论文值 vs fastProve 实测值） | 手写 LaTeX（编译：`xelatex osnip_vs_fastprove_tasks.tex`）；待考 | 跟踪 | 同上 |
| `evaluation_summary_zh.pdf` | 中文评测汇总图（PDF） | 来源待考（仓库内无对应 `.tex`/脚本） | 跟踪 | 不得作为当前结论引用 |

## 2. 未跟踪产物（本地存在、不入库）

| 文件 | 说明 |
|---|---|
| `osnip_style_table1.png`、`osnip_style_table1-1.png` | 旧渲染 PNG；`.gitignore` 排除 |
| `osnip_style_table1.aux/.log`、`osnip_vs_fastprove_*.aux/.log/.out` | LaTeX 编译中间物；`.gitignore` 排除（P1 已 `git rm --cached`） |

## 3. 生成规则

- 噪声—精度曲线与 Softmax 图必须从**已验证的 raw 记录**生成，不同
  `(alpha, preserve_top_k)` 使用独立曲线，成功点用不连线散点；
- 当前仓库内的报告管线（`scripts/build_report.py --execute`）写
  `results/tables/`、`results/figures/` 与 `results/REPORT.md`；上述历史 LaTeX 图
  不参与该管线；
- 随机 Tiny 记录不得用来出精度图；superseded 记录出图必须标注。
