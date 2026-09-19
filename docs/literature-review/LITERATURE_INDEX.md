# 文献清单（PDF 已移出仓库）

状态：CURRENT  
日期：2026-09-19  
上游依据：docs/CLEANUP_PLAN.md §10-③（用户已批准）

本目录原有 38 篇 PDF（约 72 MB，占当时 git 跟踪体积 96%），于 2026-09-19 移出跟踪以缩小仓库体积。
全文仍保留在 git 历史中，可按需恢复：

```bash
# 恢复一篇（示例）
git show 5e19c43:docs/literature-review/papers/quantization/GPTQ_2022.pdf > /tmp/GPTQ_2022.pdf
```

- 精读结论见 `docs/literature-review/reports/`，下载前先查已有笔记。
- 新增文献只提交本清单条目，不要提交 PDF 文件。

## attention-approx

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| FlashAttention_2022.pdf | 2205.14135 | — | https://arxiv.org/abs/2205.14135 |
| Performers_2020.pdf | 2009.14794 | — | https://arxiv.org/abs/2009.14794 |

## dp

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| PATE_2018.pdf | 1802.08908 | — | https://arxiv.org/abs/1802.08908 |

## formal-verify

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| QVIP_2022.pdf | 2212.11138 | 10.1145/3551349.3556916 | https://arxiv.org/abs/2212.11138 |
| ReluDiff_2020.pdf | 2001.03662 | 10.1145/3377811.3380337 | https://arxiv.org/abs/2001.03662 |

## he-mpc

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| CHET_2021.pdf | 1810.00845 | — | https://arxiv.org/abs/1810.00845 |
| Cheetah_2021.pdf | 2006.00505 | — | https://arxiv.org/abs/2006.00505 |
| CrypTFlow2_2020.pdf | 2010.06457 | — | https://arxiv.org/abs/2010.06457 |
| Gazelle_2018.pdf | 1801.05507 | — | https://arxiv.org/abs/1801.05507 |
| MPC_Pipe_2022.pdf | 2209.13643 | — | https://arxiv.org/abs/2209.13643 |
| SIRNN_2021.pdf | 2105.04236 | — | https://arxiv.org/abs/2105.04236 |
| SplitHE_2022.pdf | 2202.13351 | — | https://arxiv.org/abs/2202.13351 |
| Tabula_2022.pdf | 2203.02833 | — | https://arxiv.org/abs/2203.02833 |
| nGraphHE2_2019.pdf | 1908.04172 | — | https://arxiv.org/abs/1908.04172 |
| nGraphHE_2018.pdf | 1810.10121 | — | https://arxiv.org/abs/1810.10121 |

## llm-private

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| AERO_2024.pdf | 2410.13060 | — | https://arxiv.org/abs/2410.13060 |
| ARIANN_2020.pdf | 2006.04593 | — | https://arxiv.org/abs/2006.04593 |
| Breaking_Layer_Barrier_2025.pdf | 2508.19525 | 10.5281/zenodo.15590214 | https://arxiv.org/abs/2508.19525 |
| ConfusionPrompt_2024.pdf | 2401.00870 | — | https://arxiv.org/abs/2401.00870 |
| FastQuery_2024.pdf | 2405.16241 | 10.1145/3649329.3657374 | https://arxiv.org/abs/2405.16241 |
| LDP_ICL_2024.pdf | 2405.04032 | — | https://arxiv.org/abs/2405.04032 |
| MPC_Minimized_LLM_2024.pdf | 2408.03561 | — | https://arxiv.org/abs/2408.03561 |
| ObfuscaTune_2024.pdf | 2407.02960 | — | https://arxiv.org/abs/2407.02960 |
| OpenLLM_Private_2024.pdf | 2411.05818 | — | https://arxiv.org/abs/2411.05818 |
| PermLLM_2024.pdf | 2405.18744 | — | https://arxiv.org/abs/2405.18744 |
| PrivCirNet_2024.pdf | 2405.14569 | — | https://arxiv.org/abs/2405.14569 |
| Private_LLM_Collaborative_2024.pdf | 2412.16537 | — | https://arxiv.org/abs/2412.16537 |
| TruncFormer_2024.pdf | 2412.01042 | — | https://arxiv.org/abs/2412.01042 |

## quantization

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| AdaRound_2020.pdf | 2004.10568 | — | https://arxiv.org/abs/2004.10568 |
| GPTQ_2022.pdf | 2210.17323 | — | https://arxiv.org/abs/2210.17323 |
| INT4_Transformer_2023.pdf | 2301.12017 | — | https://arxiv.org/abs/2301.12017 |
| LSQ_2020.pdf | 1902.08153 | — | https://arxiv.org/abs/1902.08153 |
| Oscillation_free_ViT_2023.pdf | 2302.02210 | — | https://arxiv.org/abs/2302.02210 |
| QAT_NLU_2023.pdf | 2306.01076 | — | https://arxiv.org/abs/2306.01076 |
| QAT_Transformer_2023.pdf | 2302.11812 | — | https://arxiv.org/abs/2302.11812 |
| Quantized_LLM_Edge_2023.pdf | 2310.03971 | — | https://arxiv.org/abs/2310.03971 |
| Sparsity_Quant_ViT_2023.pdf | 2305.10727 | — | https://arxiv.org/abs/2305.10727 |

## tee

| 论文（原文件名） | arXiv | DOI | 链接 |
|---|---|---|---|
| Slalom_2019.pdf | 1806.03287 | — | https://arxiv.org/abs/1806.03287 |
