# Archived report writers

状态：CURRENT
日期：2026-09-19
上游依据：docs/CLEANUP_PLAN.md §P6
适用范围：scripts/archive/ 中归档报告脚本的清单与冻结原因

These scripts are **archived, unchecked-in historical writers**. They are kept
only for provenance of report artifacts that were produced before the current
pipeline existed; they are not maintained and must not be wired back into the
test suite, CI, or the documented runbook.

| Script | Why it is archived |
|---|---|
| `postprocess_llama_report.py` | No CLI, hard-coded `results/raw/llama_compare_P.json`, rewrites `results/REPORT.md` wholesale, and imports private `evals` symbols. No longer referenced anywhere. |
| `write_chinese_report.py` | Third independent report writer (writes `docs/archive/2026-08-06_REPORT_zh.md` from older compare JSON). Superseded by `scripts/build_report.py`. |

Neither file is imported by repository code. Verify with:

```bash
rg -n "postprocess_llama_report|write_chinese_report" scripts src tests evals
```

Only this README and the files themselves should match.

They read `results/raw/*` inputs that may no longer exist, so they are not
executed as part of any acceptance check. If one is ever needed again, restore
it to `scripts/` with a CLI and an explicit output guard rather than editing it
in place here.
