# Llama-3.2-3B-Instruct: plaintext vs current fastProve

> ## ⚠️ SUPERSEDED (2026-08-06) — do not quote these numbers
>
> Every task-accuracy and retention figure below was produced by a defective
> evaluation harness. Disqualifying defects, all confirmed against the local
> corpora:
>
> 1. **Example selection was corpus head-truncation** while several corpora are
>    ordered: `mmlu_test.jsonl` is sorted by subject (the first 200 rows cover
>    only `abstract_algebra` and `anatomy` out of 57 subjects) and
>    `anli_r3_test.jsonl` is sorted by label (6 runs in the first 200 rows where
>    a shuffled corpus would give ~134).
> 2. **PIQA gold labels are degenerate**: all 1838 rows of `piqa_val.jsonl` have
>    `label == 0`, so its "accuracy" is just the rate of picking `sol1` and is
>    identical across plain/structural/full, contributing a hard-coded 100%
>    retention to the average.
> 3. **Multiple-choice scoring was not length-normalized** (argmax over summed
>    logprob), which systematically favours short candidates.
> 4. **The plaintext baseline was never reconciled with a public reference**:
>    this repo scores 0.493 average where the OSNIP paper reports 0.597
>    non-private for the same Llama-3.2-3B-Instruct — a 10.4 pp deficit. Retention
>    computed on top of a mis-calibrated baseline carries no information.
>
> Fixes landed in `scripts/run_osnip_style_benchmarks.py` (seeded sampling,
> subject-stratified MMLU, degenerate-label rejection, `acc_norm`, Wilson
> intervals and chance-floor flags). A citable report requires a re-run whose
> plaintext baseline reconciles with the public reference.
>
> See also [`docs/threat_model.md`](../docs/threat_model.md) §5bis: no accuracy
> or retention number in this report is evidence of privacy.

## Scope
- Only plaintext base vs **current** fastProve conversion (structural / full).
- Legacy ModelSplit obfuscation was **not** used as baseline or result.
- Terminology: obfuscated-state relative to plaintext (LWE-keyed / LWE-inspired covariant obfuscation).

## Environment
```json
{
  "python_version": "3.12.13",
  "torch_version": "2.12.0+cu130",
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.43",
  "cuda_available": true,
  "cuda_version": "13.0",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "gpu_memory_total_bytes": 33668857856
}
```

conda env: `fastprove` (offline clone of modelsplit when network blocked).

## Plaintext base
- path: `/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct`
- architecture: `LlamaForCausalLM`
- weights_sha256: `963e552937a68f2166e65810a13ccc5b9d3a173517ff58ed9ebd2b36a5fe557e`
- config_sha256: `39fb36dc5416f445ebc4e71cb71fbcf6727e80a35836d8ba1a1474c318467b7a`
- is_plaintext_verified: True
- verification notes: ['tie_word_embeddings=true; lm_head shares embed_tokens', 'standard LlamaForCausalLM weight map; no legacy markers', 'NOT a legacy ModelSplit obfuscation artifact; safe as plaintext base']

## Evaluation design
- sample_ids (shared by all pairs): ['prompt-0000', 'prompt-0001', 'prompt-0002', 'prompt-0003']
- sequence_length=32, gen_tokens=4
- dtype=bfloat16, device=cuda
- master keys: 3 independent conversion seeds
- greedy: **all** sample_ids; padding stripped so `generate_greedy` last index is last real token

## Metric sources (do not confuse)
- **ChainLinear hard gate**: FP32 unit identity only (`_chain_linear_identity_test`).
- **E2E logit max abs**: teacher-forced plain vs obfuscated activation logits (diagnostic; BF16).
- Live run prints both; ChainLinear gate is **not** filled from e2e logits.
- chain_linear_unit_max_absolute_error: 8.940696716308594e-07
- e2e_logit_max_absolute_error: 1.32421875

## Results (current scheme only)

| mode | key | top1 agree | greedy seq | greedy n | Delta PPL_rel | top1 drop (pp) | e2e logit max |
|---|---|---:|---:|---:|---:|---:|---:|
| structural | key-00 | 0.925926 | 1 | 4 | 0.005781370813690566 | -3.704 | 1.05469 |
| structural | key-01 | 0.925926 | 1 | 4 | 0.000325017330006045 | -3.704 | 0.546875 |
| structural | key-02 | 0.925926 | 1 | 4 | -0.004691528292902714 | -3.704 | 1.36719 |
| full | key-00 | 0.962963 | 0.75 | 4 | 0.020058551643361722 | 0 | 1.32422 |
| full | key-01 | 0.925926 | 1 | 4 | -0.020252240569865396 | -3.704 | 2.30469 |
| full | key-02 | 0.925926 | 0.75 | 4 | 0.08105571744810214 | -3.704 | 1.625 |

Units: agreement fraction; top1 drop in **pp**; PPL **relative** increase.

## Gates (primary = first full-mode pair; live `--check-gates`)
```
Gate                                      | Cat  | Status | Observed | Threshold
------------------------------------------------------------------------------------------
FP32 ChainLinear max abs error             | hard | PASS   | 8.9407e-07 | 0.0001
Attention rank-flip rate                   | hard | PASS   | n/a | 0
Attention top-1 match                      | hard | PASS   | n/a | 1
Attention top-k overlap                    | hard | PASS   | n/a | 1
MoE expert set match                       | hard | PASS   | n/a | 1
Causal mask match                          | hard | PASS   | n/a | 1
Cache vs no-cache identical                | hard | PASS   | n/a | 1
LM Head inverse-permuted argmax            | hard | FAIL   | 0.962963 | 1
Greedy release-validation sequences        | hard | FAIL   | 0.75 | 1
Accuracy drop ≤ 0.5 pp                     | soft | PASS   | 0 | 0.5
PPL relative increase ≤ 1%                 | soft | FAIL   | 0.0200586 | 0.01
------------------------------------------------------------------------------------------
HARD: FAIL   SOFT: FAIL   ALL: FAIL
```

## Source notes recorded in JSON
- hard_gate_chain_linear: FP32 unit identity only (evals.layer1_operator_equiv._chain_linear_identity_test); never BF16 e2e logits
- e2e_logit_max_absolute_error: diagnostic from teacher-forced plain vs obf logits on the eval set
- greedy: all sample_ids; pad stripped so generate_greedy last index is last real token

## Artifacts
- results/raw/llama_compare_P.json
- results/raw/llama_compare_run.log

