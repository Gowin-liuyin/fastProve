# fastProve Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch reference prototype that compares plaintext, exact augmented-covariant, Top-k-preserving approximate, and free-bounded approximate Transformer inference.

**Architecture:** A small Llama-like causal LM owns the canonical plaintext weights. The obfuscated model is converted from those weights and keeps hidden/value states in well-conditioned augmented bases between explicit nonlinear checkpoints; exact checkpoints expose only final model outputs in the production API. A deterministic evaluation layer runs aligned teacher-forced and greedy comparisons, writes one machine-readable record per configuration, and derives tables, plots, and a report without inventing missing pretrained evidence.

**Tech Stack:** Python 3.9, PyTorch 2.8, PyYAML, NumPy, Matplotlib, pytest.

**Workspace note:** The requested directory contains no Git repository. Work therefore remains in `/Users/yin/code/fastProve`; worktree and commit steps are intentionally omitted instead of silently initializing version control.

---

## File structure

- `pyproject.toml`: packaging, dependency floor, and pytest settings.
- `configs/*.yaml`: tiny correctness and full required noise sweep configuration.
- `src/fastprove/config.py`: validated dataclass configuration loading.
- `src/fastprove/seed.py`: SHA-256 domain-separated deterministic seeds and generators.
- `src/fastprove/state.py`: opaque mixed-state container plus explicitly gated debug helpers.
- `src/fastprove/transforms.py`: orthogonal and bounded-condition transform generation.
- `src/fastprove/conversion.py`: row-math/PyTorch-layout conversion and offline affine conversion.
- `src/fastprove/layers/*.py`: linear, RMSNorm/RoPE, attention, SwiGLU, and router kernels.
- `src/fastprove/models/plain.py`: canonical tiny Llama-like reference weights and forward path.
- `src/fastprove/models/obfuscated.py`: converted exact/approximate path and fused-reference checkpoints.
- `src/fastprove/evaluation/*.py`: correctness, aligned accuracy, softmax, and performance metrics.
- `scripts/*.py`: correctness, sweep, and report entry points.
- `tests/*.py`: deterministic mathematical, edge-case, block, LM, and artifact tests.
- `results/`: raw records, derived CSV, figures, and generated report.
- `docs/*.md`: mathematics, threat model, and implementation/environment record.

### Task 0: Environment record and dependency scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `docs/implementation_notes.md`
- Create: `configs/tiny_exact.yaml`
- Create: `configs/tiny_approx.yaml`
- Create: `configs/eval_sweep.yaml`

- [ ] Record the exact Phase 0 commands and outputs: Python/PyTorch, CUDA/MPS, CPU/GPU, RAM, disk, installed dependencies, Git absence, and local model/data audit.
- [ ] Set `requires-python = ">=3.9"` and keep runtime dependencies limited to installed PyTorch/PyYAML/NumPy/Matplotlib; make `pytest` the test extra.
- [ ] Put the required `tau_max`, `alpha`, and `preserve_top_k` arrays in YAML, not layer code.
- [ ] Validate YAML syntax with:

```bash
python3 -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('configs').glob('*.yaml')]"
```

Expected: exit 0.

### Task 1: Seed, transforms, state, and ChainLinear identity

**Files:**
- Create: `tests/test_seed_transforms.py`
- Create: `tests/test_chain_linear.py`
- Create: `src/fastprove/seed.py`
- Create: `src/fastprove/transforms.py`
- Create: `src/fastprove/state.py`
- Create: `src/fastprove/conversion.py`
- Create: `src/fastprove/layers/linear.py`

- [ ] Write failing tests that require domain separation, repeatability, round-trip encoding, condition number enforcement, layout round trips, and the direct affine identity:

```python
def test_chain_linear_matches_augmented_affine_identity() -> None:
    expected_signal = h @ w_math + bias
    expected_noise = h @ coupling + noise @ propagator + refresh
    expected_mixed = torch.cat((expected_signal, expected_noise), dim=-1) @ out_transform.matrix
    actual = layer(encode_debug(h, noise, in_transform, enabled=True)).mixed
    assert torch.max(torch.abs(actual - expected_mixed)).item() <= 1e-5
```

- [ ] Run:

```bash
python3 -m pytest tests/test_seed_transforms.py tests/test_chain_linear.py -q
```

Expected RED: import failure for missing `fastprove` modules.

- [ ] Implement `derive_seed`, `make_generator`, `BasisTransform`, `generate_transform`, `generate_orthogonal`, `generate_signed_permutation`, `MixedState`, `encode_debug`, `decode_debug`, `math_to_torch_weight`, `torch_to_math_weight`, `convert_affine_chain`, and `ChainLinear`.
- [ ] Compute and register the inverse only during conversion. `ChainLinear.forward` may call only `torch.nn.functional.linear` plus a deterministic per-request refresh addition; it must not call `torch.linalg.inv` or `torch.inverse`.
- [ ] Distinguish `fixed_debug` from `per_request`; require a request seed for the latter and derive refresh from `(request_seed, layer_id, "refresh")`.
- [ ] Re-run the two test files and require all tests to pass with the FP32 signal maximum error at most `1e-5`.

### Task 2: RMSNorm, RoPE, masking, and four-mode attention

**Files:**
- Create: `tests/test_rmsnorm_rope.py`
- Create: `tests/test_attention.py`
- Create: `src/fastprove/layers/rmsnorm.py`
- Create: `src/fastprove/layers/attention.py`
- Create: `src/fastprove/evaluation/metrics.py`

- [ ] Write failing tests for FP32 RMS statistics, absorbed gamma projection, RoPE-before-common-orthogonal Q/K covariance, GQA head mapping, causal and padding masks, fully masked rows, deterministic noise, masked `-inf` preservation, infinity-norm bounds, strict Top-k preservation away from ties, explicit tie-to-zero behavior, free-bounded rank changes, and debug-only probability capture.
- [ ] Use the absorbed projection assertion:

```python
plain = rms_norm(h, gamma, eps) @ w_math
u = h @ rotation
absorbed = rotation.T @ torch.diag(gamma) @ w_math
converted = gamma_free_rms_norm(u, eps) @ absorbed
torch.testing.assert_close(converted, plain, atol=2e-5, rtol=2e-5)
```

- [ ] Run:

```bash
python3 -m pytest tests/test_rmsnorm_rope.py tests/test_attention.py -q
```

Expected RED: missing RMSNorm and attention APIs.

- [ ] Implement FP32 `rms_norm`, `gamma_free_rms_norm`, `absorbed_rms_projection`, `apply_rope`, `AttentionMode`, `ApproximationConfig`, `AttentionKVCache`, and `ObfuscatedAttention`.
- [ ] Build masks before noise. Use a safe masked Softmax that returns an all-zero probability row for a fully masked query while keeping every masked logit exactly `-inf`.
- [ ] For each valid query row, compute `tau = min(tau_max, alpha * margin / 2, tau_error)` in Top-k mode and `tau = min(tau_max, tau_error)` in free mode. Center sampled noise over valid positions, rescale it, and assert its realized infinity norm is no larger than the row budget.
- [ ] Keep production `forward` output limited to the attention output/cache. Put probabilities, clean/noisy scores, masks, margins, and noise only in `forward_debug`, guarded by `debug_enabled`.
- [ ] Implement tensor error, KL, JS, Top-k overlap/change fraction, valid-position rank correlation, actual-noise, zero-noise, and NaN/Inf metrics.
- [ ] Re-run the tests and require all to pass.

### Task 3: SwiGLU covariance and MoE router edge behavior

**Files:**
- Create: `tests/test_swiglu.py`
- Create: `tests/test_router.py`
- Create: `src/fastprove/layers/swiglu.py`
- Create: `src/fastprove/layers/router.py`

- [ ] Write failing tests for the exact identity:

```python
plain = torch.nn.functional.silu(gate) * up
plain = plain @ down
gate_prime = gate[:, permutation]
up_prime = (up * scale)[:, permutation]
down_prime = inverse_permuted_scaled_down(down, permutation, scale)
converted = (torch.nn.functional.silu(gate_prime) * up_prime) @ down_prime
torch.testing.assert_close(converted, plain, atol=2e-5, rtol=2e-5)
```

- [ ] Add router tests for exact expert permutation, margin-bounded set preservation, free noisy weights, deterministic tie-breaking, and physical expert reorder mapping.
- [ ] Run:

```bash
python3 -m pytest tests/test_swiglu.py tests/test_router.py -q
```

Expected RED: missing SwiGLU/router APIs.

- [ ] Implement shared neuron permutation and bounded nonzero diagonal scale with the down-weight compensation `P_f^T D_f^-1 W_d`.
- [ ] Implement a nonlinear checkpoint result carrying `e_z = z' C_z + e_side G_z + xi_z`; test that old nonzero noise contributes when refresh is zero and that refresh regenerates noise when the side state is zero.
- [ ] Implement stable-index tie-breaking in the router and reorder expert outputs consistently.
- [ ] Re-run the two test files and require all to pass.

### Task 4: Canonical plaintext decoder block

**Files:**
- Create: `tests/test_plain_block.py`
- Create: `src/fastprove/config.py`
- Create: `src/fastprove/models/plain.py`

- [ ] Write failing tests for shapes, deterministic initialization, RMSNorm → Q/K/V → RoPE → causal attention → output/residual → RMSNorm → SwiGLU/down → residual order, GQA, padding, and one-token cache equivalence.
- [ ] Run:

```bash
python3 -m pytest tests/test_plain_block.py -q
```

Expected RED: missing `PlainDecoderBlock`.

- [ ] Implement validated Python 3.9 dataclass configs and a canonical `PlainBlockWeights` module using explicit PyTorch `[out, in]` weights while conversion accessors return row-math `[in, out]`.
- [ ] Implement `PlainDecoderBlock.forward` and cached single-token decoding using FP32 logits/Softmax and configurable activation dtype.
- [ ] Re-run the test and require all cases to pass.

### Task 5: Exact and approximate obfuscated decoder block

**Files:**
- Create: `tests/test_obfuscated_block.py`
- Create: `src/fastprove/models/obfuscated.py`
- Create: `src/fastprove/evaluation/correctness.py`

- [ ] Write failing tests that convert exactly the plaintext weights, compare permitted decoded checkpoints, assert QK score error and exact Softmax error, assert final exact output tolerance, verify old noise survives nonlinear side paths, and ensure the production output has no decoded hidden/noise/probability fields.
- [ ] Run:

```bash
python3 -m pytest tests/test_obfuscated_block.py -q
```

Expected RED: missing converter/model.

- [ ] Implement `ObfuscatedDecoderBlock.from_plain`. At each fused-reference checkpoint decode only internally, preserve side noise, apply mathematically valid signal operations, refresh noise, and return a new `MixedState`.
- [ ] Apply the shared Q/K orthogonal map after RoPE. Encode each Value head as `[V,e_v]M_v`, multiply the mixed Value from the left by attention probabilities, and decode only inside the output-projection checkpoint.
- [ ] Use absorbed RMS gamma matrices for Q/K/V and Gate/Up. Use the Task 3 SwiGLU compensation for Down.
- [ ] Implement `run_block_correctness` to emit all required layer metrics without exposing them through production forwards.
- [ ] Re-run the test and require exact FP32 output to meet the documented tolerance before starting Tiny LM work.

### Task 6: Tiny causal LM and aligned evaluation

**Files:**
- Create: `tests/test_tiny_lm.py`
- Create: `tests/test_accuracy_metrics.py`
- Create: `src/fastprove/models/__init__.py`
- Create: `src/fastprove/evaluation/accuracy.py`

- [ ] Write failing tests for shared base weights, aligned teacher-forced inputs, NLL/perplexity/top-1/top-5, plaintext/obfuscated argmax agreement, greedy token exact match, sequence match, fixed-seed reproducibility, no NaN/Inf, and exact-mode greedy equality.
- [ ] Run:

```bash
python3 -m pytest tests/test_tiny_lm.py tests/test_accuracy_metrics.py -q
```

Expected RED: missing Tiny LM and evaluator.

- [ ] Implement `PlainTinyCausalLM` and `ObfuscatedTinyCausalLM.from_plain` with token embedding, multiple decoder blocks, final FP32 RMSNorm, and LM head.
- [ ] Cache deterministic synthetic token sequences and sample IDs once, then reuse the identical tensor/order for all modes.
- [ ] Implement teacher-forced and greedy evaluation plus conversion time, prefill latency, decode TPOT/tokens-per-second, peak-process memory, and explicit reference KV-cache byte accounting.
- [ ] Re-run the tests. If any exact greedy token differs, emit the offending position/logit margin and stop before approximate conclusions.

### Task 7: Reproducible scripts, sweep, and artifacts

**Files:**
- Create: `tests/test_artifacts.py`
- Create: `scripts/run_correctness.py`
- Create: `scripts/run_accuracy_sweep.py`
- Create: `scripts/build_report.py`
- Create: `README.md`
- Create: `docs/mathematics.md`
- Create: `docs/threat_model.md`

- [ ] Write a failing artifact test requiring successful/failure status, config, seed, model/data identifiers, environment, sample count, metrics, elapsed time, and unique run ID in each JSONL record.
- [ ] Run:

```bash
python3 -m pytest tests/test_artifacts.py -q
```

Expected RED: script/artifact functions are absent.

- [ ] Implement correctness output and a sweep enumerating plaintext, exact, all 63 Top-k configurations (`7 tau × 3 alpha × 3 k`), and all 7 free-bounded tau points without omitting failures.
- [ ] Implement report derivation from raw records only: summary CSV, accuracy/noise plots, ranking/Top-k plots, performance table, best-tradeoff rule, and `results/REPORT.md`.
- [ ] State prominently that synthetic random-model metrics establish software correctness only, not language capability.
- [ ] Run:

```bash
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml
python3 scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml
python3 scripts/build_report.py --raw results/raw --output results
```

Expected: exit 0; raw JSON/JSONL, CSV, PNG, and report are present.

### Task 8: Local pretrained evaluation gate

**Files:**
- Create: `src/fastprove/evaluation/pretrained.py`
- Create or update: `results/raw/pretrained_evaluation_status.json`
- Update: `docs/implementation_notes.md`
- Update: `results/REPORT.md`

- [ ] If a complete local Llama-like checkpoint, tokenizer, and public evaluation data are found, hash/identify them and write a failing adapter/alignment test before loading.
- [ ] If the three required local assets are not all present, write a machine-readable `skipped_external_dependency` record with searched cache roots and exact missing items; do not download and do not relabel synthetic results.
- [ ] Record a minimal proposed external run (small model, dataset sample count, storage, RAM/device estimate) and make that external run the only remaining permission-gated step.

### Task 9: Full verification and requirement audit

**Files:**
- Update: `docs/superpowers/plans/2026-07-31-fastprove-prototype.md`
- Update generated artifacts only by re-running their generators.

- [ ] Re-index the final repository with codebase-memory-mcp and inspect architecture/call paths for production debug leakage and inverse calls.
- [ ] Run:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml
python3 scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml
python3 scripts/build_report.py --raw results/raw --output results
```

Expected: every command exits 0; all configured points have a record or explicit failure record.

- [ ] Audit every AGENTS.md acceptance criterion against fresh output and generated artifact paths.
- [ ] Run a final independent specification review followed by a code-quality review; fix and re-run until no blocking issue remains.

## Execution selection

The user explicitly requested continuous autonomous execution and no pause for low-risk choices. Use subagent-driven execution where tasks do not share files, with the root agent integrating and independently verifying every result.

On 2026-07-31 the user narrowed the current run to prototype implementation and experiment planning. Tasks 1–6 and the implementation portions of Tasks 7–8 remain in scope, together with unit/correctness tests. The configured accuracy sweep, pretrained evaluation, and long-running performance experiments must not be executed in this run; their commands and artifact contracts are prepared for a later authorized run.

## Audit addendum (2026-08-01)

The implementation and control-plane portions of Tasks 0–7 are present and are
covered by the current test suite. Task 8 now has a machine-readable asset audit:
`results/raw/pretrained_evaluation_status.json` records the complete local Qwen2
checkpoint/tokenizer, the missing standard causal-LM evidence, the discovered
non-standard Flickr30k caption candidate, and a proposed 8/64-sample run. The
caption candidate is materialized as
`results/raw/flickr30k_caption_eval_inputs.pt` but is rejected by the runner unless
`--accept-nonstandard-caption` is explicit. Formal inference, noise sweep and
report generation remain intentionally unexecuted under the current user scope.

Latest verification: `python3 -m pytest -q` → 209 passed; `python3 -m compileall
-q src evals scripts` → success; codebase-memory index refreshed with 2,123 nodes
and 6,534 edges. The sweep CLI now records explicit device/dtype overrides,
orders exact-gate baselines before approximate points, and
caption-only records use comparative evidence rather than standard LM evidence.
The remaining external decision is the evaluation scope/data authorization (and,
if required, compute with enough memory for FP32 exact gate).
