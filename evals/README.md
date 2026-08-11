# fastProve evaluation suite

Protocol-aligned five-layer evaluation for **augmented covariant obfuscation** of Transformer inference.

**Authoritative spec:** [`docs/literature-review/reports/evaluation_protocol_fastProve.md`](../docs/literature-review/reports/evaluation_protocol_fastProve.md)

**Terminology (protocol §8.2):** report results as *obfuscated-state relative to plaintext*. This prototype is **augmented covariant obfuscation with bounded auxiliary noise**, not standard ciphertext and not fully encrypted (HE/MPC) inference.

---

## Quick start

From the repository root:

```bash
# Install package so `fastprove` imports resolve. Modern virtualenvs may use
# editable mode; the stock macOS/Xcode Python 3.9 pip should use this regular
# user install because it predates PEP 660.
python3 -m pip install --user --no-build-isolation ".[test,pretrained-lite]"

# Primary production cell (protocol P2) + gate check
PYTHONPATH=src:. python -m evals.report \
  --config evals/configs/P2.yaml \
  --keys 5 \
  --ci 0.95 \
  --check-gates

# Fast smoke (1 key, layers 1–4)
PYTHONPATH=src:. python -m evals.report \
  --config evals/configs/F1.yaml \
  --keys 1 \
  --layers 1,2,3,4 \
  --sample-count 4 \
  --check-gates
```

JSON dump:

```bash
PYTHONPATH=src:. python -m evals.report \
  --config evals/configs/P2.yaml \
  --keys 3 \
  --check-gates \
  --output results/raw/eval_P2.json
```

Full F0–P3 matrix (slow):

```bash
PYTHONPATH=src:. python -m evals.report --all-conditions --keys 3 --check-gates
```

---

## Command → protocol section map

| Command / flag | Protocol section | What it does |
|---|---|---|
| `--config evals/configs/{F0…P3}.yaml` | §3 condition matrix | Selects precision × mode cell |
| `--all-conditions` | §3 full matrix | Runs F0, F1, F2, P0, P1, P2, P3 |
| `--keys N` | §6.1 independent keys | N master keys → N full conversions |
| `--ci 0.95` | §6.2 statistics | 95% CI (Wilson / paired / stratified bootstrap) |
| `--check-gates` | §7 acceptance | PASS/FAIL hard + soft gates |
| `--layers 1,2,3,4,5` | §4 five-layer system | Subset of metric layers |
| `--R / --R-h / --R-ff / --gamma / --max-kappa` | §5 ablation | Noise dims, decay, condition number |
| `--precision fp32\|bf16\|fp16` | §5 ablation | Activation precision |
| `--seq-len` | §5 ablation | Sequence length |
| `--attention-impl` | §5 ablation | reference / sdpa / flash (prototype: reference) |
| `--kv-cache on\|off` | §5 ablation | Cache preference flag |
| `--output path.json` | §4–§7 artifacts | Machine-readable full report |

### Condition IDs (§3.1)

| ID | Precision | Mode | Role |
|---|---|---|---|
| **F0** | FP32 | plaintext | High-precision reference |
| **F1** | FP32 | structural-only | Graph + transforms; noise injection zeroed |
| **F2** | FP32 | full noise | Signal/noise decoupling |
| **P0** | BF16 | plaintext | Production baseline |
| **P1** | BF16 | structural-only | Transform numerical amplification |
| **P2** | BF16 | full nominal | **Primary result** |
| **P3** | BF16 | full stress | Stability boundary |

**Degradations reported (§3.2):**

- \(\Delta_{\mathrm{struct}}^{\mathrm{FP32}} = Q(F0)-Q(F1)\)
- \(\Delta_{\mathrm{noise}}^{\mathrm{FP32}} = Q(F1)-Q(F2)\)
- \(\Delta_{\mathrm{struct}}^{\mathrm{BF16}} = Q(P0)-Q(P1)\)
- \(\Delta_{\mathrm{noise}}^{\mathrm{BF16}} = Q(P1)-Q(P2)\)
- \(\Delta_{\mathrm{prod}} = Q(P0)-Q(P2)\)

### Structural mode (critical — §2)

`F1` / `P1` **must** keep the full obfuscation computation graph:

- augmented dimension \(D+R\), mixed weights \(\widetilde W\)
- Q/K orthogonal transform after RoPE
- mixed Value layout + KV-cache format
- FFN permutation / scaling
- LM-head vocab permutation when present

Only noise **injection** terms are zeroed: \(e_0\), \(C_\ell\), \(\xi_\ell\).  
This is **not** a near-plaintext model.

---

## Five layers (§4)

| Module | Layer | Metrics |
|---|---|---|
| `layer1_operator_equiv.py` | §4.1 operator equivalence | \(E_{\ell,\infty}\), \(E_{\ell,\mathrm{rel}}\) for Embedding, ChainLinear, RMSNorm, Q/K, Value, Attn out, SwiGLU, Down, LM Head |
| `layer2_attention.py` | §4.2 attention ranking | score max/rel error, KL, TV, top-1/top-k, **rank-flip rate**, \(m\le 2\varepsilon\) risk, AV rel error |
| `layer3_moe.py` | §4.3 MoE router | expert set/order match, gate L1/KL, \(\Delta_k\), \(2\tau<\Delta_k\) violations (skip if no MoE) |
| `layer4_output_logit.py` | §4.4 logits + trajectory | inverse-align \(\hat\ell=\tilde\ell\,\Pi_{\mathrm{voc}}^T\); teacher-forced **and** free-running |
| `layer5_downstream.py` | §4.5 downstream | MMLU/C-Eval/PIQA/IFEval/HumanEval/PPL table (synthetic proxy unless local data) |

### Units (§3.3)

- **Higher-is-better** (accuracy): report **percentage points** \(\Delta_{\mathrm{pp}}\) (e.g. 80% → 78% = **2 pp**).
- **Lower-is-better** (PPL/NLL): report **relative increase** \(\Delta_{\mathrm{rel}} = Q_{\mathrm{obf}}/Q_{\mathrm{plain}}-1\).
- Never confuse pp vs relative %.

---

## Gates (`--check-gates`, §7)

### Hard correctness (§7.1) — binary

| Gate | Threshold |
|---|---|
| FP32 ChainLinear max abs error | ≤ 1e-4 |
| Attention rank-flip rate | = 0 |
| Attention top-1 / top-k | 100% |
| MoE expert set | 100% (skipped if no MoE) |
| Causal mask match | 100% |
| Cache vs no-cache | identical |
| LM Head inverse-permuted argmax | 100% |
| Greedy release-validation sequences | 100% exact match |

### Soft utility (§7.2) — pre-specified non-inferiority

| Gate | Margin |
|---|---|
| Accuracy drop | ≤ **0.5 pp** |
| PPL relative increase | ≤ **1%** |
| 95% CI | within non-inferiority bound |

Margins are **pre-specified** and must not be retuned after seeing results (§6.2).

---

## Layout

```text
evals/
  README.md
  conftest.py
  conditions.py          # F0–P3 matrix + Δ_* helpers
  keys.py                # ≥3 independent master keys
  thresholds.py          # §7 gates
  stats.py               # Wilson, McNemar, paired/stratified bootstrap
  model_factory.py       # plain / structural / full builders
  metrics_common.py
  layer1_operator_equiv.py
  layer2_attention.py
  layer3_moe.py
  layer4_output_logit.py
  layer5_downstream.py
  report.py              # CLI entry: python -m evals.report
  configs/
    F0.yaml … P3.yaml
```

---

## Notes and limitations

1. Default models are the **random tiny** `PlainTinyCausalLM` / `ObfuscatedTinyCausalLM` from `src/fastprove`. Results are for **correctness**, not meaningful language-model accuracy (see `AGENTS.md`).
2. Layer 5 real tasks need local datasets; without them the suite uses **synthetic proxies** and labels them as such. Do not present proxy numbers as MMLU/HumanEval scores.
3. The current prototype LM head does not yet apply a vocab permutation \(\Pi_{\mathrm{voc}}\); Layer 4 uses the identity and notes this. Inverse-align is ready when the buffer is added.
4. MoE is implemented as `StableRouter` but not wired into the tiny LM; Layer 3 reports N/A for the LM and runs a standalone router check.
5. Threat model: honest-but-curious observer of persisted tensors / ordinary framework outputs. This suite does **not** claim protection against kernel hooks or register dumps (`docs/threat_model.md`).
6. **Remote ModelSplit “obfuscated” checkpoints are NOT this scheme.** Weights under the lab `ModelSplit/models` trees that come from the previous ModelSplit obfuscation pipeline must not be scored as fastProve `structural` / `full` / exact results. For Llama runs: start from **plaintext** `Llama-3.2-3B-Instruct`, convert with **this repo**, then evaluate F0–P3. See `docs/experiment_plan.md` §1.0.
7. Remote package management is **conda** (create an env such as `fastprove`); do not mix system Python with ad-hoc global installs.
