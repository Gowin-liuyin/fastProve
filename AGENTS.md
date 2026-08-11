# AGENTS.md

## Project Mission

This repository builds a reproducible research prototype for covariant obfuscation of Transformer inference.

The immediate objective is:

1. implement a plaintext Transformer reference;
2. implement an exact covariant-obfuscation path whose output should match the plaintext reference up to floating-point error;
3. implement an approximate Softmax-obfuscation path with bounded logit noise;
4. measure how much accuracy is lost relative to plaintext inference as the noise strength increases;
5. produce reproducible code, raw metrics, plots, and a report.

The prototype is intended to test mathematical and engineering feasibility. It is not a production cryptographic system.

## Research Background

The starting idea is an augmented-state covariant transform. A hidden state is extended with a low-dimensional noise state and then mixed by an invertible basis:

\[
h_\ell\in\mathbb R^{d_\ell},\qquad
e_\ell\in\mathbb R^{r_\ell},
\]

\[
z_\ell=[h_\ell,e_\ell],
\qquad
c_\ell=z_\ell M_\ell.
\]

Here:

- \(h_\ell\) is the model's semantic signal;
- \(e_\ell\) is an auxiliary noise state;
- \(r_\ell\ll d_\ell\);
- \(M_\ell\in\mathbb R^{(d_\ell+r_\ell)\times(d_\ell+r_\ell)}\) is invertible;
- \(c_\ell\) is the state exposed to the obfuscated server path.

The older one-dimensional augmented-matrix construction is the special case \(r_\ell=1\). Its linear-chain algebra is useful, but its original nonlinear bridge is incomplete: once purely multiplicative noise is set to zero before a nonlinear layer, later nonzero multipliers cannot regenerate it. This project therefore uses affine noise refresh and explicit nonlinear checkpoints.

## Claims and Non-Claims

The project may claim only what is demonstrated by mathematics or generated experiment artifacts.

Allowed terminology:

- augmented covariant obfuscation;
- chained auxiliary noise;
- bounded logit perturbation;
- approximate privacy-preserving inference prototype;
- exact mode and approximate mode.

Do not claim:

- that activations are standard LWE ciphertexts;
- fully homomorphic or end-to-end encrypted inference;
- LWE security merely because ML-KEM, an LWE-derived KDF, or an LWE-inspired seed is used;
- protection against a server that can arbitrarily modify kernels and dump registers;
- a performance, accuracy, or privacy number that was not produced by a recorded run.

If LWE/ML-KEM is added later, its role is key establishment or parameter derivation only unless a separate reduction proves more.

## Threat Model

The initial prototype evaluates representation obfuscation against an honest-but-curious observer of persisted tensors and ordinary framework outputs.

The prototype may assume that a designated fused operation does not return its internal clean temporary values. This is an implementation assumption, not a cryptographic guarantee.

If an attacker can:

- modify CUDA/Triton/PyTorch kernels;
- attach arbitrary hooks;
- dump registers or temporary buffers;
- read all client secrets;

then this prototype alone does not protect Softmax logits, probabilities, or extracted signal components. Stronger protection would require a trusted execution environment, MPC, HE, or another trusted boundary.

Every report must state this limitation.

## Canonical Mathematical Specification

Use row-vector mathematics throughout:

\[
Y=XW+b.
\]

PyTorch `torch.nn.functional.linear` stores weights as `[out_features, in_features]`, so every conversion function must document whether a weight is in mathematical or PyTorch layout.

### Mixed State

\[
c_{\rm in}=[h,e]M_{\rm in}.
\]

Default prototype dimensions:

```text
hidden noise dimension r: 8 or 16
attention value noise per head r_h: 1, 2, or 4
```

Use well-conditioned transforms. “Dense” or “all nonzero” does not imply secure or numerically stable.

Recommended structure:

\[
M=\Pi D B,
\]

where:

- \(\Pi\) is a permutation;
- \(D\) is a bounded nonzero diagonal scaling;
- \(B\) is an orthogonal or small block-orthogonal transform;
- the condition number is checked and recorded.

The default condition-number limit should be configurable and conservative, for example `max_condition_number: 10`.

Never compute a matrix inverse in `forward`. Compute transforms offline in FP64 or FP32, validate them, and store converted weights.

### Affine Linear Layer and Noise Refresh

For:

\[
y=hW+b,
\]

define:

\[
e'=hC+eG+\xi.
\]

Let:

\[
K=
\begin{bmatrix}
W&C\\
0&G
\end{bmatrix}.
\]

Deploy:

\[
\widetilde W=M_{\rm in}^{-1}KM_{\rm out},
\]

\[
\widetilde b=[b,\xi]M_{\rm out}.
\]

Then:

\[
c_{\rm out}
=
c_{\rm in}\widetilde W+\widetilde b
=
[hW+b,\;hC+eG+\xi]M_{\rm out}.
\]

This identity must have a direct unit test.

Use a stable noise propagator such as:

\[
G=\gamma P_e,\qquad 0<\gamma<1,
\]

where \(P_e\) is a permutation or signed permutation.

Two refresh modes must be distinguished:

- `fixed_debug`: fixed \(\xi\) stored in the converted bias; used only for correctness tests;
- `per_request`: \(\xi\) generated from a reproducible per-request, per-layer seed inside the designated fused operation.

Do not present deterministic fixed refresh as strong privacy.

## Nonlinear Checkpoint Rule

General dense augmented transforms do not commute with nonlinear functions.

At a nonlinear checkpoint:

1. derive the permitted transformed signal inside a reference or fused operation;
2. keep the old noise on a side path rather than destroying it;
3. apply the nonlinear function only to its mathematically valid signal representation;
4. construct a refreshed noise state;
5. return a newly mixed state;
6. do not expose clean intermediate tensors through the public production-mode API.

Reference/debug code may expose decoded states for testing, but every such function must be clearly named and disabled in production-mode configuration.

## RMSNorm

For a gamma-free RMS normalization and an orthogonal matrix \(R\):

\[
u=hR,\qquad RR^T=I,
\]

\[
\operatorname{rms}(u)=\operatorname{rms}(h).
\]

The learned elementwise scale \(\Gamma=\operatorname{diag}(\gamma)\) does not generally commute with arbitrary \(R\). It must be absorbed into the following projection.

For Query:

\[
Q=
\frac{h}{\operatorname{rms}(h)}
\Gamma W_Q,
\]

and \(u=hR\), use:

\[
Q'=
\frac{u}{\operatorname{rms}(u)}
R^T\Gamma W_QC_Q.
\]

Equivalent formulas are required for Key, Value, Gate, and Up.

RMS statistics, epsilon addition, and reductions must use FP32.

## RoPE and Q/K

The preferred prototype ordering is:

1. obtain the valid Query and Key projections inside the checkpoint;
2. apply RoPE;
3. apply a common orthogonal transform \(C\) after RoPE.

\[
Q'=\operatorname{RoPE}(Q)C,
\]

\[
K'=\operatorname{RoPE}(K)C.
\]

Therefore:

\[
Q'K'^T=Q_{\rm rope}K_{\rm rope}^T.
\]

Applying \(C\) after RoPE avoids an unnecessary and often false commutation assumption.

For grouped-query attention, every Query head sharing a KV head must use a transform compatible with that KV head. Add explicit tests for GQA mappings.

## Value Path

Value may carry an augmented noise state because attention probabilities multiply Value linearly from the left:

\[
c_V=[V,e_V]M_V.
\]

For any attention matrix \(A\):

\[
Ac_V=[AV,Ae_V]M_V.
\]

This is the principal mechanism for carrying noise through the attention block.

## Softmax Modes

Softmax is row-wise and does not commute with a general dense change of basis.

The exact covariance family is limited. For a column permutation \(P\) and row-dependent common offset \(a\mathbf 1^T\):

\[
\operatorname{softmax}(SP+a\mathbf 1^T)
=
\operatorname{softmax}(S)P.
\]

Do not add a finite zero-valued dummy position to Softmax. It changes the normalization denominator. Any dummy or invalid position must remain masked as \(-\infty\).

Implement three attention modes behind the same interface.

### `plaintext`

\[
A=\operatorname{softmax}(S).
\]

This is the reference baseline.

### `exact`

Use the common orthogonal Q/K transform, no logit perturbation, and the mixed Value path:

\[
S=\frac{Q'K'^T}{\sqrt{d_h}},
\]

\[
O_{\rm exact}
=
\operatorname{softmax}(S)c_V.
\]

This mode must match plaintext signal output up to numerical tolerance.

An optional key-position permutation may be added later:

\[
\widetilde S=SP+a\mathbf 1^T,\qquad
\widetilde V=P^Tc_V.
\]

Then:

\[
\operatorname{softmax}(\widetilde S)\widetilde V
=
\operatorname{softmax}(S)c_V.
\]

The mask and KV layout must be permuted consistently. This optional extension hides position labels only; it does not hide the score multiset or ranking structure.

### `approximate`

Add bounded noise only to valid logits:

\[
\widehat S=S+\eta.
\]

Masked positions must remain exactly \(-\infty\).

Implement two schedules:

1. `topk_preserving`: limits noise using the clean Top-k boundary margin;
2. `free_bounded`: uses a fixed error budget and allows ranking changes.

For `topk_preserving`, sort valid logits and define:

\[
\Delta_k=S_{(k)}-S_{(k+1)}.
\]

Set:

\[
\tau
=
\min\left(
\tau_{\max},
\frac{\alpha\Delta_k}{2},
\tau_{\rm error}
\right),
\qquad 0<\alpha<1,
\]

and generate:

\[
\|\eta\|_\infty\le\tau.
\]

Then:

\[
2\tau<\Delta_k
\]

guarantees that the Top-k set does not change, apart from explicitly handled ties.

For `free_bounded`:

\[
\tau=\min(\tau_{\max},\tau_{\rm error}),
\]

and ranking changes are allowed. This mode is important because preserving the complete ranking and hiding the ranking are conflicting objectives.

Compute:

\[
\widehat A=\operatorname{softmax}(\widehat S),
\]

\[
\widehat O=\widehat A c_V.
\]

Track:

\[
\widehat O-O
=
\left(
\operatorname{softmax}(S+\eta)
-\operatorname{softmax}(S)
\right)V.
\]

Noise generation must be deterministic under a recorded seed for reproducibility. Center and rescale sampled noise so that its actual infinity norm obeys the requested bound.

Margin computation, masking, noise clipping, and Softmax reduction must use FP32.

Do not return attention probabilities from production-mode APIs.

## SwiGLU

For:

\[
z=\operatorname{SiLU}(g)\odot u,
\]

use a shared neuron permutation \(P_f\) and a nonzero bounded diagonal scaling \(D_f\):

\[
g'=gP_f,
\qquad
u'=uD_fP_f.
\]

Then:

\[
\operatorname{SiLU}(g')\odot u'
=
zD_fP_f.
\]

Convert the Down weight:

\[
W_d'=P_f^TD_f^{-1}W_d,
\]

so:

\[
(zD_fP_f)W_d'=zW_d.
\]

Carry the previous noise through a side path and refresh after the elementwise product:

\[
e_z=z'C_z+e_{\rm side}G_z+\xi_z.
\]

Do not claim that SiLU commutes with arbitrary scaling or dense mixing.

## MoE Router and Top-k

For an expert permutation \(P_E\):

\[
r'=rP_E+\eta_r.
\]

If:

\[
2\|\eta_r\|_\infty
<
r_{(k)}-r_{(k+1)},
\]

the selected expert set is unchanged after accounting for \(P_E\).

Physical experts must be reordered consistently.

Implement:

- exact expert permutation with no router noise;
- margin-bounded router noise;
- optional approximate noisy gate weights.

Tie-breaking must match the plaintext implementation and have explicit tests.

## Prototype Scope

Start small. Do not begin with a 7B model or custom CUDA.

The required progression is:

1. deterministic small-tensor mathematical tests;
2. one plaintext Transformer block;
3. one exact obfuscated Transformer block;
4. one approximate obfuscated block;
5. a tiny multi-layer causal LM constructed locally;
6. a locally available small pretrained Llama-like checkpoint, if available;
7. only after correctness, optional Triton or fused-kernel work.

If no meaningful pretrained checkpoint is locally available, random tiny models may be used for correctness only. Do not describe random-model results as meaningful language-model accuracy.

Ask before downloading a large model or dataset.

## Expected Repository Layout

Create and maintain:

```text
fastProve/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── tiny_exact.yaml
│   ├── tiny_approx.yaml
│   └── eval_sweep.yaml
├── src/
│   └── fastprove/
│       ├── config.py
│       ├── seed.py
│       ├── state.py
│       ├── transforms.py
│       ├── conversion.py
│       ├── layers/
│       │   ├── linear.py
│       │   ├── rmsnorm.py
│       │   ├── attention.py
│       │   ├── swiglu.py
│       │   └── router.py
│       ├── models/
│       │   ├── plain.py
│       │   └── obfuscated.py
│       └── evaluation/
│           ├── correctness.py
│           ├── accuracy.py
│           └── metrics.py
├── scripts/
│   ├── run_correctness.py
│   ├── run_accuracy_sweep.py
│   └── build_report.py
├── tests/
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── REPORT.md
└── docs/
    ├── mathematics.md
    ├── threat_model.md
    └── implementation_notes.md
```

Do not commit downloaded weights, datasets, secrets, caches, or large generated artifacts.

## Implementation Phases

### Phase 0: Environment Audit

Before changing code:

- inspect Python and PyTorch versions;
- inspect available CUDA, MPS, CPU, RAM, and disk;
- record the result in `docs/implementation_notes.md`;
- determine whether a local pretrained model and evaluation dataset exist;
- create a minimal dependency plan.

CPU is sufficient for unit tests and tiny correctness checks. Before a full model evaluation, explicitly verify compute resources.

### Phase 1: Mathematical Kernel

Implement:

- seeded transform generation;
- condition-number validation;
- `MixedState`;
- encode/decode helpers restricted to debug/tests;
- `ChainLinear`;
- affine noise refresh;
- direct numerical tests of every identity.

Do not proceed if the linear identity fails.

### Phase 2: One Exact Block

Implement a small plaintext block and an exact obfuscated block with identical base weights:

```text
RMSNorm
→ Q/K/V
→ RoPE
→ Attention
→ Output projection
→ Residual
→ RMSNorm
→ SwiGLU
→ Down
→ Residual
```

Compare every permitted decoded signal checkpoint.

### Phase 3: Approximate Softmax

Implement:

- `plaintext`;
- `exact`;
- `approximate/topk_preserving`;
- `approximate/free_bounded`.

Noise must be applied after masking logic is established and only to valid logits.

Required initial sweep:

```text
tau_max:
  0
  0.001
  0.003
  0.01
  0.03
  0.05
  0.10

alpha:
  0.50
  0.80
  0.95

preserve_top_k:
  4
  8
  16
```

The sweep must be configurable; do not hard-code it in layer implementations.

### Phase 4: End-to-End Evaluation

Run plaintext and obfuscated inference on exactly the same:

- model checkpoint;
- tokenized examples;
- sequence lengths;
- seeds;
- dtype;
- device;
- generation settings.

Store one machine-readable record per run.

### Phase 5: Report

Generate:

- raw JSON or JSONL metrics;
- summary CSV;
- accuracy/noise plots;
- `results/REPORT.md`;
- exact commands and environment metadata.

The report must distinguish mathematical exactness, floating-point deviation, task accuracy degradation, performance overhead, and security limitations.

## Evaluation Metrics

### Layer-Level Correctness

Record:

- maximum absolute error;
- mean absolute error;
- relative L2 error;
- cosine similarity;
- NaN/Inf counts;
- QK score error;
- exact-mode Softmax probability error;
- Value/attention-output signal error;
- Top-k overlap;
- tie and mask behavior.

### End-to-End Accuracy

For a causal language model, record at minimum:

- negative log-likelihood;
- perplexity;
- next-token top-1 accuracy;
- next-token top-5 accuracy;
- plaintext/obfuscated token agreement;
- greedy-generation token exact match;
- sequence-level exact match where meaningful.

If a downstream labeled task is used, record:

- task metric appropriate to the dataset;
- absolute degradation in percentage points;
- relative degradation;
- sample count and confidence interval when feasible.

Primary comparison:

\[
\Delta_{\rm abs}
=
\operatorname{metric}_{\rm obfuscated}
-
\operatorname{metric}_{\rm plaintext}.
\]

For metrics where larger is better, also report:

\[
\operatorname{drop}_{\rm abs}
=
\operatorname{metric}_{\rm plaintext}
-
\operatorname{metric}_{\rm obfuscated}.
\]

For perplexity, report both absolute and relative increase.

### Softmax-Specific Metrics

Record:

- KL divergence between plaintext and perturbed attention distributions;
- Jensen-Shannon divergence;
- Top-k set overlap;
- rank correlation on valid positions;
- fraction of queries whose Top-k set changed;
- clean boundary margin distribution;
- actual noise infinity norm;
- fraction of queries receiving zero or near-zero noise;
- attention-output relative L2 error.

### Performance

Record separately:

- conversion time;
- prefill latency;
- decode latency or TPOT;
- tokens per second;
- peak memory;
- KV-cache memory;
- reference implementation versus optimized implementation.

Do not optimize performance before exact correctness passes.

## Initial Acceptance Criteria

Mathematical/unit-test gates:

- all transform round trips pass;
- all matrix shapes are asserted;
- no matrix inverse occurs in `forward`;
- FP32 ChainLinear maximum absolute signal error is at most `1e-5` on small tests;
- exact attention QK scores match within a documented FP32 tolerance;
- exact-mode single-block output matches the plaintext block within a documented tolerance;
- mask, causal, cache, and tie tests pass;
- fixed seeds reproduce identical metrics.

End-to-end exact-mode gate:

- no NaN or Inf;
- exact mode shows only floating-point-level deviation;
- any greedy mismatch is investigated before approximate-noise conclusions are accepted.

Approximate-mode completion gate:

- every configured noise point has a recorded result or an explicit failure record;
- the report identifies the best accuracy/privacy operating points;
- no result is silently omitted;
- conclusions cite generated artifact paths.

## Compute and Experiment Guard

Before a full evaluation:

1. check CUDA using `nvidia-smi` when available;
2. check PyTorch CUDA and MPS support;
3. record device name, memory, dtype, and software versions;
4. compare model requirements with available resources;
5. do not silently move a planned GPU evaluation to CPU;
6. do not fabricate results if compute or data is unavailable.

Tiny CPU unit tests are allowed.

## Data and Model Integrity

- Record exact model identifier, local path, revision/commit when available, and file hashes where practical.
- Record dataset name, split, sample count, preprocessing, and provenance.
- Never evaluate plaintext and obfuscated modes on different samples.
- Cache the selected evaluation example IDs or tokenized inputs so all modes use identical data.
- Never train or evaluate on secret or private material without explicit authorization.
- Do not download large artifacts without asking first.

## Coding Rules

- Use Python and PyTorch for the reference implementation.
- Use type hints and docstrings for public APIs.
- Keep mathematical-layout and PyTorch-layout conversion explicit.
- Use FP64 or FP32 for offline inversion and conditioning checks.
- Use FP32 for RMS statistics, logits, margins, clipping, and Softmax reductions.
- Make model activation dtype configurable.
- Never silently fall back from a requested secure/fused path to an eager path.
- Separate debug decode APIs from production APIs.
- Production-mode outputs must not expose decoded hidden states, noise states, attention probabilities, or router logits.
- Keep tests small, deterministic, and fast.
- Prefer configuration files over hard-coded experiment constants.
- Preserve raw results; derived summaries must be reproducible from raw files.
- Do not change mathematical definitions merely to make a failing test pass. Diagnose the discrepancy.

## Working Style for Agents

Agents should execute the current phase autonomously while staying within repository scope.

Before implementation:

- inspect existing files and uncommitted changes;
- state assumptions;
- identify the current phase;
- make a short plan.

During implementation:

- implement the smallest complete vertical slice;
- run targeted tests after each mathematical component;
- report unexpected discrepancies immediately;
- do not claim completion based only on code compilation.

At handoff:

- summarize files changed;
- list exact test and experiment commands;
- report passes, failures, and unexecuted checks separately;
- link raw result and report files;
- state the next logically required phase.

## Definition of Done for the Current Project

The initial project is complete only when:

1. plaintext, exact, top-k-preserving approximate, and free-bounded approximate modes exist;
2. exact-mode mathematical tests pass;
3. at least one meaningful pretrained small-model evaluation has run, or the absence of model/data/compute is explicitly documented;
4. the configured noise sweep has produced raw results;
5. accuracy degradation relative to plaintext is summarized in tables and plots;
6. `results/REPORT.md` explains the best operating point and limitations;
7. all reported numbers are reproducible from commands and raw artifacts in this repository.
