# Implementation Notes

## Phase 0 environment audit

Audit date: 2026-07-31 (Asia/Shanghai).

The repository started with only `AGENTS.md` and `START_PROMPT.md`. It was not a
Git repository, so there is no commit or dirty-worktree state to report. The
implementation is being created in place without silently initializing Git.

### Host

| Item | Audited value |
|---|---|
| OS | macOS 26.5.1 (build 25F80), Darwin 25.5.0, arm64 |
| Machine | MacBook Air, model identifier Mac17,3 |
| CPU | Apple M5, 10 logical cores |
| GPU | Integrated Apple M5 GPU, 10 cores, Metal 4 |
| Physical memory | 16 GiB unified memory |
| Disk | 926 GiB total, about 404 GiB available at audit time |
| CUDA/NVIDIA | unavailable; `nvidia-smi` absent and PyTorch CUDA build is `None` |
| MPS | built and available; a tensor smoke test succeeded |

Small deterministic mathematical tests use CPU and FP32. MPS availability does
not authorize silently moving a planned CUDA/full-model evaluation to MPS.
Any later full evaluation must repeat the resource preflight.

Sanitized audit commands:

```bash
sw_vers
uname -a
/usr/sbin/sysctl -n machdep.cpu.brand_string
/usr/sbin/sysctl -n hw.logicalcpu
/usr/sbin/sysctl -n hw.memsize
df -h .
/usr/bin/memory_pressure -Q
command -v nvidia-smi
system_profiler SPDisplaysDataType
```

Unfiltered `system_profiler SPHardwareDataType` output is not stored because it
contains a serial number and hardware UUID.

### Python software

Two global stacks exist and must not be mixed:

| Component | `python3` | `python` |
|---|---:|---:|
| Python | 3.9.6 | 3.13.12 |
| PyTorch | 2.8.0 | 2.11.0 |
| pytest | 8.4.2 | 9.0.3 |
| PyYAML | 6.0.3 | 6.0.2 |
| NumPy | 2.0.2 | 2.4.4 |
| Matplotlib | 3.9.4 | 3.10.9 |
| Transformers | 4.57.6 | 5.9.0 |
| Datasets | unavailable | 5.0.0 |

Both PyTorch builds report no CUDA and an available MPS backend. Prototype tests
use the `python3`/PyTorch 2.8 stack for a conservative Python 3.9 compatibility
floor. A later isolated environment should use the locally available Python 3.12.

Minimal dependencies are PyTorch, NumPy, PyYAML, pytest, and Matplotlib.
Transformers, Datasets, and Safetensors are optional and needed only for a later
pretrained evaluation. No package was installed or downloaded during the audit.

### Local model and data inventory

Complete standard caches include BERT, multilingual E5, CLIP, Whisper, and
ResNet weights. They are not causal Llama-like checkpoints suitable for the
required language-model evaluation. A bounded audit outside the standard cache
did find two complete, offline-loadable Llama-like candidates:

- DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 causal LM, RMSNorm/RoPE/SwiGLU/GQA),
  stored under the user's existing `model_artifacts` tree. Its single
  Safetensors file is 3,554,214,621 bytes and has SHA-256
  `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945`.
- Llama-3.2-3B-Instruct, also stored under the existing `model_artifacts` tree,
  with two complete Safetensors shards. It is a stricter Llama reference but is
  less suitable for a 16 GiB machine during conversion.

Both candidates load offline through Transformers. Their exact upstream
revision cannot be recovered from the local artifacts, so a later record must
use the absolute local path and weight hashes with `revision_unavailable`.

`HuggingFaceTB/SmolLM2-135M-Instruct` has only a reference entry and no snapshot
weights. The GPT-2 cache contains configuration/tokenizer files but no model
weights.

A cached Flickr30k test split is present, but it is a multimodal
caption/retrieval dataset rather than a ready causal-LM benchmark. No project-local
LM evaluation dataset exists.

Random tiny-model runs may establish implementation correctness only. The local
Qwen2 1.5B checkpoint is the preferred later adapter target, loaded one mode at
a time to control memory. A standard public causal-LM dataset is still absent.
Flickr30k captions could support a limited public-caption NLL check, but they
are not a standard LM benchmark and the current `python3` environment lacks the
Arrow loader. Downloading WikiText-2 or installing dataset dependencies remains
permission-gated.

### Current execution scope

The user requested implementation and an experiment plan before running
experiments. This pass does not run the formal noise sweep, a meaningful
pretrained accuracy evaluation, or a long performance benchmark. A bounded
one-sample adapter/runner smoke may run outside `results/raw` to validate the
control plane; it is never treated as accuracy evidence. The scripts and record
schemas are prepared for a later run without fabricating artifacts.

### Remote GPU hosts (2026-08-01 recon)

| Host | User | GPU | Free VRAM (recon) | Models | Workdir | Env |
|---|---|---|---|---|---|---|
| 10.144.144.6 | nss-d | RTX 5090 ~32GB | ~31GB free | `/home/nss-d/dcy/codes/ModelSplit/models` | `/home/nss-d/yhr/code/fastProve` | conda |
| 10.144.144.5 | nss-marker | RTX 4090 ~24GB | ~4GB free (busy) | `/home/nss-marker/dcy/code/ModelSplit/models` | `/home/nss-marker/yhr/code/fatsProve` | conda |

Both hosts list `Llama-3.2-3B` and `Llama-3.2-3B-Instruct` under the model trees.
Workdirs were empty placeholders at recon time.

**Critical (user-confirmed, 2026-08-02):** even when a remote directory is labeled
or known as “obfuscated,” that artifact comes from a **previous / different
ModelSplit obfuscation pipeline**, **not** the current fastProve augmented-state
covariant scheme. Such weights must **never** be reported as fastProve
structural/full/exact results. Primary evaluation must start from **plaintext
base weights**, convert with **this repository’s** conversion path and master
keys, then compare under the protocol F0–P3 matrix. See
`docs/experiment_plan.md` §1.0.

Also note in README/evals reports: terminology remains *obfuscated-state relative
to plaintext* (real-valued augmented covariant obfuscation), not ciphertext.
The phrases "LWE-keyed" and "LWE-inspired" must **not** be used: the map
`h -> c` is exactly invertible over the reals, so `h = c (M^-1)[:, :d]` holds for
any auxiliary state, and no LWE property (mod-q reduction, discrete error
distribution, noise flooding) is present. See `docs/threat_model.md` §5bis.

## Prototype implementation status

This pass completed the reference prototype and plan without running the
configured experiment sweep:

- row-vector layout conversion, `MixedState`, deterministic domain separation,
  well-conditioned bases, debug-only encode/decode, and `ChainLinear`;
- plaintext and exact/approximate Decoder Blocks with RMSNorm gamma absorption,
  RoPE-then-Q/K orthogonal transforms, mixed Value, SwiGLU compensation, noise
  side paths, causal/padding/all-mask handling, GQA, ties, and incremental cache;
- plaintext, exact, top-k-preserving, and free-bounded attention modes;
- a shared-weight random Tiny Causal LM for correctness only;
- teacher-forced, greedy, layer, Softmax, performance, and artifact metrics;
- plan-only-by-default correctness, 72-point sweep, and report CLIs;
- an exact-mode hard gate that blocks and records all approximate points as
  skipped if exact tolerances or token agreement fail;
- a deterministic calibration candidate selector that retains every baseline,
  schedule endpoint and Pareto-frontier point, and a `--spec-ids-file` full-run
  subset interface;
- pre-inference raw-output preflight, strict success/failure/skipped record
  semantics, required success-metric contracts, expected-run manifest
  reconciliation, cohort-safe unconnected report points, raw file/line
  traceability, and separate Top-k/free-bounded tradeoff plus Pareto summaries;
- pending result directories and report text that contain no fabricated values.

The hidden and Value checkpoint basis material is captured inside the reference
designated-operation closures and is absent from ordinary module
`state_dict`/`named_buffers`. This narrows the persisted-tensor observation
surface, but Python introspection, arbitrary hooks, modified kernels, and memory
dumps remain explicitly outside the protection claim.

Latest local verification in this implementation pass (2026-08-02):

```text
python3 -m pytest -q
209 passed

python3 -m compileall -q src evals scripts
completed successfully
```

The pytest run is a unit/correctness verification, not a noise sweep or
pretrained language-model experiment. Matplotlib emitted 14 dependency
deprecation warnings from its PyParsing compatibility layer; no test failed.
Static mypy checking is not a project gate and was not completed under the
Python 3.9 environment because mypy is unavailable there.

The sweep CLI now accepts explicit `--device` and `--activation-dtype` overrides.
On this 16-GiB MPS host, the no-weight preflight estimates roughly 17.1 GB for
FP32 and 10.7 GB for BF16; BF16 remains a separate condition after the FP32 exact
gate and never silently replaces it.

The reference runner records process peak RSS on CPU/MPS and the MPS allocator
counter when PyTorch exposes it; CUDA additionally records the CUDA allocator
peak. These are diagnostic measurements, not production-kernel performance.

## Post-review experiment blockers addressed (2026-08-01)

- raw evaluation key records now contain only a key identifier/label; master and
  conversion seeds are not serialized, and the previously generated local raw
  JSON files were scrubbed accordingly;
- BF16 condition cells cast both plaintext and obfuscated modules (including
  registered conversion buffers), so the recorded dtype is now the executed
  dtype rather than a label;
- each obfuscated KV cache carries a conversion/model/layer identity and the
  request identity, rejecting cross-conversion and cross-request reuse;
- approximate-logit noise uses a coordinate-stable vectorized hash based on
  absolute query/key positions, preserving cache/full-prefix alignment without
  the former Python four-loop bottleneck;
- production attention skips clean probability/output materialization unless
  `forward_debug` is explicitly selected; `fixed_debug` refresh now fails fast
  for non-debug construction;
- basis mutation is revalidated at debug encode/decode boundaries, and
  non-finite attention/router/config bounds are rejected;
- structural-only evaluation disables dynamic per-request refreshes and initial
  and refresh noise scales are applied independently.

A one-sample P2 all-layer smoke was run after these changes (outside
`results/raw`, so it is not a formal reported experiment): the model executed
in BF16 and the cache check passed. A local Qwen2 adapter is now implemented in
`src/fastprove/pretrained/qwen2.py`; it validates the Safetensors key set,
handles Q/K/V bias and GQA layout, restores structural buffers after
`to_empty`, and records checkpoint/tokenizer hashes. A one-sample BF16
plaintext smoke and a one-sample P2 layer-1 conversion smoke also completed
outside `results/raw`. These are integration checks, not formal accuracy
evidence. The BF16 adapter matched the local Transformers reference on a short
input at argmax level; framework-level BF16 logit rounding is retained as a
separate Gate B tolerance question.

## Deferred external evaluation work

`src/fastprove/evaluation/token_cache.py`, `scripts/prepare_token_cache.py`,
`scripts/preflight_experiment.py`, and the real-model options on
`scripts/run_accuracy_sweep.py` now provide the Gate B/C plumbing. A formal run
still requires an approved/already-authorized causal-LM corpus; no public corpus
was downloaded and no formal noise sweep was run. No model or dataset was
copied into this repository, and no remote GPU was used.

The read-only asset audit in
`src/fastprove/evaluation/pretrained.py` and
`scripts/audit_pretrained_assets.py` now records
`results/raw/pretrained_evaluation_status.json`. Its current
`skipped_external_dependency` status is explicit: the local Qwen2 checkpoint and
tokenizer are complete, while an approved standard causal-LM corpus is missing.
A local Flickr30k caption-only cache with 405 sequence-length-24 samples has been
prepared at `results/raw/flickr30k_caption_eval_inputs.pt`; the audit deliberately
does not count it as standard evidence unless `--accept-nonstandard-caption` is
explicitly supplied. The same manifest records the proposed 8-sample
calibration/64-sample full run and the conservative FP32 memory estimate.
For a deliberately limited caption-only start, the separately preserved
`results/raw/pretrained_evaluation_status_caption_candidate.json` records
`ready_for_experiment` while retaining the non-standard evaluation scope.

The real-model FP32 preflight intentionally reports a resource warning on this
16-GiB host: the conservative working-set estimate is about 17.1 GB for one
plain+converted Qwen2 copy and FP32 basis arithmetic. A short one-input exact
integration check completed, and the current CPU/MPS FP32 runner smoke passes the
calibrated deep exact envelope. Full 64-sample runs should still use batch size 1
and an isolated attempt; a larger-memory/CUDA host remains the recommended path
for reliable throughput. BF16 can be used for a separate adapter/resource
calibration, but approximate accuracy conclusions require the FP32 exact gate
first.
The runner does not silently change device or dtype when those resources are
unavailable. `preflight_experiment.py` now requires the subsequent device
explicitly and performs a no-weight matmul/Softmax capability smoke; an enabled
`PYTORCH_ENABLE_MPS_FALLBACK` is rejected.
The preflight also requires the same `--accept-nonstandard-caption` opt-in as
the sweep CLI before a caption-only cache can be marked ready; otherwise it
fails before writing a ready manifest.

The runtime checkpoint path is device-aware: offline transform generation and
inversion remain FP64, CPU checkpoints may use FP64 arithmetic, and MPS/CUDA
checkpoints use FP32 because MPS rejects FP64 tensors. The MPS path is covered
by `tests/test_mps_runtime.py`; this prevents a no-weight preflight from
mistaking an unsupported FP64 checkpoint operation for a usable accelerator.

The current no-weight admission manifests are the explicit-device refreshes
`results/raw/flickr30k_caption_preflight_cpu_fp32.json`,
`results/raw/flickr30k_caption_preflight_mps_fp32.json` and
`results/raw/flickr30k_caption_preflight_mps_bf16.json`. They record the actual
device/dtype smoke and `checkpoint_compute_dtype=float32`; none of these
manifests contains model-inference metrics. The older generic
`flickr30k_caption_preflight*.json` files are retained as historical diagnostics
only and are not admission manifests under the current explicit-device and
scope-opt-in contract.

The executable runner applies the same fallback guard before allocating model
weights, and records the fallback environment variable in every run manifest.

The current handoff status is also serialized in
`results/raw/experiment_readiness.json`. It is a readiness manifest only: it
does not count as a model evaluation or populate any accuracy table.

Packaging audit: the stock Xcode Python environment has pip 21.2.4 and
setuptools 58.0.4. Its editable PEP 660 path fails (and legacy `develop`
tries to write the system framework directory), so `setup.py` now carries
explicit legacy metadata and the supported installation command on this host
is `python3 -m pip install --user --no-build-isolation '.[test,pretrained-lite]'`. A regular user install was
built and imported successfully; modern virtualenvs may continue to use
editable mode.

The bounded attention and router samplers now clip their FP32 scale to the
representable predecessor of the requested Python-float bound. A one-point
Qwen CPU smoke initially exposed a `4.75e-11` one-ulp overrun at
`tau_max=0.001`; the fix is covered by strict regression tests rather than an
epsilon-only assertion.

The latest non-formal Qwen2 runner smoke used one cached token sequence and
completed outside `results/raw`. With the deep-model FP32 envelope (logit
max error `2.71e-4`, relative-L2 `9.24e-6`; QK max error `8.79e-3`,
relative-L2 `2.97e-7`; Softmax max error `3.65e-4`, relative-L2 `3.03e-6`),
the exact gate passed and greedy/teacher-forced token agreement was 1.0.
This is an integration check only: the cache was synthetic and it does not
establish language-model accuracy or authorize the noise sweep.

After the MPS checkpoint dtype fix, a real one-sample Qwen2 run on the local
caption cache completed on `mps:0`/BF16 with no NaN/Inf (`13.8 s` wall time),
but exact-vs-plaintext logit relative-L2 was `0.220772` and teacher-forced
argmax agreement was `0.833333`. This is an observed BF16/accelerator
numerical-condition diagnostic, not a passed exact gate; it is intentionally
not written to `results/raw` and does not authorize approximate-mode claims.

The complete sweep runner was then exercised with the same one-sample
MPS/BF16 settings in `/tmp/fastprove_runner_mps_bf16_smoke.jsonl` (outside the
report evidence set): all 72 declarations reconciled, `plaintext` and `exact`
completed successfully, and the 70 approximate points were explicitly
`skipped` by the failed exact gate. This confirms the real-model loading,
debug-metric capture, record validation, and gate propagation path; it is not a
noise-sweep result.

A follow-up one-sample FP32 runner smoke was executed outside `results/raw` on
both explicit CPU and MPS cohorts. Each produced two successful baseline/exact
records and 70 exact-gate skips before the calibrated gate profile was updated.
CPU measured logit max/relative-L2 `2.32697e-4`/`1.55612e-5` and Softmax
max/relative-L2 `1.29962e-3`/`9.28577e-6`; MPS measured
`7.30038e-4`/`4.43967e-5` and `1.42086e-3`/`8.39037e-6`. Both had teacher-forced
and greedy agreement `1.0`, no NaN/Inf, and the only old-gate failure was the
`1e-3` Softmax absolute threshold. The runner now records a calibrated deep
FP32 Softmax envelope of `2e-3` while retaining the `1e-4` relative-L2 and exact
token gates. These files remain in `/private/tmp` and are diagnostic, not
formal language-model evidence or a noise sweep.

## Experiment-start blocker review (2026-08-02)

The sweep control plane was rechecked without allocating model weights or
running inference. The former `evaluation_override.rationale` metadata field
in `configs/eval_sweep.yaml` would have been passed to the dataclass constructor
and caused every real sweep point to fail; it is now top-level metadata, and
unknown override fields fail early with an explicit error. All 72 configured
specifications now construct successfully with the declared 3 warmups, 10 timed
repetitions, and 1,000 paired sample-unit bootstrap replicates.

Raw failure/skipped records now retain model/checkpoint/tokenizer/cache hashes,
and candidate/report cohort keys include model configuration, tokenizer,
activation dtype, checkpoint arithmetic dtype, and generation-length dimensions.
Converted-model manifests record the hidden-basis
and per-Value-head condition numbers plus public fingerprints; matrices, inverses,
and seed material remain excluded.

Current no-weight Qwen2 preflight was repeated for the local caption cache with
BF16: artifact, tokenizer, cache identity, 405 samples and 24-token blocks all
passed; estimated working set is 10,662,643,863 bytes on the 16-GiB MPS host.
This is a resource/readiness result, not an accuracy result. The FP32 estimate
remains about 17.1 GB and is reported as a warning. CUDA is unavailable, so a
CUDA/FP32 exact gate still requires an authorized larger-memory environment;
BF16/MPS must be reported as a separate condition and cannot replace it.

Final local verification after this review:

```text
python3 -m compileall -q src evals scripts
python3 -m pytest -q
209 passed, 14 dependency deprecation warnings
```

## Continuation readiness audit (2026-08-02)

The current interpreter remains Python 3.9.6 with PyTorch 2.8.0;
CUDA is unavailable and MPS is available. A read-only scan of the local data
roots found no additional WikiText/PTB/other standard causal-LM token corpus.
The only aligned local candidate remains the 405-sample Flickr30k caption cache,
whose non-standard evidence scope is enforced by the explicit CLI flag.

The repository currently contains no `results/raw/**/*.jsonl` attempt. This is
intentional: `build_report.py --execute` refuses to overwrite the pending report
when no raw run records exist, so an empty or failed attempt cannot be mistaken
for an accuracy result. The exact plan and first calibration command are
serialized in `results/raw/experiment_readiness.json`.

The final control-plane pass also made two pre-run safeguards explicit. A
candidate manifest is normalized to execute `plaintext` and then `exact` before
any approximate point, even if the calibration JSONL was produced out of order;
otherwise an approximate point could be permanently skipped before the exact
gate had a chance to pass. The resource preflight now rejects zero generation
tokens, matching the runtime evaluation contract and avoiding a false-ready
manifest that would later fail during TPOT calculation.

## Staged large-checkpoint execution (2026-08-02)

The pretrained runner now separates the plaintext and converted-model phases
to reduce the resident-memory window on the 16-GiB host:

1. Load one plaintext model, collect teacher-forced logits, greedy output, and
   plaintext performance metrics.
2. Convert the plaintext weights once. The unavoidable conversion peak is
   recorded separately from inference; conversion still requires enough memory
   for the source and deployed weights at the same time.
3. Release the plaintext module and run obfuscated debug/production forwards,
   greedy generation, KV-cache measurement, and obfuscated performance metrics.
4. Merge both phase records into the historical performance schema and retain
   `peak_memory.measurement_phases` so the report does not imply that two full
   models were resident throughout the benchmark.

Research-only layer diagnostics are detached to CPU immediately after each
debug batch. This keeps the metrics available while avoiding a device-backed
graph/list growing across all cached samples. Production timing continues to
use the non-debug forward path; attention probabilities and decoded states
are not part of its public return value.

The staged path removes the largest avoidable memory blocker, but it does not
make FP32 Qwen2 conversion safe on every 16-GiB machine. The existing preflight
warning (about 17.1 GB conservative FP32 working set) remains authoritative;
use batch size 1 and an isolated calibration attempt, or move to a larger-memory
CUDA host. No formal pretrained run has been started in this implementation
pass.

## 阶段 B：开销实测与热点（任务 B7，2026-08-11）

`scripts/measure_overhead.py`（eager 参考实现，非融合 kernel）实测
（d=512/1024/2048，4 层，CPU FP32，seq=128，8 个 decode token，5 次取 min）：

| hidden_size | prefill overhead | decode overhead |
|---:|---:|---:|
| 512 | +54.7% | +58.5% |
| 1024 | +30.7% | +50.9% |
| 2048 | +46.1% | +46.6% |

全部记录带 `"implementation": "eager reference, not a fused kernel"`，
原始文件 `results/raw/overhead_B7_d*.json`。相比阶段 A 开始时（+150.3%）
明显下降；这仍是参考实现，与手册 §69 的 ≤5% 融合 kernel 目标无关。

eager overhead 仍 >30%，按要求定位了热点（torch.profiler，d=1024，
CPU，prefill，self CPU time，共 29.2ms）：

| 排名 | 算子 | self CPU 占比 | 说明 |
|---|---|---|---|
| 1 | aten::mm | 35.2% | 部署权重 GEMM（q/k/v/attn_out/gate/up/ffn_out/head/embedding） |
| 2 | aten::bmm | 15.5% | attention 分数与概率×value |
| 3 | aten::copy_ | 7.0% | 布局转换与 cat（cache/value/去重） |
| 4 | aten::select | 4.5% | view/拆分算子 |
| 5 | aten::index | 4.1% | RoPE 与基的 gather |

其余（add/div/einsum/mul/silu 等）各 ≤3.4%。einsum 的 total 占比 32.3%
（内部派发到 mm/bmm）。结论：额外开销来自部署路径新增的少量 GEMM
（`(c@N)@Wnz`、`rho` Gram 归约）与 attention 的 value einsum；算术预算
表（文档 §2.3）显示融合 kernel 下增量约 +1.8%，参考实现达不到该数。
