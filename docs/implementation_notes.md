# Implementation Notes

状态：CURRENT
日期：2026-09-19
上游依据：AGENTS.md、docs/CLEANUP_PLAN.md
适用范围：环境审计与分阶段实现记录

## Phase 0 environment audit

Audit date: 2026-07-31 (Asia/Shanghai).

The repository started with only `AGENTS.md` and `docs/archive/2026-09-19_START_PROMPT.md`. It was not a
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
`docs/archive/2026-09-19_experiment_plan.md` §1.0.

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
by `tests/models/test_mps_runtime.py`; this prevents a no-weight preflight from
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
tries to write the system framework directory), so a legacy setup.py shim
carried explicit metadata until P1 removed it (metadata now lives in
`pyproject.toml`), and the supported installation command on this host
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

## 增广带噪内核：阶段 A 执行记录（2026-09-14）

依据 `docs/augmented_noisy_obfuscation_plan.md` 第 10 节阶段 A 与阶段 B 第 1、2 组。本次为小张量内核实现与数值验证，不含任何安全结论。

### 新增代码

- `src/fastprove/transforms.py`：新增 `generate_two_sided_transform`（`M = Q_L D Q_R`，双 FP64 QR，对角 `exp(u)`，`u ~ U[-log κ/2, log κ/2]`），保留原 `Pi D Q` 生成器作对照。
- `src/fastprove/augmented_noise.py`（新文件）：`NoiseRefreshSpec`（r、γ、β）、`generate_propagator`（`G = γP_e` 符号置换）、`record_generator`（按 `(key_epoch, request_nonce, sample_id, token_position, layer_id, operation, head_id)` 域分离）、`sample_initial_noise` / `sample_refresh_noise`（均匀有界分布，方差匹配 σ_e² 与 σ_e²(1−γ²)）、`apply_noise_refresh`（`γeP_e + ξ`，C=0）、`noise_coverage_metrics`（列范数 min/max、近零比例、有效秩）、`observed_noise_ratios`（实际 ‖e‖/‖h‖ 分位数）。
- `src/fastprove/coa_views.py`（新文件）：`transcript_coa` / `cloud_coa` 两视图输入白名单契约与 manifest 落盘（方案 4.3 节）。
- `configs/augmented_noise_kernel.yaml`：内核实验配置草案（矩阵族、r、κ、β、γ、攻击视图清单）。
- `scripts/run_augmented_noise_kernel.py`：阶段 A 闭环脚本，写 `results/raw/augmented_noise_kernel/kernel_records.jsonl`。
- `tests/math/test_augmented_noise_kernel.py`（18 项）与 `tests/security/test_coa_view_contract.py`（7 项）。

### 数值验证结果（CPU、FP64 转换、FP32 部署门禁）

- 条件数：d=64、r=16、κ目标 3 时实测 κ₂=2.90；κ=1 时为 1.0000（正交）；κ=10 时 9.31，均未超限。
- 逆残差（存储精度）：FP64 <1e-13；FP32 存储 <1e-6。
- 恒等式：`M_n P ≈ 0`（<1e-13）；完整 `[y, e']` 输出信号最大绝对误差 ~1e-14（FP64）/ ~1e-7（FP32 部署，通过 ≤1e-5 门禁）。
- 噪声链平稳性：20k 样本下 Var(e₀)=σ_e²、Var(e₁)=σ_e²（γ=0.5，均匀刷新，采样误差 <5%）。
- 覆盖度：双侧基 r=16 有效秩 15.51/16（无近零坐标）；基线 `Pi D Q` 为 13.01/16；κ=10 时 14.12/16。
- 实际噪声比：β=1 时 ‖e‖/‖h‖ 中位数 0.127、最大 0.165，远低于 FP32 Gram 上限 30。
- 复现性/隔离：固定种子逐位复现；不同 token 位置、不同样本、不同层的噪声流互不相同；同一绝对位置 prefill/decode 一致。β=0 时流相等（无噪对照，符合预期）。
- 测试：`pytest tests/ -q --ignore=tests/test_mps_runtime.py` → 338 passed。

### 边界

- 本轮未接入 Transformer block（方案阶段 D 未开始）；未运行任何 COA 攻击实验；`attacker_views` 仅落配置与契约测试。
- 双侧基对攻击难度的提升仍是待检验假设；当前数据仅证明数值正确性与坐标覆盖差异。
- per_request ξ 动态注入已在闭环验证（`ξ @ M_out,n` 外加项），尚未接入 `ChainLinear` 的 per_request forward 路径。

## v2 数学方案与实施指南环境记录（2026-09-16）

本次为文档与独立 CPU 小矩阵检查，不是上述阶段 A runner 的修复或重新验收。阶段 A 历史记录的 dtype、β 标定、域分离、YAML 接入问题仍应参照 `docs/archive/2026-09-14_augmented_noise_stage_a_review.md` 和 `docs/archive/2026-09-15_AUGMENTED_NOISE_HANDOFF_20260915.md`，不能从本次检查推断已修复。

- Python：`/Users/yin/miniconda3/bin/python`，3.13.12；NumPy 2.4.4；PyTorch 2.11.0。
- OS：macOS 26.6.2 arm64；RAM 16 GiB；检查时磁盘剩余 178,448,125,952 字节。
- CUDA 不可用，未找到 nvidia-smi；MPS 可用但本次没有使用。执行设备为 CPU。
- 已确认本地 Qwen 1.5B 与 Llama 3.2 3B 的 base_model 目录及 `results/raw/real_scenario_prompts_1500.jsonl` 存在；未载入、校验权重或认定该语料适合正式评测。
- 依赖计划：使用已有 NumPy 运行数学 oracle；后续实现复用现有 PyTorch/YAML/pytest，不新增依赖、不下载模型数据。
- 新方案入口（旧框架，已归档）：`docs/archive/2026-09-16_DERIVATION_PACKAGE.md`；证明：`docs/archive/2026-09-16_PROOF_PACKAGE.md`；工程合同：`docs/archive/2026-09-16_augmented_covariant_implementation_v2.md`。
- 新数学检查脚本：`docs/verification/verify_augmented_covariant_v2.py`；扩维后 concat/full 两个方案的更新记录：`results/raw/expanded_augmentation_v2_math_check_20260916.json`。该记录不代表 fastprove runtime 测试或模型/安全评测；此前 compact-primary 和 concat-only 记录保留为历史文件，不作为当前方案验收。

## 代码结构梳理 P0：环境审计与数值基线（2026-09-19）

P0 目标是在任何结构改动前冻结可验证基线。

- 环境：Python 3.13.12（`/Users/yin/miniconda3/bin/python`）；PyTorch 2.11.0；NumPy 2.4.4；macOS arm64；CPU + MPS，无 CUDA。
- 包可导入性：`python -m pip install -e .` 成功；`python -c "import fastprove"` 与裸 `python -m pytest`（收集 382）均通过。此前裸 pytest 收集失败与脚本内 13 处 `sys.path` 引导的问题由可编辑安装解决；P5 将移除这些引导。
- 测试基线：`python -m pytest tests -q` → 382 passed in 11.99s；本机 MPS 可用，MPS 测试实跑，无 skip。`--durations=25` 最长用例为 `tests/test_cli_plan.py::test_execution_flags_are_not_interchangeable`（1.44s）。
- 数值基线：`python scripts/verify_refactor_equivalence.py` 生成 `results/raw/refactor_baseline_5e19c43ea2b2.json`（12 个 section、2395 个摘要叶子，覆盖种子/变换/StructuredBasis/噪声原语/预算/ChainLinear/部署权重/四种单块/tiny LM/router+MoE/converter 产物/评测流水线）；`--check` 重算全等。
- 门：每个重构提交必须同时通过 `python -m pytest tests -q` 与 `python scripts/verify_refactor_equivalence.py --check`；等价性失败只允许回滚，不允许改基线。

## P4 去重执行记录（2026-09-19）

已完成并按序提交（每步双门通过）：

- P4a：`_sha256_file` 7 份 → `fastprove/hashing.py`（分块读取，digest 不变）。
- P4b：checkpoint I/O → `pretrained/_checkpoint_io.py`（错误信息逐字保留）。
- P4c：dtype 校验 / `BasisDescriptor` / 指纹原语 → `fastprove/basis_common.py`（指纹逐位不变）。
- P4d：dense 变换生成助手 → `transforms.py` 内部（抽样次序与形状不变）。
- P4e：Gram `rms_scale` 三份 → `structured.gram_rms_scale`（einsum/clamp_min/除法/加法/sqrt 顺序不变）。
- P4f：bounded-noise 位混合原语 → `fastprove/bounded_noise.py`；三种 centering 策略保持显式；
  top-k 预算两处语义差异已补注释（不合并）。

已知延期（不阻塞行为，待单独立项）：

- `evaluation/performance.py` 的 `_performance_metrics` 与
  `_obfuscated_performance_metrics` 共同骨架参数化（计划 §P4 末项）：两者含实测计时副作用，
  收益低于门禁风险，本次未合并，行为保持不变。
- plain/obfuscated 的 greedy 循环平行结构说明：移入 P8 文档。

## P5–P9 与后续修复/重跑（2026-09-20）

`implementation_notes.md` 于 P4 之后不再逐阶段记录；P5–P9 的执行、每步门禁与提交
hash、以及 §3.4/§3.5/§3.6 的修复与重跑证据，统一维护在 `docs/STATUS.md`。本处只留
指针与最终数字，避免两处漂移：

- P5 scripts 库化（3 个脚本入 `fastprove`）、P6 `evals` 打包化 + 指标逐位一致、
  P7 测试重组（8 子目录 + conftest + 两个验收缺口）、P8 文档重建、P9 新克隆冒烟；
- §3.4 Wilson 统一到 `fastprove/evaluation/stats.py`；§3.5 sweep 基础口径 ρ=0 并
  完成重跑（125/125 非性能指标逐位相同）与 39 点 replica（Δtop1=0、Δppl≤9.54e-07、
  ΔJS≤8.63e-11）；§3.6 bf16 debug dtype 修复；
- 当前门：`pytest -q` 424 passed、等价性 12/12、evals exact 门禁（fp32）
  HARD/SOFT/ALL PASS；
- 仍开放：S6/S7 被 checkpoint 缺失阻塞；performance 参数化延后（见 STATUS §6）。
