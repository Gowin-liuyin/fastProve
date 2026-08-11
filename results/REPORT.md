# fastProve 阶段 A–D 结果报告（2026-08-11）

> 本报告覆盖 `docs/IMPLEMENTATION_PLAN.md` 阶段 A–D 的实测结果。所有数字
> 均可由本仓库命令与 `results/raw/` 原始文件复现。**没有可用作语言模型
> 精度证据的预训练 checkpoint**（本地仅有 BERT/GPT-2/whisper/clip 缓存，
> Llama 系只有元数据桩）：本报告的全部端到端数字来自**随机 tiny 模型，
> 仅作正确性证据**（AGENTS.md 允许的范围），不得描述为有意义的大模型
> 精度结果。

## 1. 范围与环境

- 平台：macOS（arm64），Python 3.13.12，torch 2.11.0，CPU，无 CUDA；
  MPS 可用但本报告全部数字来自 CPU FP32（tiny 评测）与 CPU FP32
  （性能记录），`activation_dtype: float32`。
- 实现：`implementation = "eager reference, not a fused kernel"` 贯穿所有
  性能记录。
- 评测基座：`PlainTinyCausalLM`（随机初始化，correctness-only）+ 阶段
  A–C 的协变混淆转换（结构化基、部署权重、词表置换、SecureEmbedding/
  SecureLMHead、Converter 落盘拆分）。
- 数学预言机：`docs/design_identity_check.py` 13 项恒等式全部通过
  （FP64，误差 ≤ 3e-14，`python docs/design_identity_check.py`）。

## 2. 五类结论（按 AGENTS.md 分开陈述，不得混为一谈）

### 2.1 数学精确性（FP32 门禁）

exact 模式在 FP32 下通过全部硬门禁（`results/raw/exact_gate_D1.json`）：

```text
FP32 ChainLinear max abs error             | hard | PASS | 7.15e-07 | 1e-4
Attention rank-flip rate                   | hard | PASS | 0
Attention top-1 match / top-k overlap      | hard | PASS | 1 / 1
Causal mask match, Cache vs no-cache       | hard | PASS | 1 / 1
LM Head inverse-permuted argmax            | hard | PASS | 1.0     | 1
Greedy release-validation sequences        | hard | PASS | 1.0     | 1
Accuracy drop ≤ 0.5 pp / PPL rel ≤ 1%      | soft | PASS
HARD: PASS   SOFT: PASS   ALL: PASS
```

复现：

```bash
PYTHONPATH=src:evals python -m evals --base-config configs/tiny_exact.yaml \
    --config evals/configs/P2.yaml --precision fp32 --keys 1 --layers 1,2,3,4 \
    --check-gates --output results/raw/exact_gate_D1.json
```

词表侧：100 条随机 prompt 的 greedy token 序列经逆置换后与明文
**100% 一致**（`tests/test_secure_lm_head.py::test_greedy_token_sequence_matches_plaintext_after_decode`）。
服务端目录可脱离客户端密钥独立运行（`tests/test_converter.py`）。

### 2.2 浮点偏差

- 块对角 Gram 归约是精度瓶颈：`ρ` 的 FP32 相对误差 1.9e-7（FP64 对照
  2.4e-16），因此 block 恒等式的可达容差 ≈ 1e-5
  （`tests/test_deployed_weights.py`，`_TOL = 1e-5` 为实测依据）。
- exact 模式 logits 偏差随阶段变化（`results/raw/exact_tolerance_*.json`，
  由 `scripts/record_exact_tolerance.py` 生成）：

| 阶段 | max abs err | rel L2 |
|---|---:|---:|
| A4（FP32 化前基线） | 1.594e-06 | — |
| A5（结构化基） | 1.073e-06 | 4.39e-07 |
| B2（部署 attention） | 6.557e-07 | 3.22e-07 |
| B3（部署 FFN） | 7.153e-07 | 2.65e-07 |

- BF16 是独立条件，**不是 exact 门禁**：BF16 下 logit 偏差 ~1.3 量级是
  BF16 舍入本身（`results/raw/exact_gate_D1_bf16.json`：
  argmax 0.984，greedy 0.875，全部在 FP32 下判定 exact 是否成立）。

### 2.3 任务精度退化（近似模式，随机 tiny，仅正确性证据）

完整 sweep：72/72 点成功（2 基线 + 63 topk_preserving + 7 free_bounded），
exact 门禁通过故无 skipped 点（`results/raw/eval_sweep.jsonl`，
`configs/eval_sweep.yaml`）。固定 seed 两次运行 72 个 metric 树**逐位
相同**。

topk_preserving（7 tau × 3 alpha × 3 k 的均值）：

| tau_max | 下一位 top-1 一致率 | greedy 序列一致 | KL(明文‖混淆) | 零噪声查询占比 | Top-k 集合变化率 |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 1.0000 | 1.0 | 0 | 1.000 | 0.0000 |
| 0.001 | 0.9998 | 1.0 | 1.0e-07 | 0.389 | 0.0000 |
| 0.003 | 0.9998 | 1.0 | 8.4e-07 | 0.389 | 0.0000 |
| 0.010 | 0.9995 | 1.0 | 7.6e-06 | 0.389 | 0.0000 |
| 0.030 | 0.9992 | 1.0 | 4.1e-05 | 0.389 | 0.0000 |
| 0.050 | 0.9989 | 1.0 | 7.6e-05 | 0.389 | 0.0000 |
| 0.100 | 0.9989 | 1.0 | 1.4e-04 | 0.389 | 0.0000 |

free_bounded（排名允许改变）：

| tau_max | top-1 一致率 | greedy 序列一致 | KL | Top-k 集合变化率 |
|---:|---:|---:|---:|---:|
| 0.001 | 0.9993 | 1.0 | 1.8e-07 | 0.004 |
| 0.010 | 0.9993 | 1.0 | 1.8e-05 | 0.038 |
| 0.030 | 0.9973 | 1.0 | 1.6e-04 | 0.118 |
| 0.050 | 0.9946 | 1.0 | 4.6e-04 | 0.188 |
| 0.100 | 0.9891 | 1.0 | 1.8e-03 | 0.341 |

观察（仅对随机 tiny 模型成立）：topk_preserving 在整个配置面保持 Top-k
集合不变（预算 2τ < Δ_k 的保证被实测钉住）；free_bounded 在 τ=0.05–0.1
时 top-1 一致率降至 0.99 附近。**这些数字不构成大模型精度结论。**

明文基线校准（harness 缺陷 4）：本地无 Llama-3.2-3B，无法与 OSNIP 论文
基线（0.597）对账。本报告**不引用**任何 retention 数字。

### 2.4 性能开销（eager 参考实现，非融合 kernel）

阶段 B 结束的实测（`results/raw/overhead_B7_d*.json`，
`scripts/measure_overhead.py`；CPU FP32，seq=128，8 decode）：

| hidden | prefill overhead | decode overhead |
|---:|---:|---:|
| 512 | +54.7% | +58.5% |
| 1024 | +30.7% | +50.9% |
| 2048 | +46.1% | +46.6% |

阶段 A 开始时为 **+150.3%**（FP64 checkpoint）；A5 结束时 +58.3%。
D2 记录（`results/raw/performance_D2.json`，d=1024）：prefill +33.4%，
decode +48.0%；TTFT（prefill 代理）25.9ms vs 19.3ms；TPOT 9.7ms vs
6.6ms；profiler kernel 数 18556 vs 2817；KV cache +1.56%（dh 64→66，
与 `(128+130)/(128+128)=+0.78%`@dh128 同构）。

**未达标项（必须照实报）**：

- eager 参考实现的开销（+31%…+58%）**远高于手册 §69 的 ≤5% 目标**；
  算术预算表（文档 §2.3）显示融合 kernel 下增量约 +1.8%，参考实现
  达不到该数。本报告的所有性能数字都带
  `"implementation": "eager reference, not a fused kernel"`，不得以
  参考实现速度代表最终方案。
- tied-embedding 峰值内存超 5% 目标：`untied_deployed` 下词表侧内存
  结构比 `2n/d ≈ 2.03`（d=3072, r=16），3B/BF16 投影 +0.79 GB
  （`performance_D2.json` 的 `tied_embedding_memory` 节；投影未实测，
  本地无 tied 预训练 checkpoint）。`fused_norm_head` 抛
  `NotImplementedError`，不静默回退。

### 2.5 安全限制（`docs/threat_model.md` §5bis 与 §5bis.8）

本原型**不是**密码系统。阶段 A–C 新增并记录的五条限制（各带复现命令）：

1. 块对角基把已知明文恢复 `M` 的样本代价从 O(n) 降到 O(b)/块（A2）。
2. `N` 进入部署权重：观察者可恢复噪声态 `e`（up to 可逆 r×r），
   不可规避（B1.3）。
3. `common_qk` 服务端可见：QK 几何结构不受保护，`runtime_config.json`
   显式标注（C4.3）。
4. `ρ` 的 FP32 精度上界 `‖e‖/‖h‖ ≤ 30`；加大噪声会污染信号路径（A2.4）。
5. tied-embedding 内存取舍与 `fused_norm_head` 的显式拒绝（B4.2）。

连同原有 5bis.1–5bis.6：`h` 可从 `c` 精确线性解码（M 可逆）、静态密钥
复用、`e ≈ hC` 通道、刷新粒度低于"逐请求随机"字面含义等。**本报告没有
任何数字是隐私证据。**

## 3. 未执行/未达成的检查（显式列出）

- 预训练模型评测（Llama-3.2-3B / Qwen2）：未执行。本地无可用 checkpoint
  （仅元数据桩），按任务约定不下载。`run_pretrained_compare.py` 的
  3B FP32 gate、OSNIP 基线对账（harness 缺陷 4）与 tied-embedding 实测
  均依赖它。
- 融合 kernel（阶段 E）：未实现；所有性能数字是 eager 参考。
- MPS/CUDA 上的性能记录：未做（MPS 有单测覆盖正确性；
  `tests/test_mps_runtime.py` 通过）。
- BF16 门禁：按设计不作为 exact 门禁，仅作为独立 condition 记录
  （`exact_gate_D1_bf16.json`）。

## 4. 复现命令

```bash
python docs/design_identity_check.py                 # 13 项恒等式
python -m pytest -q                                  # 297 passed
PYTHONPATH=src:evals python -m evals --base-config configs/tiny_exact.yaml \
    --config evals/configs/P2.yaml --precision fp32 --keys 1 --layers 1,2,3,4 \
    --check-gates --output results/raw/exact_gate_D1.json
PYTHONPATH=src python scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml \
    --execute-deferred                                 # 72 点，写 eval_sweep.jsonl
PYTHONPATH=src python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_B3.json
PYTHONPATH=src python scripts/measure_overhead.py --hidden-size 1024 \
    --output results/raw/overhead_B7_d1024.json
PYTHONPATH=src python scripts/record_performance_D2.py --output results/raw/performance_D2.json
```

## 5. 原始工件

- `results/raw/exact_gate_D1.json` / `exact_gate_D1_bf16.json`
- `results/raw/eval_sweep.jsonl`（72 条记录；第二遍复现对照在 /tmp，未入库）
- `results/raw/exact_tolerance_A4{,_before}.json`、`exact_tolerance_A5.json`、
  `exact_tolerance_B2.json`、`exact_tolerance_B3.json`
- `results/raw/overhead_B7_d{512,1024,2048}.json`
- `results/raw/performance_D2.json`
- `results/raw/recoverability_bound.json`（既有，5bis 复核）
- 源码测试：`tests/`（297 项，覆盖 A5–D1 全部数学恒等式与方向测试）
