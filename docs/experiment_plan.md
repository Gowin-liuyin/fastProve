# fastProve 实验执行方案

## 1. 状态、目标与禁止事项

状态：**方案已定义；远端 GPU 已勘察；正式 Llama 评测尚未跑通**。

### 1.0 关键约束：远端「混淆模型」≠ 当前 fastProve 方案（必须知晓）

两台实验机上的 `ModelSplit/models` 中，若存在已「混淆」的权重目录，那是
**旧版 / 另一套 ModelSplit 混淆方案** 的产物，**不是** 本仓库当前实现的
增广状态协变混淆（`z=[h,e]`、可逆基 \(M_\ell\)、链式噪声刷新、非线性检查点）。

因此：

| 允许 | 禁止 |
|---|---|
| 把远端目录当作 **权重存放位置** | 把旧混淆权重当作 fastProve `structural` / `full` / exact 的结果 |
| 使用 **原始明文** Llama 权重，再经 **当前代码** 转换后评测 | 拿旧混淆 checkpoint 与明文比，声称「fastProve 精度损失」 |
| 若做方法对比，必须 **单独标注** 为 “legacy ModelSplit obfuscation” | 在报告中混用两套方案却只写「混淆」一词 |

**正确对照链（协议 F0–P3）：**

```text
原始明文 Llama-3.2-3B-Instruct
    →  [当前 fastProve conversion + 独立 master keys]
    →  structural-only / full-noise 混淆态
    →  与同精度明文 (F0/P0) 做五层评测
```

旧混淆权重 **不得** 替代上式中的「当前 conversion」一步。若无法确认某目录
是否为原始权重，应重新下载官方/可复现的明文 checkpoint，再上传到模型目录，
并在 run record 中记录路径与文件哈希。

### 1.1 远端机器与目录约定

| 角色 | 主机 | 用户 | GPU | 模型目录（只放权重） | 工作目录（代码+结果） | 环境管理 |
|---|---|---|---|---|---|---|
| 主测 | 10.144.144.6 | nss-d | RTX 5090 32GB | `/home/nss-d/dcy/codes/ModelSplit/models` | `/home/nss-d/yhr/code/fastProve` | **conda** |
| 辅测 | 10.144.144.5 | nss-marker | RTX 4090 24GB | `/home/nss-marker/dcy/code/ModelSplit/models` | `/home/nss-marker/yhr/code/fatsProve` | **conda** |

- 代码、配置、`results/`、日志 **只写工作目录**；模型大文件 **只放模型目录**。
- 远端 Python 依赖用 **conda** 创建隔离环境（例如 `fastprove`），不要混用系统
  `python3` 与多套全局包。
- 目标模型名：`Llama-3.2-3B-Instruct`（明文基线）。当前两机上已有同名目录，
  但 **必须先验明是否为原始权重**；在未验证前不得默认当作 fastProve 输入。

后续实验要回答五个问题：

1. exact 模式是否只产生浮点级误差；
2. 随 Softmax 噪声增强，NLL/perplexity、next-token accuracy 和生成 token
   一致率各下降多少；
3. top-k-preserving 和 free-bounded 的排序保持、输出误差及任务精度有何差异；
4. 按预先声明的规则，哪个配置是当前样本上的最佳精度/扰动折中；
5. 哪些安全结论不在原型能力范围内。

任何未执行配置不得生成伪成功记录，任何随机 Tiny LM 结果不得作为真实语言
精度证据。

## 2. 执行总览与阶段门

实验按以下顺序推进，前一阶段不通过则后续阶段停止并写失败记录。

### Gate A：数学与 Tiny 正确性

- 全部变换往返、布局、ChainLinear 恒等式测试通过；
- `forward` 中没有求逆；
- FP32 ChainLinear 小张量 signal max absolute error 不超过 \(10^{-5}\)；
- exact QK、Softmax、Value covariance 和单块输出在固定容差内；
- causal/padding/all-mask/tie/GQA/cache 测试通过；
- Tiny LM exact logits 无 NaN/Inf；
- Tiny LM exact greedy token 与明文一致；
- 相同 seed 产生相同指标。

若 exact greedy 不一致，保存输入、logit 差、top-2 margin、首个分叉位置、
层级最大误差和 dtype，再停止 approximate 结论。

### Gate B：预训练 Qwen2 适配

- 完成 checkpoint 完整性和权重映射；
- 单层 FP32 exact 对照通过；
- 小批全模型 exact 对照通过；
- 确认 tokenizer、special token、position、GQA 和 cache 语义一致；
- BF16/MPS 仅在 FP32 参考通过后单独验收。

### Gate C：72 点校准

- 在完全对齐的 8 条校准样本上执行全部 72 个配置；
- 每个配置均有 success、failure 或 skipped 记录；
- 无 NaN/Inf、实际噪声不超过配置上界；
- 所有失败点保留，不参与但不从汇总中删除。

### Gate D：候选完整评测

- 按第 7 节的预注册筛选规则选择候选；
- 在固定的 64 条完整样本上重跑候选；
- 生成成对指标、性能指标、CSV、图和最终报告。

配置文件中的 `execution.run_in_current_prototype_pass: false` 是本轮暂停标记。
后续必须由用户明确开始实验，不能由报告构建脚本隐式触发推理。

## 3. 环境与资源预检

每次真实评测开始前重新记录：

- UTC 时间与时区；
- OS、Python、PyTorch、Transformers、NumPy 和 fastProve 版本/提交；
- `nvidia-smi` 是否存在、PyTorch CUDA/MPS 可用性；
- 实际 device 名称、activation dtype、batch size；
- 物理/统一内存、评测前可用磁盘；
- 模型权重大小、预计转换副本和 KV-cache 需求；
- 是否发生任何 device/dtype fallback。

当前审计主机为 Apple M5、16 GiB unified memory，PyTorch MPS 可用、CUDA
不可用。完整评测不得把原计划的 CUDA 静默改为 CPU 或 MPS。建议：

- 离线转换在 CPU FP64 中逐层进行；
- 部署推理优先 BF16/MPS，但先用小输入验证支持性；
- 明文、exact 和近似模式一次只加载一个，避免并存多个 1.5B 权重副本；
- 每个模式结束后显式释放模型并记录进程 RSS/MPS allocator 状态；
- 若 MPS 算子不支持而发生 fallback，当前 run 记为 failure，另建经批准的
  CPU run，不能复用同一 run ID。

## 4. 真实模型适配方案

### 4.1 首选本地 checkpoint

首选本机已有：

```text
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/deepseek-r1-distill-qwen-1.5b/base_model
```

审计信息：

- architecture：`Qwen2ForCausalLM`；
- hidden size：1536；
- layers：28；
- attention heads：12；
- KV heads：2；
- head dimension：128；
- intermediate size：8960；
- vocabulary size：151936；
- RMS epsilon：\(10^{-6}\)；
- RoPE theta：10000；
- declared dtype：BF16；
- `model.safetensors` 大小：3,554,214,621 bytes；
- SHA-256：
  `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945`；
- upstream revision：无法从本地工件恢复，记录为 `revision_unavailable`。

加载必须使用 `local_files_only=True`，禁止为补文件自动联网。路径、配置文件
哈希、tokenizer 文件哈希和权重哈希写入 run manifest，不复制权重到仓库。

### 4.2 权重映射

适配器逐层映射并验证：

```text
embed_tokens
input_layernorm
self_attn.q_proj / k_proj / v_proj / o_proj
post_attention_layernorm
mlp.gate_proj / up_proj / down_proj
final norm
lm_head
```

适配时必须显式处理 Q/K/V bias、无 bias 投影、`tie_word_embeddings=false`、
GQA 的 12:2 head 映射和 Hugging Face cache/position 约定。每个映射张量记录
shape、dtype 和内容哈希；转换后信号权重不得随机初始化或从另一 checkpoint
补齐。

分阶段验证：

1. **配置映射**：只加载 config/tokenizer，检查维度、RoPE、special token；
2. **单层 FP32**：固定 1–2 条短 token 序列，比较 Hugging Face 明文层、
   fastProve 明文层和 exact 层的逐检查点误差；
3. **全模型小批**：比较最终 logits、NLL 和 greedy 首个分叉位置；
4. **BF16/MPS**：在完全相同 token 上单独确定容差，不覆盖 FP32 记录；
5. **转换基准**：逐层转换，记录条件数分布、转换耗时和峰值内存。

深层预训练模型的 exact gate 使用 `max|Δlogit|≤1e-3`、logit relative-L2
`≤1e-4`、QK `max|ΔS|≤2e-2` 且 relative-L2 `≤1e-4`、Softmax
`max|ΔA|≤2e-3` 且 relative-L2 `≤1e-4` 的 FP32 参考包络。后一个绝对阈值是
当前 CPU/MPS FP32 runner smoke 中观察到的 1.30--1.42e-3 深层累积误差后登记的
校准包络；每条记录会写入 `metrics.exact_gate_tolerances`，不能只靠 dtype 标签
改变它。它仍要求逐 token agreement、greedy agreement、相对误差和 NaN/Inf
全部通过，不是放宽 BF16 结果。BF16 仍按单独条件记录，若 token/relative 误差未
通过，必须保留失败记录并停止近似结论。checkpoint mix/unmix 的 basis 乘法已在
FP64 中执行以减少深层累积误差。

在单层/全模型 exact gate 通过前，不进入噪声扫描。

### 4.3 已实现适配器与执行接口

本地 Qwen2 适配器和对齐 token cache 已纳入执行 CLI。先准备 token cache，再
做不分配模型权重的预检：

若尚未有数据，先运行只读资产审计；它不会联网、不会加载模型权重，并会写出
机器可读的 `skipped_external_dependency` 或 `ready_for_experiment` 状态：

```bash
python3 scripts/audit_pretrained_assets.py \
  --output results/raw/pretrained_evaluation_status.json
```

当前审计已确认本地 Qwen2 checkpoint/tokenizer 完整，但没有接受为标准 causal-LM
评测的公开语料。当前已准备好
`results/raw/flickr30k_caption_eval_inputs.pt`（405 个长度 24 的完整 block），
但它仍标为非标准 caption-only 候选，不会自动当作通用语言模型基准。

若明确选择这个限定评测，可先把资产状态切换为可执行的候选状态（仍不启动推理）：

```bash
python3 scripts/audit_pretrained_assets.py \
  --dataset-path results/raw/flickr30k_caption_eval_inputs.pt \
  --accept-nonstandard-caption \
  --output /tmp/fastprove_caption_status.json
```

真正启动 sweep 时还必须在同一条命令上显式加入
`--accept-nonstandard-caption`；否则 runner 会在分配模型前拒绝该 cache，避免
把 caption-only 结果误写成标准 LM 证据。

```bash
python3 scripts/prepare_token_cache.py \
  --model-path "/absolute/path/to/qwen2-1.5b" \
  --input-jsonl "/absolute/path/to/eval.jsonl" \
  --output results/raw/eval_inputs.pt \
  --sequence-length 24 \
  --max-samples 64

python3 scripts/preflight_experiment.py \
  --model-path "/absolute/path/to/qwen2-1.5b" \
  --dataset-cache results/raw/eval_inputs.pt \
  --device cuda \
  --activation-dtype fp32 \
  --calibration-count 8 --full-count 64 \
  --output results/raw/preflight.json

python3 scripts/run_accuracy_sweep.py \
  --config configs/eval_sweep.yaml \
  --model-path "/absolute/path/to/qwen2-1.5b" \
  --dataset-cache results/raw/eval_inputs.pt \
  --execute-deferred
```

`preflight_experiment.py` 要求显式指定后续 sweep 的 device，并在不加载模型权重的
情况下执行小型 matmul/Softmax smoke；若启用 `PYTORCH_ENABLE_MPS_FALLBACK`，会
直接失败而不是把 CPU 执行误记为 MPS。

在本机 16-GiB 主机上，建议用 `--calibration-only --batch-size 1
--device cpu --activation-dtype fp32` 作为可复核的最小正式起点；CPU FP32
runner smoke 已通过深层校准 exact envelope。MPS FP32 也能运行，但更慢且接近
统一内存上限；`--device mps --activation-dtype bf16` 只用于资源/适配器校准。
这些参数会逐条写入 run config，且不会静默替换 device/dtype。BF16 结果单独记录，
不能覆盖 FP32 结论。
注意：MPS/CUDA 的隐藏/Value 检查点运行算术固定使用 FP32（离线变换生成和
求逆仍在 FP64），因此预检和 run manifest 都会记录该实际算术 dtype；这不是
把 BF16 结果冒充 FP32 exact gate。

默认执行 `full_sample_count`（当前为 64）；资源受限时显式加
`--calibration-only` 执行 `calibration_sample_count`（当前为 8），并把该阶段
写入每条 record。也可以用 `--sample-count N` 或 `--batch-size 1` 做明确的、可
追溯覆盖；批大小变化会写入 run config。正式性能协议默认使用 3 次预热和 10
次计时；校准若需要缩短控制面耗时，可显式使用 `--warmup-runs`/
`--timed-runs`，覆盖值会写入 run config，不能与默认 full 结果混为同一 cohort。
正式记录默认以样本为重采样单位执行 1,000 次 paired bootstrap；如校准阶段
显式使用 `--bootstrap-replicates 0`，该记录会标明未计算区间，不能冒充完整
评测统计。

`--model-path` 与 `--dataset-cache` 必须成对出现。真实路径只读加载，权重不会
复制到仓库；每个 success/failure record 都带 checkpoint、tokenizer 和 cache
hash。缺 cache、长度/词表不符、资源预检失败会在推理前退出，不会生成伪结果。
该命令仍不会绕过 `execution.run_in_current_prototype_pass: false`；必须显式
提供 `--execute-deferred`，且输出 JSONL 必须为空/不存在。
若真实模型或 cache 预检失败，CLI 会将同一错误复制为 72 条带
`stage=pretrained_preflight` 的 failure 记录，保留完整配置清单；这类记录仍不
参与精度汇总。

## 5. 数据与严格对齐

当前本机没有合适的标准 causal-LM 评测集。后续优先请求一次明确许可，下载
小型公开 WikiText-2 测试数据；预计只需小规模网络和磁盘开销，但实际文件
大小、版本和哈希必须在下载后记录。无需下载时，也可使用已准备的 Flickr30k
公开 caption 做“caption NLL 限定评测”，但必须明确它不是标准语言模型基准，
并在独立 experiment ID 中保留 `caption_only_nonstandard_lm_candidate` 证据范围。

数据准备只执行一次。`prepare_token_cache.py` 只保留长度足够的完整 token block，
因此 greedy generation 不会把 PAD 当成 prompt 尾部；被丢弃的短样本数量写入
cache metadata：

1. 固定 dataset name、config、split 和 revision；
2. 按稳定样本 ID 排序；
3. tokenizer 使用本地 Qwen tokenizer，不添加模式特有模板；
4. 固定截断方向、sequence length、BOS/EOS 与 padding；
5. 先选择 8 条校准样本，再选择包含它们的 64 条完整样本；
6. 缓存 `sample_id`、`input_ids`、`attention_mask` 和原始文本哈希；
7. 对缓存文件计算 SHA-256，并在所有 run 中引用同一个哈希；
8. 原始文本如受许可约束，只保存 ID/哈希和 token，不复制不必要内容。

公平比较的不可变字段：

- checkpoint、权重 hash 和 tokenizer；
- 样本、样本顺序和 tokenized cache；
- sequence length、padding 和 causal mask；
- seed 及请求 ID；
- activation dtype、checkpoint arithmetic dtype、device、batch size；
- teacher-forcing target mask；
- greedy generation 的 max new tokens、temperature=0、无采样、停止条件；
- KV-cache 开关和实现；
- 性能测量的 warmup/repetition。

任一字段变化都必须产生新 experiment ID，不能与旧明文基线拼接比较。

## 6. 72 点噪声消融

固定网格：

```text
tau_max = [0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.10]
alpha = [0.50, 0.80, 0.95]
preserve_top_k = [4, 8, 16]
tau_error = 0.10
```

当前 `configs/eval_sweep.yaml` 的 `base_config`、`model_id` 和 `dataset_id`
仍明确指向随机 Tiny/确定性合成 token，只用于枚举、正确性 smoke 和记录
协议验证。真实 Qwen 运行必须生成一个冻结的派生 manifest，替换模型、数据、
sequence length 和输入 cache hash，但不得改变上述网格；Tiny 记录与 Qwen
记录使用不同 experiment ID，不能在同一精度曲线中混合。

独立记录数：

- plaintext：1；
- exact：1；
- top-k-preserving：\(7\times3\times3=63\)；
- free-bounded：7；
- 合计：72。

即使多个 \(\tau_{\max}=0\) 点数学行为相同，也不去重。它们用于验证每个
\((\alpha,k)\) 分支的零噪声一致性，并保留独立 run ID。

每个 run 复用同一明文参考 logits/生成结果，但必须在 record 中引用其
baseline run ID 和输入 cache hash。禁止为某个近似配置重新采样更有利的数据。

校准完成后使用仓库内的确定性筛选器生成候选 manifest，再让 full sweep 只执行
候选 ID；没有通过 exact gate 的 manifest 不能启动 full run：

```bash
python3 scripts/select_sweep_candidates.py \
  --raw results/raw/calibration/eval_sweep.jsonl \
  --output results/raw/calibration/candidate_manifest.json

python3 scripts/run_accuracy_sweep.py \
  --config configs/eval_sweep.yaml \
  --model-path "/absolute/path/to/qwen2-1.5b" \
  --dataset-cache results/raw/eval_inputs.pt \
  --spec-ids-file results/raw/calibration/candidate_manifest.json \
  --output results/raw/full/eval_sweep_candidates.jsonl \
  --execute-deferred
```

`candidate_manifest.json` 保留校准 failure、端点、Pareto 前沿、cohort 和
exact gate；full raw 只包含入选 ID，并在每条 config 中记录 manifest 路径。

## 7. 校准后的预注册筛选规则

资源允许时，72 点全部在 64 条完整样本上运行。若算力不足，先在 8 条校准
样本上运行全部 72 点，再按下列确定性规则选择完整评测候选：

1. plaintext 和 exact 总是进入完整评测；
2. 所有 failure 保留原始记录，但不进入候选排序；
3. 对 top-k-preserving 的每个 \((\alpha,k)\)，保留
   \(\tau_{\max}=0\) 和 \(0.10\) 两端点；
4. free-bounded 保留 \(\tau_{\max}=0\) 和 \(0.10\) 两端点；
5. 对两种 schedule 分别计算 Pareto 前沿：最小化
   `top1_absolute_drop` 和 `perplexity_relative_increase`，同时最大化
   `actual_noise_infinity_norm`；任何一项严格改善且其他项不变差才支配；
6. 取“端点 ∪ Pareto 前沿”作为完整候选，不根据预期结论手工删点；
7. 未入选点保留校准 success 记录，并在 full-stage manifest 中标为
   `skipped: calibration_only`，写明筛选输入和规则版本。

这样既能减少完整评测成本，又不会隐藏失败点或只报告最优点。若前沿过大，
不再做主观裁剪；应降低 full sample count 并记录资源原因，或请求更强算力。

## 8. 指标定义

### 8.1 层级正确性

对每个允许比较的明文/混淆检查点记录：

- max absolute error；
- mean absolute error；
- relative L2 error；
- cosine similarity；
- NaN count 和 Inf count；
- QK score error；
- exact Softmax probability error；
- Value/attention output signal error；
- causal、padding、all-mask 和 tie 行为；
- cache/full-forward 一致性。

### 8.2 Softmax 与排序

仅通过 debug 评测路径捕获：

- \(D_{\rm KL}(A\|\widehat A)\)；
- Jensen–Shannon divergence；
- Top-k set overlap；
- Top-k changed query fraction；
- valid position 上稳定 tie-break 的 rank correlation；
- clean boundary margin 的 min/mean/max 和分位数；
- actual noise infinity norm；
- zero/near-zero noise query fraction；
- attention output relative L2 error。

masked position 不参与排序和 divergence，且必须保持 \(-\infty\)。报告配置
\(\tau_{\max}\) 的同时必须报告实测噪声范数。

### 8.3 Teacher-forced 端到端精度

在相邻两个有效 token 上计算：

\[
\operatorname{NLL}
=-\frac1N\sum_{i=1}^{N}\log p(x_{i+1}\mid x_{\le i}),
\qquad
\operatorname{PPL}=e^{\operatorname{NLL}}.
\]

同时记录：

- next-token top-1 accuracy；
- next-token top-5 accuracy；
- plaintext/obfuscated top-1 token agreement；
- 有效 target token 数。

对任一“越大越好”指标 \(m\)：

\[
\operatorname{drop}_{\rm abs}=m_{\rm plain}-m_{\rm obf},
\]

\[
\operatorname{drop}_{\rm rel}
=\frac{m_{\rm plain}-m_{\rm obf}}{m_{\rm plain}},
\]

若分母为零则报告 undefined，而不是伪造 0。对 perplexity：

\[
\Delta_{\rm PPL}
=\operatorname{PPL}_{\rm obf}-\operatorname{PPL}_{\rm plain},
\]

\[
\Delta_{\rm PPL,rel}
=\frac{\operatorname{PPL}_{\rm obf}-\operatorname{PPL}_{\rm plain}}
{\operatorname{PPL}_{\rm plain}}.
\]

每项必须同时列出明文绝对值、混淆绝对值、绝对变化和相对变化。

### 8.4 Greedy generation

固定 prompt、cache、停止条件和 max new tokens，记录：

- 每个生成位置的 token exact match；
- 全部生成 token 的 exact-match rate；
- sequence exact match；
- 首个分叉位置；
- 分叉前明文 top-2 logit margin。

exact 模式出现任何分叉即触发 Gate A/B 调查，不得直接归因于正常噪声。

### 8.5 成对不确定性

完整评测以 sample 为重采样单位，使用固定 bootstrap seed 做 1,000 次成对
bootstrap，给出主要 degradation 指标的 95% 区间。token 级 NLL 可同时给出
token 加权点估计，但置信区间不能把同一序列内 token 错当成完全独立样本。

## 9. 性能测量

性能结果必须标为“PyTorch eager reference”，与未来融合/优化实现分栏。

层级 Softmax 诊断需要在评测控制面显式调用 `forward_debug` 才能捕获 clean/noisy
probability、margin 和逐层误差；这不等于生产 API 暴露这些值。每条 run config
写入 `evaluation_debug_capture`，而 prefill/decode 计时仍调用不返回诊断的生产
`forward`。

### 9.1 测量项

- conversion time：读取基础权重后，离线生成基和转换权重的时间；
- prefill latency：固定 batch/sequence 的完整 prompt 延迟；
- decode latency/TPOT：预填充后逐 token 解码延迟；
- tokens/s：分别给出 prefill 和 decode；
- peak process RSS；
- MPS current/driver allocated memory（若 API 可用）；
- KV-cache memory：实测增量和理论估算；
- 模型加载时间另列，不混入 conversion 或 inference。

KV 理论字节数：

\[
\operatorname{bytes}_{KV}
=2\,L\,B\,H_{KV}\,T\,d_h\,s_{\rm dtype}.
\]

对当前 Qwen2 1.5B，\(L=28,H_{KV}=2,d_h=128\)。该公式不含 allocator
碎片、metadata 或临时 buffer，因此必须与实测内存分开标注。

### 9.2 测量协议

- 固定电源状态，关闭明显后台负载并记录；
- 每个模式单独进程执行，随机化模式顺序或使用 ABBA 顺序；
- 预热至少 3 次，正式重复至少 10 次；
- MPS/CUDA 计时前后同步；
- 报告 median、P5/P95 或完整样本，不只报告最佳一次；
- plaintext、exact 和近似使用相同 batch、长度、dtype、cache；
- OOM、fallback 或 thermal throttling 记为失败/警告，不删点；
- reference 与 optimized 结果不得混成一个均值。

当前参考 runner 对大 checkpoint 采用分阶段生命周期：先测 plaintext，再完成离线
转换并释放 plaintext，最后测 obfuscated。`peak_memory.bytes` 是两个独立阶段的
最大值，原始记录同时保留 `peak_memory.measurement_phases.plaintext` 和
`.obfuscated`；它不声称两个完整模型在整个 benchmark 中同时驻留。转换阶段仍可能
短暂同时持有源权重和部署权重，因此必须把 conversion-time OOM 作为真实 failure
记录，而不能用分阶段测量掩盖资源不足。

## 10. 最佳折中点规则

“最佳”只表示当前模型、数据和预先声明约束下的经验操作点，不表示最安全。

报告生成器采用：

1. 只考虑 success、approximate 且
   `actual_noise_infinity_norm > 0` 的 run；
2. 首选同时满足
   `top1_absolute_drop <= 0.01`（不超过 1 个百分点）和
   `next_token_top1_agreement >= 0.99` 的配置；
3. 在可行集内最大化实际噪声无穷范数；并列时最小化 perplexity 相对增幅；
4. 若无配置满足约束，则最小化
   `top1_absolute_drop + max(0, perplexity_relative_increase)`，
   并列时选择实际噪声更大的配置；
5. 同时报告 top-k-preserving 和 free-bounded 各自的最佳点及完整前沿，
   不用单一赢家隐藏 schedule 差异。

该规则把“扰动大小”当作工程轴，不把它解释成可证明的隐私效用。

## 11. 原始记录与失败语义

每个 run 追加一条 JSONL，至少含：

```text
schema_version
run_id
status: success | failure | skipped
timestamp_utc
config
config.expected_run_ids
seed
model
dataset
environment
sample_count
metrics
elapsed_seconds
```

`failure` 必须额外包含：

```text
stage
error.type
error.message
last_completed_sample_id
partial_metrics_available
```

规则：

- JSON 使用 `allow_nan=false`；NaN/Inf 作为计数和失败原因记录，不写非法数值；
- append-only，不覆盖旧 run；当前 CLI 在推理前拒绝非空输出，且不实现
  partial resume。当前重试应使用独立的空 raw 目录；正式 Gate B–D runner
  增加 attempt ID 后，还必须引用原失败 run；
- success 不得带 error；failure 不得省略 error；
- success 必须包含可验证的明文/混淆 NLL、PPL、top-1/top-5、agreement、
  degradation、greedy、Softmax 和 performance 核心指标；任意占位字段不能
  使 run 成为成功证据；
- skipped 必须写结构化 reason，例如 `calibration_only`、`dataset_missing`、
  `oom_preflight` 或 `user_deferred`；
- 捕获异常后先落 failure 记录，再退出非零状态；
- 原始 JSONL 不由报告脚本改写，CSV/图/Markdown 全部可从它重建；
- `config.expected_run_ids` 是 stage 覆盖清单；只有 observed 与 expected
  精确一致、全部成功、四模式齐全且 exact gate 通过，完整 sweep 才标记为
  `completed`；
- 图表用不连线散点表示成功记录，并在图注列出失败/缺失数，不能跨失败点连线；
- 每个结论在报告中引用对应 raw 文件、行号和 run ID。

## 12. 报告产物

完整执行后生成：

```text
results/raw/*.jsonl
results/raw/eval_inputs.pt
results/tables/summary.csv
results/figures/accuracy_vs_tau.png
results/figures/softmax_vs_tau.png
results/REPORT.md
```

`results/REPORT.md` 固定包含：

1. 实验目标；
2. 数学实现摘要；
3. 环境、模型和数据；
4. 精确模式正确性；
5. 噪声强度—精度曲线；
6. Softmax 排序与 Top-k 变化；
7. 最佳折中配置；
8. 性能开销；
9. 威胁模型和安全限制；
10. 完整复现命令。

当前 pending 报告只能写“未执行”，不得生成零值曲线或最佳配置。

## 13. 复现命令

以下是当前原型已约定的控制面。默认均为 plan-only，只有显式执行开关才运行：

```bash
# 查看/验证 Tiny 正确性计划，不执行推理
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml

# 显式执行 Tiny 正确性（后续经用户启动）
python3 scripts/run_correctness.py \
  --config configs/tiny_exact.yaml \
  --execute

# 枚举 72 点并验证计划，不执行扫描
python3 scripts/run_accuracy_sweep.py \
  --config configs/eval_sweep.yaml

# 显式执行 deferred 扫描（后续经用户启动）
python3 scripts/run_accuracy_sweep.py \
  --config configs/eval_sweep.yaml \
  --execute-deferred

# 只从已有 raw 记录重建表、图和报告
python3 scripts/build_report.py \
  --raw results/raw/full/eval_sweep_candidates.jsonl \
  --output results \
  --execute
```

预训练 Qwen 的 `--model-path/--dataset-cache` 已实现；实际执行前还要：

```bash
python3 -m pytest -q
python3 -m compileall -q src evals scripts
```

并把完整终端命令、配置文件 hash、环境清单和退出码写入 run manifest。
若同时保留校准和 full 子目录，报告命令应指向单一 attempt 的 JSONL；候选
full attempt 会显示 `completed_selected_subset`，不代表完整 72 点覆盖。

## 14. 停止条件与外部阻塞

以下情况停止当前阶段并保留证据：

- exact 误差超过容差或 greedy 分叉；
- 权重/tokenizer/hash 与预注册 manifest 不一致；
- 数据缺失且下载未获授权；
- 设备/dtype fallback；
- OOM、磁盘不足或持续非有限值；
- mask、cache 或样本顺序无法与明文严格对齐；
- 任一噪声点违反实际无穷范数上界；
- 报告无法从 raw 独立重建。

数据下载、额外依赖安装、远程 GPU 或大型模型下载属于外部授权事项。遇到这些
阻塞时，报告明确需求、预计资源和已完成的本地工作，不以随机 Tiny 结果替代。
