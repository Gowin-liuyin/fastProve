# 增广协变推理 v2：工程实施与验收指南

日期：2026-09-16。对应 [方案](../DERIVATION_PACKAGE.md)、[证明 P1–P15](../PROOF_PACKAGE.md) 与 [攻击工作因子合同](work_factor_security.md)。

本文是后续实现合同，不表示下列新接口已经存在。本次只交付设计、证明和独立小矩阵检查；没有替换生产 forward、修复历史 runner 或执行预训练模型评测。

## 1. 实现目标与默认决策

先实现并验证需求原型 `expanded_concat`：复用 AloePri 的 $\hat P,\hat Q$，在 $h\hat P$ 后拼接逐请求辅助噪声，再做联合混合。随后实现推荐候选 `expanded_full`：在 $\ker(\hat Q^T)$ 注入 $D-d$ 维补空间噪声，同时保留拼接链式噪声。二者完成部署参数审计后，再接入一个 block 和 tiny LM。`compact_direct` 只作为消融；所有编码复用同一规范化转换核心，避免维护多套 Transformer。

| 决策 | v2 首轮值 | 原因 |
|---|---|---|
| 主状态 | $c=[h\hat P,e]M$，$D=d+2s$，$r=8/16$ | 严格满足“扩维后增广”要求 |
| AloePri 对 | 每层保存 $\hat P,\hat Q$，$\hat P\hat Q=I$ | 只在离线转换端使用；记录条件数与 hash |
| 联合基 | 双侧 dense 候选；$\kappa(M)\le3$，对照 $1/10$ | 明确条件数；不预设安全收益 |
| 推荐增强 | $c=[h\hat P+aB,e]M$，$\dim a=D-d$ | 真正填掉固定低秩零空间，仍需云端攻击审计 |
| 消融 | `compact_direct` | 检验额外扩维与补空间噪声是否产生实测收益 |
| 噪声链 | $q=[a,e]$，首轮 $C_q=0$，$G_q=\operatorname{diag}(0.5P_a,0.5P_e)$ | concat 令 $a$ 为空；full 分别刷新两类噪声 |
| 残差 | 一个 block 内同一基 | 避免遗漏跨基转换；跨 block 显式换基 |
| exact | logit tau=0，rho noise=0 | 辅助噪声允许非零 |
| approximate | 两种 logit schedule，rho noise 暂为 0 | 单独归因；保留旧 rho-noise 配置供后续实验 |
| 端侧切分 | 默认不启用；可选完整 block 边界 | 不采用仅 RMSNorm 输出的错误切分 |
| 执行 | CPU 小矩阵 → 单 block → tiny → 本地预训练 | 新矩阵不能继承旧 structured 的验收结果 |

正确性 PASS 与攻击成本评价必须是两个字段。可计算、误差受控不代表攻击工作因子提高；能够恢复也不自动判定研究失败，关键是达到同一恢复质量的最便宜攻击是否比基线显著更贵。

## 2. 当前代码位置与差距

本次先通过 codebase-memory 图谱定位，再对未被图谱索引的新文件直接读取。图谱中未包含 `layers/deployed.py` 和 `augmented_noise.py` 等新定义，因此下面以当前磁盘文件为准。审查 HEAD 为 `60035dd7ced63285c02c8884b05c60dfe67cf6e0`；存在未提交修改，HEAD 不是完整快照。

| 位置 | 当前可复用内容 | v2 工作 |
|---|---|---|
| `src/fastprove/transforms.py` | `BasisTransform`、`generate_transform`、`generate_two_sided_transform` | 新增 AloePri 扩维对和补空间对照；重算 cast 后条件数；不重复实现联合基生成器 |
| `src/fastprove/state.py` | MixedState、debug 编解码 | 状态改为扩维宽度 $D$ 加噪声宽度 $r$；保存 `expansion_fingerprint` 和 `basis_fingerprint` |
| `src/fastprove/conversion.py`、`layers/linear.py` | affine conversion、ChainLinear | 实现 $\hat Q_iW\hat P_o$ 和 $\hat Q_iC$ 块；接入逐记录刷新、矩形输出和非零 bias |
| `src/fastprove/augmented_noise.py` | `record_generator`、能量采样/刷新 | 修正校准调用并进入实际链路；实现批次重排/分块重放 |
| `scripts/run_augmented_noise_kernel.py` | v1 小矩阵 runner | 修复 G1 的 A1–A4，配置驱动，逐条保存成功/失败 |
| `src/fastprove/layers/attention.py` | mask、Top-k 预算、有界噪声 | 复用；当前噪声坐标包含 batch_index，增加稳定 sample_id 契约 |
| `src/fastprove/layers/rmsnorm.py`、`swiglu.py`、`router.py` | 非线性模块和测试基础 | 核对 P6–P8、偏置、ties、RoPE/GQA 与真实模型特性 |
| `src/fastprove/layers/deployed.py` | 旧 StructuredBasis 部署融合 | dense 支持须单独实现/明确拒绝；先审计 `Dq=LΓWq`、Gram 和噪声映射 |
| `src/fastprove/models/obfuscated.py` | 旧 block/LM 路径 | 新 basis 不能静默回退旧 structured；backend 显式标记 |
| `src/fastprove/converter.py` | server/client bundle、manifest | 区分未直接导出秘密与代数不可恢复；新增实际参数审计 |
| `src/fastprove/coa_views.py` | 字符串视图合同 | 增加实际张量文件导出和攻击输入隔离；合同本身不是隔离器 |
| `src/fastprove/evaluation/`、`evals/` | 指标、raw schema、sweep | 扩展 v2 字段；旧结果保留原版本和 scope |

两项当前源码事实必须保留：`compute_topk_budget` 已对 ties/短行返回零预算；有界噪声已做中心化和缩放。后续应修具体缺口，不将已存在能力重写为“全新实现”。

## 3. 状态、布局和配置合同

### 3.1 状态与离线产物

每个 `MixedState` 保存混合 tensor、semantic/expanded/noise 宽度、expansion/basis fingerprint、layer/position metadata。在线宽度为 $N=D+r$，不是 $d+r$。合法方转换对象保存 $\hat P,\hat Q,M$ 及离线逆；server state 不因方便调试携带它们。

`expanded_concat` 先产生并验证 $\hat P,\hat Q$，随后对 $[h\hat P,e]$ 使用 $M$。`expanded_full` 求 $\ker(\hat Q^T)$ 行基 $B$，构造 $T=[\hat P;B]$、$F=\operatorname{diag}(T,I_r)$、$M_{eff}=FM$，规范化状态为 $[h,a,e]M_{eff}$。`compact_direct` 令 $\hat P=\hat Q=I_d$，用于判断额外扩维是否提供实测收益。SVD/QR、右逆、求逆、条件数检查均在离线 FP64 执行。保存矩阵族、尺寸、奇异值区间、$\|\hat P\hat Q-I\|$、$\|B\hat Q\|$、联合基与有效变换条件数和 hash。

cast 后的条件数定义为：把实际保存的 FP32 矩阵转 FP64 再计算条件数。另记离线逆残差、保存矩阵乘积的 FP64 残差、真实 FP32 matmul 残差。不得把三者混作一项。

### 3.2 配置：新增字段是提案

下列是 **待实现的 schema v2 示例，当前 `load_config` 不保证可解析**。先增加显式校验，再保存正式配置文件；本次不修改现有 YAML。

```yaml
schema_version: fastprove.augmented.v2
encoding:
  kind: expanded_concat         # 需求基线；推荐候选 expanded_full；消融 compact_direct
  expansion_extra_dim: 32       # D = d + expansion_extra_dim
  expansion_family: aloepri_v2
  hidden_noise_dim: 8
  value_noise_dim_per_head: 2
  basis_family: dense_two_sided
  max_condition_number: 3.0
  activation_dtype: float32
  backend: reference_checkpoint # 与 audited_fused 分开；不自动 fallback
noise:
  complement_beta: 0.0        # expanded_concat；expanded_full 首轮为 1.0
  appended_beta: 1.0
  complement_gamma: 0.5
  appended_gamma: 0.5
  coupling: zero
  refresh_mode: per_request
  calibration_manifest: REQUIRED
  randomness_mode: reproducible_research
attention:
  mode: exact                  # exact | topk_preserving | free_bounded
  tau_max: 0.0
  tau_error: 0.0
  alpha: 0.8
  preserve_top_k: 4
  rho_noise_bound: 0.0
evaluation:
  views: [transcript_coa, cloud_coa]
  key_seeds: [20260916, 20260917, 20260918]
  observation_budgets: [100, 1000]
  device: cpu
```

现有代码将 `topk_preserving`、`free_bounded` 直接作为 attention mode；文档里的 `approximate/topk_preserving` 是概念路径，不能直接当现有枚举传入。沿用现有枚举或显式迁移，避免两套名称并存。

非法尺寸、$\hat P\hat Q$ 残差超限、非有限参数、条件数超限、exact 中非零 tau/rho、生产中 fixed_debug、无法实现的 backend 均应在启动前失败。

## 4. 随机性、缓存与非线性实现规则

噪声记录 ID 使用 `(key_epoch, request_nonce, sample_id, absolute_position, layer_id, operation, noise_role, head_id)`；`noise_role` 区分 `complement` 与 `appended_chain`，attention 再包含 absolute query/key position。batch 数组索引不是 sample_id。每个域的测试都从新发生器首个位置读取，不能因推进同一随机流误判域隔离。

expanded-full 首轮令 $q=[a,e]$、$G_q=\operatorname{diag}(\gamma_aP_a,\gamma_eP_e)$，两类刷新独立。分别用 $\sigma_a=\beta_aH/\sqrt{D-d}$、$\sigma_e=\beta_eH/\sqrt r$ 校准，并分别记录能量；不能把两者合成一个 beta 后失去归因。P4 的方差保持结论逐块应用。只有在二因子消融结束后才考虑 $a/e$ 交叉耦合。

研究复现 seed 存在评估方 manifest；攻击输入不直接携带目标私有 seed。若 seed 或 PRNG 状态本来就在云端实际 runtime 中，则必须作为 cloud view 输入，不能为了攻击失败将其移除。分离记录复现权限不等于改变实际部署可见性。

刷新在生成 K/V 的逻辑时刻发生。cache 保存 `(epoch,basis_id,sample_id,absolute_position,K',cV)`；基或 key_epoch 改变后拒绝直接复用旧 cache，除非实现显式迁移。全段、逐 token、分块 prefill 必须在相同有效前缀上比较。beam 分支继承哪些 request/sequence 标识需要另订协议，首轮不支持 beam search。

attention 行中心化/归一化必须遍历完整有效 key 支持。即使逐坐标原始采样一致，在不同 key 集合上独立中心化也会产生不同噪声；不能将局部 tile 的中心化当成全行结果。在线分块 kernel 需要先求全有效行统计或实现数学等价的多遍归约。

RMS、mask、margin、clipping、logits、Softmax reduction 用 FP32。RoPE 参数、QKV bias、GQA 顺序和最终 norm/head 都来自原 checkpoint。SwiGLU 变换包括 gate/up bias，down bias 在逆补偿后添加。残差更新输出完整混合状态；旧噪声只能按约定传播一次。

Top-k 的数学 margin 保证针对参与预算计算的分数。FP32 先记录实际 budget/noise/addition error；严格模式在 debug 验证集合后，若舍入导致边界失守则缩小预算重算，最终允许零噪声并记录原因。mask 全空输出零并记 `all_masked_rows`；不要用有限 dummy key 补位。

## 5. G0–G5 实施顺序与退出条件

### G0：环境、快照与输入完整性

保存 HEAD、dirty status、相关文件 hash、Python/PyTorch/device/dtype、内存和磁盘。本次仅小矩阵 CPU 检查：Python 3.13.12、PyTorch 2.11.0、NumPy 2.4.4、macOS 26.6.2 arm64、RAM 16 GiB；CUDA 不可用、MPS 可用、无 `nvidia-smi`。检查时剩余磁盘约 178.45 GB。执行解释器为 `/Users/yin/miniconda3/bin/python`。

本次只确认两个本地模型目录存在，未载入或校验权重：

```text
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/deepseek-r1-distill-qwen-1.5b/base_model
/Users/yin/dr-claw/基于协变混淆的边云协同推理/model_artifacts/raw_remote_pull/models/llama-3.2-3b-instruct/base_model
```

`results/raw/real_scenario_prompts_1500.jsonl` 存在，但其模板化及来源约束使它不能自动作为正式 benchmark。完整评测前重新检查设备与显存/统一内存余量、权重 hash、tokenizer、模型许可及样本来源；不得擅用私人数据或下载大型模型/数据。MPS 可用不等于目标模型必定装得下；不静默转 CPU。

### G1：修复阶段 A，完成矩阵与刷新闭环

先修 [旧复核 A1–A4](augmented_noise_stage_a_review.md)：

1. 每个配置实际执行 FP64 reference 与 FP32 deployment，两条记录分别给 dtype 和误差，FP32 gate 只能读取后者。
2. $H=\sqrt{\operatorname{mean}\sum h_{cal}^2}$，$\sigma_e=\beta H/\sqrt r$；输入/输出噪声比例分别用同检查点的信号，不混用输入 $h$ 与输出 $e'$。
3. `record_generator` 进入实际编码、刷新、attention 噪声路径；通过全段、分块、逐 token 和 batch 重排一致性检查。
4. runner 读取配置，预写 expected manifest，每个 spec 完成即追加 success/failure；不同密钥 seed 与同 seed 重复分别标识。旧 raw 永不覆写。

首批宽度取 $d=8,D=12,r=2$，验证 expanded-concat 的非方形 affine 和跨基测试；expanded-full 另有 $k=D-d=4$，随后运行 $d=64,D=96,k=32,r=8/16$。线性 FP32 小测试 signal max abs error ≤ $10^{-5}$；失败先排布局、bias、$\hat P/\hat Q$、$B$、联合基、刷新和数值尺度，不放宽数学定义。对照测 $B\hat Q=0$、$T/F$ 可逆，以及 compact/concat/full 在合法解码后的语义输出一致。

必须直接记录理论秩：expanded-concat 的最大总体秩为 $d+r$、环境宽度为 $D+r$、理论亏损为 $D-d$；expanded-full 在三块协方差正定时理论秩为 $D+r$。任何 runner 不得把 `noise_coverage`、有限样本数值秩或满秩本身命名为安全性。

### G2：导出真实攻击视图，确定 Tier-0 成本上界

三个目录分别保存 `client_secret/`、`server_bundle/`、`evaluator_private/`，攻击导出 `attack_views/<view>/`。使用独立进程和明确文件读取列表；若声称文件系统隔离，必须配置实际 OS 权限/沙箱，仅子进程参数隔离不能这样命名。

`transcript_coa` 含真实链路输入/返回值、公开 checkpoint、公开算法和元数据；`cloud_coa` 加实际转换权重、Gram、注入映射、缓存、普通输出、runtime 配置和可见随机状态。若模型输出本来含有目标信息，须定义允许泄漏函数而非删除观测。

禁止评分方提供目标标签、私有基、私有重放 seed 或目标编码 oracle；从允许输入计算出的等价秘密属于有效攻击成果。公开 embedding 是候选字典，不是已配对的目标明文。对目标 encoder 主动请求自选输入属于 CPA，单列实验。

按 P13 依次检查：公开投影右逆；可见 Gram 重建；可见噪声映射零空间；无标签 SVD 是否恢复预期的 $D-d$ 维零空间；重复输入能否估计条件均值/噪声子空间；公开词表匹配。每个 probe 记录前提是否满足、矩阵来源/hash、秩阈值、条件数、攻击算术 dtype、恢复残差、候选覆盖率，以及准备/逐目标 FLOPs、墙钟、峰值内存和观测量。`not_applicable` 不等于 `resistant`。

**退出条件：**能从实际 bundle 独立复现攻击输入及评分，并得到最便宜 Tier-0 路径的成本上界。若存在直接恢复，记录其成本和接口条件；只有当它相对基线显著变贵时，才能支持工作因子提升。若成本同阶且实测倍率无显著差异，后续昂贵攻击不能覆盖这一结果。增加输出侧变换后重新检查联合参数泄漏，不能靠矩阵名字变化获得成本增益。

### G3：一个 exact block 与 tiny LM

复用现有 plain 路径，接入 expanded-concat 的 reference checkpoint backend。没有同时支持 $\hat P,\hat Q$ 与一般 dense $M$ 时直接拒绝，不能塞进旧 StructuredBasis 并假定状态仍是 $[h,e]$ 或 Gram 稀疏。

按 P6–P9 检查 RMS 投影、RoPE 后 QK、Value 聚合、attention residual、SwiGLU、FFN residual、最终 logits。FP32 建议首轮 absolute/relative tolerance 为：小型 block max abs $10^{-4}$、relative L2 $10^{-5}$；它们是预注册工程门槛，不是理论常数。真实模型按尺度另作独立校准并冻结，不用测试集调容差。QK 门槛必须在 raw 明确写出。

测试包括非零 QKV bias、GQA 4Q/2KV、causal/padding/all-mask、ties、不同绝对位置、cache 增长、basis 不匹配拒绝及 production 不返回 clean outputs。tiny 的 greedy mismatch 要定位到首个分叉及 logit margin；不能以“浮点正常”跳过调查。

数学正确的参考 backend 可以作为实现起点；如果它或其参数泄漏明文，只能标为 correctness/reference，不可自动升级为 audited_fused 或安全部署路径。

### G4：approximate 与真实 checkpoint 精度

固定 rho noise=0，分别运行 topk-preserving 和 free-bounded：tau_max=`[0,0.001,0.003,0.01,0.03,0.05,0.10]`；topk 再遍历 alpha=`[0.50,0.80,0.95]`、k=`[4,8,16]`。free-bounded 不复制无效 alpha/k 组合。预先枚举全部 spec，零噪声/重复配置明确标记，不默默遗漏。

每条记录含实际噪声 infinity norm、margin 分位数、零/近零噪声比例、Top-k overlap/change、rank correlation、KL/JS、输出误差、NaN/Inf、mask/tie 情况。KL/JS 只在共同有效支持计算，处理 FP32 下溢，单独标记全 mask 行。P11 的数值界验证先在 FP64 oracle，再在 FP32 加明确数值容差；容差不能覆盖真正超预算噪声。

Qwen/Llama 实际模型先与官方实现比较 logits/cache/RoPE/偏置/权重布局，再比较混淆路径；同一个错误适配器的两条路径相等不算模型适配成功。冻结 tokenized IDs、split、sample count、preprocessing 和 hash，各模式完全共用。记录 NLL、PPL、top-1/5、token agreement、greedy token/sequence match；含任务指标时记录差值与置信区间。随机 tiny 只作正确性，不称 LM accuracy。

第一次 G4 正式模型评测前必须重新进行 compute preflight；模型/数据不可用时明确 blocked/skipped，不补造数字。

### G5：攻击对照、性能与报告

同时报告 `expanded_concat` 需求基线和 `expanded_full` 推荐候选，并用 `compact_direct` 消融；匹配语义数据、补空间/拼接噪声能量、可见噪声尺度、观测预算及三把独立密钥。这样分别回答“扩维本身是否增加防护”和“填掉原低秩约束是否影响攻击”。完整 AloePri 若未取得可验证实现，则只能称公式代理基线，不能称官方复现。分别扫描补空间噪声 $\beta_a$ 和拼接噪声 $\beta_e$，先用 `{0,1}` 做二因子消融，再对有效轴扩展到 `{0.1,1,3}`；零值必须报告实际扰动为零。学习型反演的训练/校准/目标测试分离。

统计 token top-1/5、候选集合覆盖与大小、序列恢复、相对频率基线和不确定性；多个 token 属于同一 prompt 时，优先按 prompt/密钥聚类 bootstrap，不把所有位置当完全独立样本。不同视图分别报告，不能用 transcript 防御代表 cloud 防御。

按 [工作因子合同](work_factor_security.md) 对每个攻击拆分 `setup` 与 `per_target` 成本，报告 $m\in\{1,10^2,10^4\}$ 或冻结后的实际业务规模。成本原始向量至少含 FLOPs/MACs、参考硬件时间、峰值内存、查询、观测和攻击训练数据；主比较在相同恢复目标 $\rho$ 上计算 $\widehat W$、倍率 $G$ 和 $\log_2G$。未在预算内达到目标的点标为右删失，不能把预算上限写成实际攻击成本。

攻击调度顺序固定为 Tier 0 代数/统计、Tier 1 优化、Tier 2 学习型。每一层都允许攻击者选择本层最优方法，最终取所有层中成本最低者。超参搜索、失败初始化和训练数据生成均计入攻击总成本；预训练通用公开攻击器的成本另列，并报告其是否能跨 key 迁移。

性能单独测 conversion、prefill、decode/TPOT、吞吐、峰值内存、KV 大小、客户端成本和网络字节。GEMM 次数相同不等于 FLOPs/延迟相同；主方案用实际 $(D+r)$ 维与矩形 FFN 形状计算预算。结构化优化改变可见参数时重新跑 G2。不开 GPU 计时就不报告 GPU 性能。

最终选择任务精度、攻击工作因子、防御端成本的 Pareto 候选。基本统计门槛是相对最强基线的 $G$ 的 95% 置信区间下界大于 1；建议预登记十倍工作因子作为有意义工程目标。若都能直接恢复但某候选显著提高恢复成本，可以报告限定攻击集合下的成本倍率；若便宜旁路与基线同阶，则报告“未观测到工作因子提升”。

## 6. 证明到测试的验收映射

| 证明 | 独立 oracle / 必测反例 | 对应实现验收 |
|---|---|---|
| P1/P2 | concat 秩为 $d+r$、亏损 $D-d$；full 在充分条件下秩为 $D+r$；固定 h 条件秩 | 扩维对、补空间基、联合/有效基条件数、三种编码对照 |
| P3 | $\hat Q_iW\hat P_o$、不同 $D_i/D_o$、非零 bias/refresh | ChainLinear FP32 ≤1e-5；right-inverse/inverse 不在 forward |
| P4 | 有界递推；协方差公式；禁止声称全分布平稳 | 能量标定和同检查点统计 |
| P5 | 跨扩维/混合基残差等式；漏掉 $\hat Q_i\hat P_o$ 应失败 | expansion/basis id 检查、噪声不重复相加 |
| P6 | RMS 旋转、gamma/bias；错误整态归一化反例 | FP32 reduction、rho=0 的 exact |
| P7 | RoPE 后共同变换、GQA、混合 Value | mask/cache/position 一致 |
| P8 | SwiGLU 非齐次反例、正确补偿、router ties | neuron/expert 映射和偏置 |
| P9 | 对应局部恒等式；完整归纳见证明 | 一层/两层 LM、最终 head 与缓存 |
| P10/P11 | Top-k 边界、分布/Value 界、零/单有效位置 | 实际 clipping、FP32 排序、KL/JS |
| P12 | 同前缀输出间隔充分条件 | 分叉定位，不把局部界当全网认证 |
| P13 | 给定前提的右逆/投影/Gram 恒等式 | 必须从真实 bundle 获取矩阵再攻击 |
| P14 | epsilon 范数残留、残差反例 | split 只在完整状态边界接入 |

## 7. 结果 schema 与可复现记录

复用 `evaluation/artifacts.py` 的 schema 工具，新增字段不要另造并行记录系统。每次运行至少包含：

```text
schema_version, run_id, status, failure_stage, failure_reason
timestamp_utc, git_head, dirty_status, source_file_hashes
python, torch, device, activation_dtype, reduction_dtype, attack_dtype
model_id, revision, weight_hashes, tokenizer_hash, data_manifest_hash
encoding_kind, basis_family, dimensions, key_id, condition_numbers
complement_beta, appended_beta, calibration_H, sigma_a, sigma_e
complement_gamma, appended_gamma, refresh_mode, domain_scheme
attention_mode, tau_max, tau_error, alpha, top_k, rho_noise_bound
expected_spec_id, completed_spec_id, correctness_metrics
view, visible_file_manifest_hash, attack_id, prerequisites, recovery_target
attack_setup_flops, attack_setup_seconds, attack_per_target_flops
attack_per_target_seconds, attack_peak_memory, queries, observations, train_examples
attack_metrics, censoring_status, observed_work_factor, multiplier_vs_baseline
meaningful_lm_evidence, accuracy_metrics, performance_metrics
```

success/failure/skipped 每个期望 spec 都有一条终态记录。私有种子存受控 evaluator manifest，公开记录用 key_id/hash；这不允许藏起云端实际可见的随机状态。旧 JSONL 可能被 `.gitignore` 忽略，交付时用 manifest 明确列出实际工件。

## 8. 本次交付验证与可执行命令

本次新增的独立数学检查为 `docs/verification/verify_augmented_covariant_v2.py`，仅依赖已有 NumPy，CPU 上执行。它不导入 fastprove，不测试实际 server bundle，不评测模型；目的是为证明保留可重复的小矩阵正例与反例。默认打印 JSON；带 `--output` 时只创建新文件，拒绝覆盖旧结果。

```bash
/Users/yin/miniconda3/bin/python docs/verification/verify_augmented_covariant_v2.py \
  --output results/raw/expanded_augmentation_v2_math_check_20260916.json
```

文件已存在时重放不指定输出即可，不删除原始记录。原始记录含 seed、解释器、NumPy 版本、实际 dtype、逐项数值与脚本 SHA256。实际执行结果以该 JSON 为准；不沿用旧“338 passed”作为本次结果。

本次实际结果：[数学检查 raw](../results/raw/expanded_augmentation_v2_math_check_20260916.json)，40 项全部通过，包含 96 组 Softmax 扰动实例。示例取 $d=4,D=8,r=2$：concat 的总体秩为 $6=d+r$、零空间维度为 $4=D-d$；full 加入 $k=4$ 个补空间自由度后总体秩为 $10=D+r$，语义读取最大误差 `2.6645352591003757e-15`。扩维线性层 FP64 最大误差为 `1.0658141036401503e-14`、实际 FP32 signal max abs error 为 `1.1920928955078125e-6`；跨基残差 FP64 最大误差为 `5.329070518200751e-15`。这些数值只适用于脚本中的尺寸、种子和计算路径。

未执行：生产 backend 回归、完整 Transformer/LM 运行、真实 bundle 攻击、预训练模型 accuracy、MPS/CUDA 测试和性能基准。P9 的完整网络结论依靠文中的模块归纳证明，本次数值脚本只检查局部构件；P12 的全网误差常数仍未认证。

以下是 **已有实现的后续回归命令，本次未执行**，不能替代 v2 接入验收：

```bash
/Users/yin/miniconda3/bin/python -m pytest \
  tests/test_seed_transforms.py tests/test_chain_linear.py \
  tests/test_rmsnorm_rope.py tests/test_attention.py tests/test_swiglu.py \
  tests/test_router.py tests/test_obfuscated_block.py tests/test_converter.py -q

/Users/yin/miniconda3/bin/python scripts/run_correctness.py \
  --config configs/tiny_exact.yaml \
  --output results/raw/v2_existing_backend_correctness.jsonl --execute
```

后一个命令执行当前旧 backend；只有工程师完成 G3 并确认 backend 记录后，才能用于新 v2。完整 sweep 的现有入口是 `scripts/run_accuracy_sweep.py`，其执行 flag 为 `--execute-deferred`；正式配置、模型路径和已验证 token cache 在 G4 冻结后再执行。

本次不创建承诺可运行却尚未实现的 `--encoding-kind`、`--attacker-view` CLI。G1/G2 新参数及配置必须与代码一同实现，再写入正式运行命令。

## 9. 工程交付完成条件

G1 有可靠的新 FP32 raw；G2 可独立重建攻击输入、标签仅在评分端，并记录 Tier-0 成本；G3 exact 的 block/LM/cache 通过；G4 每个噪声点都有终态记录和同数据准确性对照；G5 分视图给出预算—恢复曲线、工作因子倍率、防御成本、限制及可复现报告。

若未观测到成本提升，交付最便宜攻击、接口条件和成本上界；若资源/数据缺失，交付明确缺口。任何情况下都不把生产 API 不返回内部值解释为能够保护可修改内核、读取寄存器或读取客户端秘密的服务器。
