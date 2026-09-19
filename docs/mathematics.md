# fastProve as-built 数学规格

状态：CURRENT
日期：2026-09-20
上游依据：AGENTS.md、docs/architecture.md、docs/IMPLEMENTATION_PROMPT_dual_channel.md
适用范围：当前代码实际实现的数学构造、离线融合公式、误差与数值策略；公式逐条对照 `src/fastprove` 源码

本文是 as-built（按实现现状）规格：只描述当前 `src/fastprove` 实际执行的数学，
每条公式给出对应的 `file.py:line`。2026-07 时代的旧版数学说明已归档为
[`docs/archive/2026-09-19_mathematics_july_construction.md`](archive/2026-09-19_mathematics_july_construction.md)。

实现是 PyTorch eager 参考实现；它验证代数恒等式、掩码语义、确定性噪声与端到端
Tiny Causal LM 的正确性，不是已经融合或经过密码学证明的部署内核。威胁边界见
[`docs/threat_model.md`](threat_model.md)。

---

## 1. 记号与权重布局

全文采用行向量约定（`src/fastprove/conversion.py:1`）：

\[
Y=XW+b .
\]

对 \(X\in\mathbb R^{n\times d_{\rm in}}\)，数学布局权重为
\(W_{\rm math}\in\mathbb R^{d_{\rm in}\times d_{\rm out}}\)；`torch.nn.functional.linear`
接收的 PyTorch 布局是 \(W_{\rm pt}=W_{\rm math}^{T}\)。
转换助手：`math_to_torch_weight` / `torch_to_math_weight`
（`src/fastprove/conversion.py:16`、`:24`）。所有融合产物在数学布局中构造，最后一步才转置。

## 2. 混合状态与结构化基

### 2.1 状态定义

第 \(\ell\) 层语义信号与辅助噪声为 \(h_\ell\in\mathbb R^{d}\)、\(e_\ell\in\mathbb R^{r}\)，
增广后右乘可逆基：

\[
z_\ell=[h_\ell,e_\ell],\qquad c_\ell=z_\ell M_\ell .
\]

生产路径携带 `MixedState(mixed, basis)`（`src/fastprove/state.py:14`），只含张量与
`BasisDescriptor`（`src/fastprove/basis_common.py:22`：signal_dim / noise_dim /
condition_number / fingerprint），不含逆矩阵。`encode_debug` / `decode_debug`
（`src/fastprove/state.py:66`、`:100`）必须显式传入 `enabled=True`，否则抛
`PermissionError`（`src/fastprove/state.py:49`）。

### 2.2 因子形式：\(M=\Pi_1 D B \Pi_2\)

当前实现生成的是**双侧排列**的因子分解，而非单侧 \(\Pi DB\)：

\[
M=\Pi_1\,D\,B\,\Pi_2 ,
\]

其中 \(\Pi_1,\Pi_2\) 是排列，\(D\) 是有界非零对角缩放，\(B\) 是块宽为 \(b\) 的块对角正交矩阵。
模块 docstring 写明 `M = Pi1 @ D @ B @ Pi2`（`src/fastprove/structured.py:1`），
`mix` 的四个步骤——按 `perm_in` 聚集、乘 `scales`、块正交作用、按 `perm_out` 重排——
对应右乘这四个因子（`src/fastprove/structured.py:153`）；`unmix` 逆序执行
（`src/fastprove/structured.py:164`）。

> **与 AGENTS.md 的差异（已核对，非代码缺陷）**：AGENTS.md 推荐的默认结构是
> \(M=\Pi DB\)（三个因子）。实现是它的双侧排列推广；双侧排列不影响条件数
> （排列因子正交），只改变 \(M\) 作用的坐标顺序。本文件以代码为准。

### 2.3 生成与条件数策略

`generate_structured_basis(signal_dim, noise_dim, seed, domain, block_size,
max_condition_number=10.0, dtype=float32)`（`src/fastprove/structured.py:425`）：

1. 抽样 `perm_in`、`perm_out` 两个均匀排列（`:461`、`:462`）；
2. 对角缩放 \(D_{ii}=\exp((2u_i-1)\cdot\frac12\ln\kappa_{\max})\)，\(u_i\sim U[0,1)\)
   （`:463`–`:465`），因此 \(\kappa_2(M)=\max D/\min D\le\Pi\) 上限；
3. 每个块独立抽样高斯矩阵并 QR 正交化，用 `sign(diag(R))` 固定符号（`:466`–`:475`）；
4. 实测条件数 \(=\max D/\min D\)，超过
   `max_condition_number + max(1e-9, 1e-9·κ_max)` 即报错（`:477`–`:483`）；
5. 先 cast 到部署 dtype，再计算指纹与 Gram 块，保证指纹与实际存储值一致
   （`:485`–`:503`）。

`block_size` 必须整除 \(d+r\)；当 \(d+r\le b\) 时退化为单个稠密正交块
（`:450`–`:456`）。Value 基总是使用单块构造（`src/fastprove/models/obfuscated.py:126`–`:141`）。
部署矩阵只接受 FP32/FP64（`src/fastprove/basis_common.py:37`）；FP16/整数/复数被拒绝。

`StructuredBasis.__post_init__` 会复查排列双射、缩放非零有限、块正交
（FP32 容差 `5e-6`，`:107`–`:115`）与指纹一致（`:98`–`:106`）。
`validate_integrity`（`:328`）额外验证 \(M M^{-1}=I\)（FP32 容差 `1e-4`）。
`structured_condition_number`（`:517`）用 FP64 稠密矩阵计算 2-范数条件数，用于记录。

### 2.4 信号读出、Gram 与 \(\rho\)

离线转换端取：

\[
P=M^{-1}[:,:d]\quad(\text{signal read-out}),\qquad
N=M^{-1}[:,d:]\quad(\text{noise read-out}),
\]

\[
M_{\rm top}=M[:d],\qquad M_{\rm bot}=M[d:].
\]

对应 `signal_projection`、`noise_projection`、`signal_rows`、`noise_rows`
（`src/fastprove/structured.py:234`、`:239`、`:244`、`:249`）。生产路径满足
\(h=cP\)、\(e=cN\)。

Gram \(A_{\rm gram}=PP^{T}\) 在 `perm_out` 坐标下块对角（`:397`–`:422` 校验非块部分
\(\le 10^{-9}\)），因此

\[
\|h\|^2=c\,A_{\rm gram}\,c^{T}
\]

可以按块做 \(O(nb)\) 收缩，只物化每块一个标量（`gram_signal_norm_squared`，
`src/fastprove/structured.py:354`）。RMS 标量为

\[
\rho=\sqrt{\|h\|^2/d+\epsilon}
\]

（`gram_rms_scale`，`:380`；`StructuredBasis.rms_scale`，`:215`），全部归约在 FP32 中
执行并 `clamp_min(0)`。

**数值约束**：Gram 靠相消湮灭辅助子空间，FP32 相对误差按 \((\|e\|/\|h\|)^2\) 增长；
实测表见 `signal_norm_squared` docstring（`:191`–`:208`）。因此
`AUXILIARY_MAGNITUDE_BOUND = 30.0` 是实测的数值上限（`src/fastprove/policy.py:21`），
`check_auxiliary_magnitude`（`src/fastprove/policy.py:24`）在转换时报错，
`validate_auxiliary_budget`（`src/fastprove/layers/deployed.py:365`）按稳态估计
\(\|hC\|/(1-\|G\|_2)\) 检查。

## 3. \(\ker(L)\) 与双通道噪声账本

### 3.1 语义读出算子 \(L=P\) 与核空间

语义读出 \(h=cL\) 的算子就是 \(L=P\)。噪声子空间

\[
\ker(L)=\{n:nL=0\}
\]

是唯一与非线性检查点严格交换的噪声方向：检查点只消费 \(h=cL\)，落在该子空间的
扰动不改变输入。`free_noise_basis`（`src/fastprove/structured.py:254`）对 \([n,d]\) 的
\(P\) 做 FP64 SVD，取零奇异值对应的左奇异向量转置作为行基；当前生产构造
\(D=d\)，故 \(\dim\ker(L)=r\)，与 `noise_rows` 张成同一子空间。该基只在离线转换/诊断
路径调用，`forward` 从不调用（`:275`）。

`channel_decomposition`（`:287`）把任意噪声向量正交拆分为 channel 0（\(\ker L\)）与
channel 1，并返回 \(\|nL\|\)（恰好是语义误差）、分量范数与 `free_fraction`。

**信息论地位（不得夸大）**：channel 0 噪声对 \(h\) 的条件熵贡献**恰为 0**——知道 \(L\)
的观察者可从 \(c\) 确定性恢复 \(h\)，任意大的 in-cone 噪声都不改变这一点
（`src/fastprove/structured.py:267`–`:273`；T4 见
`results/raw/noise_ordering_analysis_20260919.json`）。channel 0 **不得**被描述为
隐私来源。

### 3.2 机器可读账本

`DualChannelBudget`（`src/fastprove/noise_budget.py:126`）把每个噪声旋钮分配到唯一的
通道，并强制：

- channel 0：`accuracy_cost_bound == 0.0` 且
  `information_theoretic_hiding_bits == 0.0`，`bound_source is None`
  （`ChannelBudget.__post_init__`，`:81`–`:104`）；
- channel 1：必须带白名单误差来源
  `("margin_delta_k", "value_diameter_D_V", "rho")`（`:42`）与有限的
  `accuracy_cost_bound`（`:105`–`:118`）。单位 channel-1 噪声对 \(h\) 的扰动恰好是
  \(nL\)（定理 2）。

`DualChannelBudget.from_config`（`:169`）从配置派生：free 维度取
`basis.free_noise_basis().shape[0]`（无 basis 时退回 `hidden_noise_dim`），
semantic 的 `sources = {rho: rho_noise_bound, margin_delta_k: max(tau_max,
tau_error, router_tau_max)}`，`accuracy_cost_bound = max(sources)`。
当注意力模式是 `plaintext`/`exact` 而 semantic 成本非零时，账本模式记为
`exact_attention_with_semantic_noise`（`:210`），拒绝把它标成 `exact`。
`assert_consistent_with`（`:274`）在转换时校验账本与配置及真实 \(\ker L\) 维度一致。
`manifest()`（`:240`）写出 `fastprove.noise_budget.v1` 记录。

`semantic_hiding_bits`（`:47`）给出高斯语义信道的保留比特
\(0.5\log_2(1+\mathrm{SNR})\)，SNR \(=(\sigma_h/\sigma_\delta)^2\)；它是保留量口径，
**不是**安全等级，也不建模 `docs/PROBLEMS.md` 记录的已知明文代数攻击。

## 4. 仿射层与噪声刷新（参考路径）

设明文仿射层 \(y=hW+b\)，辅助通道
\(e'=hC+eG+\xi\)，块矩阵

\[
K=\begin{bmatrix}W&C\\0&G\end{bmatrix},
\]

离线转换（`convert_affine_chain`，`src/fastprove/conversion.py:40`；\(K\) 的构造在
`:80`–`:99`）：

\[
\widetilde W=M_{\rm in}^{-1}KM_{\rm out},\qquad
\widetilde b=[b,\xi]M_{\rm out},
\]

分别见 `:100`–`:106` 与 `:115`–`:122`。于是

\[
c_{\rm out}=c_{\rm in}\widetilde W+\widetilde b
=[hW+b,\ hC+eG+\xi]M_{\rm out}.
\]

`ChainLinear`（`src/fastprove/layers/linear.py:21`）保存
\(\widetilde W_{\rm pt}=\widetilde W_{\rm math}^{T}\) 与 \(\widetilde b\)；
`forward` 只执行一次 `F.linear`，不调用任何逆运算（`:209`–`:234`）。
刷新分两种模式（`VALID_REFRESH_MODES`，`:31`）：

- `fixed_debug`：\(\xi\) 固定写入 \(\widetilde b\)，只用于正确性/回归测试；
- `per_request`：在指定操作内部从请求种子派生 \(\xi\)，经输出基混合后加入
  （`_request_refresh`，`:186`–`:207`）。固定刷新不等于隐私。

`BasisTransform`（`src/fastprove/transforms.py:27`）是 dense 参考变换，
`inverse` 在离线构造时生成（`:37`–`:81`）；`generate_transform` / 
`generate_two_sided_transform`（`:229`、`:274`）是参考生成器。

## 5. 生产路径：离线融合与无解码 forward

### 5.1 总览

`models/obfuscated.py` 的 `_run`（`src/fastprove/models/obfuscated.py:433`）执行部署
forward；融合权重由 `layers/deployed.py` 构造，公式汇总见该模块 docstring
（`src/fastprove/layers/deployed.py:11`–`:30`）：

```text
c              = [h, e] @ M
P              = M^-1[:, :d],  N = M^-1[:, d:]
rho            = sqrt(||h||^2 / d + eps)
Q              = (c @ Wq) / rho        Wq = P @ diag(gamma_a) @ W_Q
K              = (c @ Wk) / rho        Wk = P @ diag(gamma_a) @ W_K
c_V            = (c @ Wv) / rho        Wv = [P diag(gamma_a) W_V | C_V] @ M_V
c_post         = c + mixed_ctx @ Wattn + (c @ N) @ Wnz_a + xi_a
z'             = silu(c_post @ Wg / rho2) * (c_post @ Wu / rho2)
c_next         = c_post + z' @ Wffn + (c_post @ N) @ Wnz_f + xi_f
```

部署 block 的 GEMM 数量与明文 block 相同；干净 \(h\)、注意力上下文 \(O\)、FFN 输出
从不物化（`_run` 中注释 `:600`–`:601`）。

### 5.2 Q/K/V 与噪声耦合

`build_deployed_attention`（`src/fastprove/layers/deployed.py:101`）：

- \(\Gamma_a\) 吸收：`absorbed = P * gamma[None, :]`（`:187`–`:188`），
  `deployed_query = absorbed @ q_math`、`deployed_key = absorbed @ k_math`（`:189`–`:190`）；
- Value 融合：对第 \(j\) 个 KV 头，
  \([\,P\Gamma_a W_V^{(j)}\ |\ C_V^{(j)}\,]\,M_{V_j}\)
  （`:192`–`:207`），其中 \(C_V\in\mathbb R^{n\times H_{kv}\times r_h}\) 直接从混合状态
  读噪声（`src/fastprove/models/block_builder.py:34`–`:51`）。
  \(c_V\) 的第二分量不来自独立 \(e_V\) 通道，而是 \(c\) 的线性像——这是当前实现；
- 输出融合（`:209`–`:221`）：

\[
W_{\rm attn}[i]
=M_{V_{j(i)}}^{-1}[:,\!:\!d_h]\,W_O^{(i)}\,(M_{\rm top}+C_OG\,M_{\rm bot})
+\frac{1}{H}\,M_{V_{j(i)}}^{-1}[:,\!d_h:]\,A_{\rm aux}M_{\rm bot},
\]

其中 \(j(i)=\text{kv\_index}[i]\)，\(A_{\rm aux}\in\mathbb R^{r_h\times r}\) 把 Value
噪声折入 hidden 噪声，\(1/H\) 是 Query 头平均因子。forward 中
`mixed_context`（宽度 \(d_h+r_h\)）直接右乘 `deployed_attn_out`，不先解码
（`src/fastprove/models/obfuscated.py:602`–`:606`）。

`_run` 中 Q/K/V 的部署计算：`q_flat = (mixed @ deployed_q)/scale + bias`
等（`:470`–`:486`），`scale` 即 \(\rho\)。

### 5.3 仿射噪声刷新与 \(\xi\)

\[
W_{\rm nz}=(G-I)M_{\rm bot}
\]

在离线构造（`src/fastprove/layers/deployed.py:224`、`:344`；
`fixed_debug` 时 `refresh_out = diag(ξ)M_bot`，`:225`–`:232`、`:345`–`:352`）。
forward 的残差更新是

\[
c_{\rm post}=c+\mathrm{einsum}(\text{mixed\_ctx},W_{\rm attn})
+(cN)W_{\rm nz,a}+\xi_a,
\]

\[
c_{\rm next}=c_{\rm post}+z'W_{\rm ffn}
+(c_{\rm post}N)W_{\rm nz,f}+\xi_f,
\]

见 `src/fastprove/models/obfuscated.py:602`–`:614` 与 `:627`–`:637`。
因为 \(e=cN\)，加入 \((eG-e)M_{\rm bot}\) 恰好把辅助状态从 \(e\) 更新为 \(eG\)；
`fixed_debug` 的 \(\xi\) 行是 `deployed_attn_refresh_out.sum(dim=0)`
（`src/fastprove/models/obfuscated.py:399`–`:400`），
`per_request` 则抽样 \(\xi\sim\mathcal N(0,1)\cdot\sigma_{\rm free}\cdot\text{scale}\)
后右乘 `noise_rows`（`:401`–`:412`）。

channel 0 每分量 \(\sigma_{\rm free}\) =
`deployed_free_noise_sigma(free_noise_energy_ratio, r)`
（`src/fastprove/models/obfuscated.py:186`–`:190`；实现
`src/fastprove/augmented_noise.py:65`），部署约定下等于
`free_noise_energy_ratio` 本身（标定见该函数 docstring）。该旋钮只影响数值条件，
不买任何隐私（§3）。

### 5.4 FFN

`build_deployed_feed_forward`（`src/fastprove/layers/deployed.py:268`）：

\[
W_g'=P\Gamma_f W_gP_f,\qquad
W_u'=P\Gamma_f W_uD_fP_f,\qquad
W_d'=P_f^{T}D_f^{-1}W_d,
\]

对应 `deployed_gate`（`:334`）、`deployed_up`（`:335`–`:337`）与
`down_converted`（`:339`）。输出融合把 \(M_{\rm top}+C_dM_{\rm bot}\) 与
\(C_zM_{\rm bot}\) 一起吸收（`:340`–`:342`）。forward 中
`z_prime = F.silu(gate_prime) * up_prime`，再 `post_mixed + z_prime @
deployed_ffn_out + noise + refresh`（`src/fastprove/models/obfuscated.py:624`–`:637`）。

### 5.5 风险记录

`layers/deployed.py` docstring `:32`–`:44` 记录两条不可省略的后果：\(N\) 出现在部署
权重里使观察者可恢复 \(e\)（up to 可逆 \(r\times r\)）；部署矩阵是 \(P\) 与
\(M_{\rm top}\) 的乘积，从它们恢复 \(M\) 是分解问题而非密码学难题。正式口径见
`docs/threat_model.md` 5bis.3/5bis.8。

## 6. RMSNorm、RoPE 与 Q/K

### 6.1 RMSNorm

gamma-free FP32 统计：`rms_no_gamma_fp32`（`src/fastprove/layers/rmsnorm.py:91`）：

\[
\operatorname{rmsnorm}_0(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}} ,
\]

加法、均值、`rsqrt` 与乘法在 FP32；学习缩放
`rms_norm_fp32`（`:105`）在 FP32 乘入后转回激活 dtype。

对正交 \(R\)（\(u=hR\)）有 \(\operatorname{rms}(u)=\operatorname{rms}(h)\)。
\(\Gamma=\operatorname{diag}(\gamma)\) 不与 \(R\) 交换，必须在后续投影中吸收：

\[
W_Q'=R^{T}\Gamma W_Q T_Q
\]

的参考形式是 `absorbed_rms_projection`（`:117`–`:142`）。

### 6.2 RoPE 与 RoPE 后的共同正交变换

`apply_rope`（`:150`）实现 Llama 风格 RoPE（FP32 计算），顺序固定：

1. 归一化信号投影得到 \(Q,K\)；
2. RoPE；
3. 对共享同一 KV 头的 Query 头与 KV 头应用共同正交 \(C_j\)：
   `apply_qk_orthogonal_after_rope`（`:191`–`:223`），

\[
Q_i'=\operatorname{RoPE}(Q_i)C_j,\qquad
K_j'=\operatorname{RoPE}(K_j)C_j
\;\Longrightarrow\;
Q_i'K_j'^{T}=\operatorname{RoPE}(Q_i)\operatorname{RoPE}(K_j)^{T}.
\]

GQA 通过显式 `kv_index` 映射，每个 Query 头只使用其 KV 头的 \(C_j\)
（`src/fastprove/layers/rmsnorm.py:208`–`:223`）。当前实现**不假设** \(C\) 与 RoPE
交换；`C_j` 在 RoPE 之后应用且无法吸收进部署权重，因而被归类为服务端可见密钥
材料（`src/fastprove/models/obfuscated.py:101`–`:114`）。

## 7. Softmax：四种模式

### 7.1 掩码与安全 Softmax

有效位置由 Query 有效性、Key 有效性与因果性共同决定
（`src/fastprove/models/obfuscated.py:571`–`:576`）：

\[
\operatorname{valid}(q,k)=m_q\land m_k\land(p_k\le p_q).
\]

`safe_masked_softmax_fp32`（`src/fastprove/layers/attention.py:73`）以精确 \(-\infty\)
填充无效位置，全掩码行返回全零概率，指数/归约/归一化全部 FP32；不添加有限 dummy
位置。

### 7.2 `plaintext`

\[
S=\frac{QK^{T}}{\sqrt{d_h}},\qquad A=\operatorname{softmax}(S),\qquad O=AV .
\]

Softmax 由同一个 `safe_masked_softmax_fp32` 产生；注意力核实现见
`ObfuscatedAttention._run`（`src/fastprove/layers/attention.py:245`–`:360`）。

### 7.3 `exact`

使用 RoPE 后共同正交 Q/K、混合 Value 路径与零 logit 噪声：

\[
S'=\frac{Q'K'^{T}}{\sqrt{d_h}}=S,\qquad
O_{\rm exact}=\operatorname{softmax}(S)\,c_V .
\]

参考级恒等式由 `docs/design_identity_check.py` 在 FP64 下验证（13 项恒等式全过）。
数值级容差：QK 分数 \(\le 5\times10^{-5}\)、Softmax 概率 \(\le 5\times10^{-6}\)、
注意力输出 `atol=5e-5`、单块最终信号 `atol=rtol=1e-4`、Tiny LM logits
`atol=rtol=2e-4`（`tests/models/test_obfuscated_block.py:50`–`:76`、
`tests/models/test_tiny_lm.py:55`）。

**合同边界**：exact 正确性合同要求 `rho_noise_bound: 0` 与 `fixed_debug`
（`configs/tiny_exact.yaml`）。72 点 sweep 的 plaintext/exact 点从
`base_config: configs/tiny_approx.yaml` 继承了 `rho_noise_bound: 1e-5` 与
`per_request`，因此**按当前配置重跑的 sweep 基线点不是 ρ=0 意义上的 exact**
（`docs/PROBLEMS.md` §3.5；`noise_budget.py` 会把它标为
`exact_attention_with_semantic_noise`）。已落盘的 exact 门禁
`results/raw/exact_gate_D1.json` 走 `tiny_exact`，不受影响。

列排列 + 行公共偏移的精确族为

\[
\operatorname{softmax}(SP+a\mathbf 1^{T})=\operatorname{softmax}(S)P .
\]

词表侧通过 `TokenCodec` 的排列实现该族：`encode`/`decode`
（`src/fastprove/codec.py:48`、`:54`），SecureEmbedding 行置换
（`src/fastprove/layers/embedding.py:30`、`:57`），SecureLMHead 列置换
（`src/fastprove/layers/head.py:26`、`:53`；`softmax(l\,\Pi_{\rm voc}) =
softmax(l)\Pi_{\rm voc}`）。该族只重标位置，不隐藏分数多重集合或排序结构。

### 7.4 `topk_preserving`

只对有效 logits 加噪。`compute_topk_budget`（`src/fastprove/layers/attention.py:95`）：

\[
\Delta_k=S_{(k)}-S_{(k+1)},\qquad
\tau=\min\left(\tau_{\max},\frac{\alpha\Delta_k}{2},\tau_{\rm error}\right),
\qquad \|\eta\|_\infty\le\tau,
\]

实现条件（`:119`–`:133`）：仅当有效位置数 \(>k\)、margin 有限且 \(>0\) 时预算非零，
否则 \(\tau=0\)（计入零噪声 Query 比例）。`_sample_bounded_noise`（`:141`）按绝对
query/key 位置做坐标稳定哈希（CPU 整数运算），在有效坐标上去均值、按最大绝对值
归一，并乘 `nextafter(tau, 0)` 使实测无穷范数严格不超过配置界（`:207`–`:217`）。
因为 \(\alpha<1\)，\(2\|\eta\|_\infty\le\alpha\Delta_k<\Delta_k\)，Top-k 集合不变
（tie 除外）。该预算规则与 `StableRouter._margin_and_tau` 的掩码/margin 语义**故意
不同**，不得合并（`:100`–`:107`）。

### 7.5 `approximate/free_bounded`

\[
\tau=\min(\tau_{\max},\tau_{\rm error}),
\]

所有有效行取同一预算，允许 Top-k 与完整排序改变（`attention.py:308`–`:327`）。

### 7.6 输出与调试

\[
\widehat A=\operatorname{softmax}(S+\eta),\qquad
\widehat O=\widehat A\,V,\qquad
\widehat O-O=\left[\operatorname{softmax}(S+\eta)-\operatorname{softmax}(S)\right]V .
\]

生产 `forward`（`attention.py:362`）只返回混合注意力输出；clean/noisy logits、概率、
margin、预算、噪声只由显式启用的 `forward_debug`（`:389`）返回，且当
`debug_enabled=False` 时抛 `PermissionError`。噪声生成是记录种子下的可复现实验
随机性，不是密码学流。

## 8. SwiGLU

`generate_swiglu_transform`（`src/fastprove/layers/swiglu.py:45`）生成共同神经元
排列 \(P_f\) 与有界正缩放 \(D_f\)（元素 \(\in[0.5,1.5)\)）。恒等式：

\[
z=\operatorname{SiLU}(g)\odot u,\qquad
g'=gP_f,\quad u'=uD_fP_f
\;\Longrightarrow\;
\operatorname{SiLU}(g')\odot u'=zD_fP_f,
\]

\[
W_d'=P_f^{T}D_f^{-1}W_d
\;\Longrightarrow\;
(zD_fP_f)W_d'=zW_d .
\]

参考转换 `convert_swiglu_weights`（`:69`–`:127`，公式见其 docstring `:80`–`:84`）
与部署融合 `build_deployed_feed_forward`（`src/fastprove/layers/deployed.py:334`–`:339`）
一致；R 与 \(\Gamma_f\) 在 Gate/Up 上被同时吸收。非线性后的辅助刷新
\(e_z=z'C_z+e_{\rm side}G_z+\xi_z\) 的参考实现是 `refresh_swiglu_noise`
（`:130`–`:155`）。不得声称 SiLU 与任意缩放或稠密混合交换。

## 9. MoE Router

`StableRouter`（`src/fastprove/layers/router.py:75`）实现专家排列

\[
r'=rP_E+\eta_r ,
\]

并在 `2\|\eta_r\|_\infty<r_{(k)}-r_{(k+1)}\) 时保持选中专家集合不变。
细节：

- `_margin_and_tau`（`:118`）：对 **canonical** 专家序排序，margin
  `clamp_min(0)`，\(\tau=\min(\tau_{\max},\tau_{\rm error},
  \alpha\,\text{margin}/2)\)；margin \(\le 0\) 时 \(\tau=0\)；
- `_noise`（`:146`）：逐坐标 `seed_for("router-noise", layer_id, row,
  canonical_expert)` 生成均匀噪声，在全部专家上中心化、按最大绝对值归一，乘
  `nextafter(tau,0)`；由于在 canonical 序生成后按 `permutation` 重排，物理专家
  与决策一致；
- tie 使用稳定 `argsort(..., stable=True)`，先还原 canonical 序再选物理索引
  （`_select_physical`，`:109`–`:116`）；
- 生产 `forward` 只返回 `physical_expert_indices` 与 `gate_weights`，不返回原始
  logits（`RouterDecision`，`:44`）；margin/预算/噪声只在 `forward_debug`（`:222`）
  返回。
- 物理专家重排助手 `reorder_experts`（`:62`）；MoE 生产模块
  `ObfuscatedMoE`（`src/fastprove/layers/moe.py:218`）。

它与 `attention.compute_topk_budget` 的差异是登记在案的语义差异，不得统一。

## 10. 数值策略与验收边界

- 离线变换生成、求逆、条件数检查与 SVD 基：CPU FP64；
- 部署存储 dtype：FP32 或 FP64（`basis_common.validate_basis_dtype`）；
- RMS 统计/\(\rho\)、logits、mask、margin、噪声裁剪、Softmax 归约：FP32
  （`policy.REDUCTION_COMPUTE_DTYPE`，`src/fastprove/policy.py:55`）；
- `forward` 不调用 `inverse`/`inv`/`solve`（部署权重离线融合；测试
  `tests/deployment/test_no_decode_in_forward.py` 的运行时哨兵覆盖）；
- 所有路径统计 NaN/Inf，出现非有限值即失败；
- 不允许为通过测试而修改上述数学定义。

当前 FP32 小张量回归测试冻结的容差（测试路径已按 P7 重组）：

| 检查点 | 最大允许误差或 `assert_close` 容差 | 测试位置 |
|---|---|---|
| ChainLinear signal max absolute error | \(10^{-5}\) | `tests/math/test_chain_linear.py:74` |
| exact QK score max absolute error | \(5\times10^{-5}\) | `tests/models/test_obfuscated_block.py:50` |
| exact Softmax probability max absolute error | \(5\times10^{-6}\) | `tests/models/test_obfuscated_block.py:51` |
| exact attention output | `atol=rtol=5e-5` | `tests/models/test_obfuscated_block.py:60` |
| exact 单块最终 signal | `atol=rtol=1e-4` | `tests/models/test_obfuscated_block.py:76` |
| Tiny LM exact logits | `atol=rtol=2e-4` | `tests/models/test_tiny_lm.py:55` |
| ChainLinear 部署融合等价 | `atol=1e-10` | `tests/layers/test_deployed_matches_chain_linear.py:56` |

这些是随机 Tiny/小张量正确性阈值，不是预训练模型或其他硬件的自动容差。

## 11. 实现与旧文档/规范差异（核对结果）

1. **双侧排列**：AGENTS.md 推荐 \(M=\Pi DB\)，实现是
   \(M=\Pi_1DB\Pi_2\)（`src/fastprove/structured.py:1`、`:153`–`:162`）。
   排列正交，条件数不变；本文按实现描述。
2. **旧 `docs/mathematics.md` 已归档**：它描述 7 月构造，未提及
   `StructuredBasis`、部署权重、\(\ker(L)\)、双通道与词表置换；现行口径以本文为准。
3. **精确族中的 `c_V` 第二分量**：实现从完整混合状态 \(c\) 线性读出
   （`src/fastprove/models/block_builder.py:34`–`:51`），不是独立 \(e_V\) 通道；
   AGENTS.md 的 \(e'=hC+eG+\xi\) 框架与之相符，但旧文档的
   “`e_V` 独立”直觉不适用于当前部署构造。
4. **`forward` 不物化 `AV`/`O`**：旧数学文档第 10 节描述参考块“取出信号 \(AV\)”；
   当前生产路径直接消费 `mixed_context`（`src/fastprove/models/obfuscated.py:600`–`:606`），
   这是实现比旧描述更强的地方。
5. **condition number 记录值**：生成时记录的是
   \(\max D/\min D\)（`src/fastprove/structured.py:477`），与
   \(\kappa_2(M)\) 精确相等；`structured_condition_number` 用稠密 FP64 复核。
6. **sweep 基线继承**：数学上 exact 要求 ρ=0；当前 sweep 配置不满足
   （`docs/PROBLEMS.md` §3.5），本文与 `noise_budget.py` 的标签口径一致。
