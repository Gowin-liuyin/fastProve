# 九种混淆式 LLM 推理防御：机制、原理与工程落地

状态：CURRENT
日期：2026-09-09
上游依据：docs/literature-review/LITERATURE_INDEX.md
适用范围：九种混淆式 LLM 推理防御的机制与工程对照（文献调研记录）

日期：2026-09-09
配套：`2026-09-09_five_papers_inspiration.md`（DeepInvert 等五篇的启发分析）

DeepInvert 把这九种按**技术手段**分为四类（Table 2）：
VP = 词表置换、CM = 聚类/混合、TS = token 替换、NA = 加噪。

| 方案 | 类别 | 作用位置 | 客户端做什么 | 服务端看到什么 | 需重训/微调 | 支持生成 | 保证类型 |
|---|---|---|---|---|---|---|---|
| SentinelLMs (AAAI'24) | VP | 词表 + embedding 矩阵 | 本地分词 + 密钥加密 | 变换后的 token id 与 embedding | 需微调 | 否（作者列为 future work） | 无形式化保证（启发式） |
| ObfusLM (ACL'25) | VP+CM | 词表 + embedding + LM head | 本地分词 → 混淆 index | 混淆后的模型与 index | 需微调 | **是** | (k,ε)-anonymity + Laplace DP |
| Santext (ACL'21) | TS | token id | 本地替换 token | 替换后的 token id | 可不重训 | 部分 | metric-LDP |
| Custext (ACL'23) | TS | token id | 本地替换（定制候选集） | 替换后的 token id | 可不重训 | 部分 | ε-DP（指数机制） |
| TextObfuscator (ACL'23) | NA | 词表示 | 本地算表示 + 加噪 | 加噪的 clustered 表示 | **需重训**（原型聚类） | 否 | 声称 DP，但无归一化 → 无界 |
| DPNR (EMNLP'20) | NA | 句/词表示 | 本地 min-max 归一化 + 加噪 | 加噪表示 | 可不重训 | 部分 | 声称 DP（sensitivity 被证有误） |
| CAPE (EMNLP'21) | NA | 句表示 | 本地 L1 归一化 + 加噪 | 加噪表示 | 需对抗训练 | 否 | DP（修正后）+ 对抗不变性 |
| DP-Forward (CCS'23) | NA | embedding / 子层输出 | 本地 Frobenius 归一化 + 矩阵高斯噪声 | 加噪表示 | 需 noisy pre-train | 部分 | SeqLDP（后处理封闭性） |
| Datamix (ECCV'20) | CM | 中间特征 | 边缘跑前几层 + 私密系数混合 + 解混 | 混合后的特征 | 需 mixing-aware 训练 | 否（CV/语音） | 无非可逆性证明（启发式） |

---

## A 类：词表置换与参数混淆 —— 「改编码本，不改语义」

### A1. SentinelLMs（AAAI 2024，Mishra, Li, Deo）

**做法**（Algorithm 1 `AdaptLM`，一次性适配）：
1. 对 embedding 矩阵做 $n_{glide}$ 次 **glide reflection**：
   $$\mathbf e_{reflected} = \mathbf e - 2\frac{\mathbf e\cdot\mathbf l}{\mathbf l\cdot\mathbf l}\mathbf l,\qquad \mathbf e_{glide} = \mathbf e_{reflected} + \mathbf t$$
   其中 $\mathbf l$（反射轴）与 $\mathbf t$（平移）为随机向量，每轮重采。
2. 用 **Blake2b** 按用户 passkey 加密词表（同一 token 在不同 passkey 下得到不同密文）。
3. 随机置换词表与 embedding 行索引（排除 padding / CLS / SEP 等特殊 token）。

**原理**：reflection 与 translation 都是等距变换，因此 **Minkowski 空间中任意两 embedding 的距离保持不变** → 语义关系不变 → 模型能力保留；但绝对位置被彻底搬走 → 反推困难。

**落地**：一次适配（不需要重新预训练）→ 用**同一 passkey** 加密的任务数据微调 → 推理时用户本地分词+加密后上传。多租户成本极低（每个客户端不同 passkey、不同 glide 次数，都是确定性变换）。

**数字**：GLUE 上与原始模型差距 2% 以内（CoLA: BERT 56.53→55.3；RoBERTa 67.8→59.05 @ nglide=10）。直接 embedding 恢复成功率 0.009%。

**致命弱点**：**保距正是它的软肋**。DeepInvert 里连 KNN 都能拿到 0.91+ 的 top-1，InvBert/DeepInvert 接近 1.0。距离不变 ⇒ 公开词表空间中重新对齐即可破解。

### A2. ObfusLM（ACL 2025，南京大学 + 字节；与 covariant obfuscation 同组）

**做法**（Model Obfuscation，混淆的是**模型组件**而非输入）：
1. 客户端从预训练 LLM 取词表 $V$、embedding 矩阵 $E$、LM head $H$。
2. 生成随机置换 $\sigma$，按同一顺序打乱 $V$、$E$、$H$ 的行；$\sigma(V)$ 永远留在本地。
3. **EmbedCluster**：贪心地把 embedding 按余弦相似度划分为若干簇 $\mathcal E_i$，$|\mathcal E_i|\ge k$，相似度阈值 $\gamma$；由余弦相似度的三角不等式给出簇内两两相似度下界 $\min\sigma_{j,k}\ge 2\gamma^2-1$。孤立点放宽 $\gamma$ 以满足 k-匿名。
4. **WeightSynth**：簇内按相似度算合成权重 $\mathbf u$，加 Laplace 噪声 $\mathbf z\sim\mathrm{Lap}(\Delta u/\epsilon)$，其中 $\Delta u=\max(\mathbf u)-\min(\mathbf u)$；生成代表向量 $\mathbf e_{cluster}=\sum w_i\mathbf e_i + noise$，并用同一套权重混淆 $\sigma(E)$ 与 $\sigma(H)$，发给服务端。

**安全定义 $(k,\epsilon)$-anonymity**：对任意 $\mathbf e\in E$，存在 $|\mathcal S|\ge k$ 的子集，使得对任意 $\mathbf e'\in\mathcal S$ 与任意输出集 $O$：
$$\Pr[P(\mathbf e)\in O]\le e^{\epsilon}\Pr[P(\mathbf e')\in O]$$

**原理**：混淆后一个词在其所属子集内与至少 $k-1$ 个近义词不可区分（k 维），且加噪使区分概率被 $e^\epsilon$ 夹住（ε 维）。$\epsilon=0$ 时须在全词表上混淆（$k=|V|$）；$\epsilon$ 极大时无需混淆。

**落地**：客户端本地用打乱词表分词 → 上传混淆 index → 服务端用混淆模型微调/推理 → 生成任务返回"密文"index 序列，客户端本地解码。**是九种里唯一完整支持生成任务的**。ObfusLM+ 进一步对 embedding 的**语义维度**聚类，防中间层泄漏语义（需同步调整模型参数）。

**数字**：效用比 SANTEXT+ 高约 10%，与非隐私基线差距 4% 以内；对 KNN 类反演约 80% 抵抗率。

**作者自己列的威胁**：
- Token Frequency Attack：对同一 token 是**确定性**混淆，频次统计仍可破（类比替换密码）。
- Substitution Deciphering Attack：破译出一个 token 后可用 LM 顺推其余。
- **嵌入替换攻击**：token 替换只改编码方式，语义信息在 attention 层已被提取成 K/V；若不用混淆后的 $H'$ 而用原始 $H$ 编码，攻击者理论上可拿明文。

---

## B 类：token 级替换 —— 「换词，不换结构」

### B1. Santext / SANTEXT+（ACL 2021 Findings，Yue et al.）

**做法**：metric-LDP + 指数机制。对每个 token $w$，按
$$\Pr[\tilde w = v\mid w]\propto \exp\left(-\frac{\epsilon\cdot d(w,v)}{2}\right)$$
在整个词表上采样替换，$d$ 为 embedding 空间的欧氏距离。客户端本地完成，逐 token 独立。

**原理**：语义越近的词替换概率越高 → 替换后句子仍"能用"；但真实 token 被模糊在指数分布的钟形里。

**落地**：最轻量的一类——纯客户端文本预处理，服务端模型完全不用改，适合"把文本发给第三方 API"的场景。

**弱点**：候选集是**整个词表**，总有不可忽略的概率抽到语义无关的词 → 效用损失大而无隐私收益。

### B2. Custext / CUSTEXT+（ACL 2023 Findings，Chen et al.，阿里 + 蒙特利尔）

**改进点**：
1. 指出 MLDP 要求距离满足三角不等式，**余弦相似度不满足** → 退回原始 $\epsilon$-DP + 指数机制，但兼容任意相似度度量。
2. **定制候选集**：每个 token 先取语义最相似的 $K$ 个（如 $K=20$）作为候选，只在候选内指数采样。
3. **CUSTEXT+**：跳过停用词（几乎不携带敏感信息，但对语法至关重要）。

**落地**：同 Santext，纯客户端。代码开源 `github.com/sai4july/CusText`。

**数字**：SST-2 / QNLI 上比此前 SOTA 提升 20%+；MedSTS 上 Pearson 提升 50%。

**弱点**：DeepInvert 上提升幅度最小——因为替换只在局部破坏，反演模型可以靠句子级语义上下文补齐（InvBert 用 LM backbone 也能做到）。

---

## C 类：表示级加噪（Embedding-level DP）—— 「加噪，但要把 sensitivity 管住」

这一类四家的**唯一实质差别就是怎么归一化**（归一化决定 sensitivity，sensitivity 决定噪声量）：

| 方案 | 归一化 | sensitivity | 噪声 | 结果 |
|---|---|---|---|---|
| TextObfuscator | **无** | 无界 | Laplace | 无形式化保证 |
| DPNR | min-max → [0,1]，**逐坐标** | 声称 $\Delta f=1$ | Laplace | 被证错误，真实 $1\cdot n$ |
| CAPE | L1（$x_e \leftarrow x_e/\\|x_e\\|_1$） | 修正后 $\Delta f=2$ | $\mathrm{Lap}(2/\epsilon)$ | 向量级，保证成立 |
| DP-Forward | Frobenius（$\\|f(\cdot)\\|_F=C$ 或 $f/\max(1,\\|f\\|_F/C)$） | $S_2(f)=2C$ | 矩阵高斯（aMGM） | 向量/矩阵级，最严谨 |

### C1. TextObfuscator（ACL 2023 Findings，复旦 Zhou et al.）

**做法**（两步）：
1. **找任务相关原型**：token 级任务（如 NER）直接用标签做原型；句子级任务用 K-means + TF-IDF 重划分，确保 "excellent" 与 "terrible" 不被聚到一起。
2. **私有表示训练**：用 clustering loss 把功能相同的词（John / Mike）拉向同一原型；推理时对表示加 Laplace 噪声。

**原理**：**先聚再扰**。簇内已不可区分，加噪只需抹平簇内差异 → 噪声不必很大；且聚类本身起正则化作用 → 效用不降反升。

**落地**：需要在客户端侧重训（学原型），但**推理时零额外时延**。适合"客户端有算力做一次性训练、之后按次调用云 API"的场景。

**数字**：CoNLL2003 F1 **89.11**（非隐私 91.72，DPNR 79.14，CAPE 84.47）；SST-2 top-1 恢复从 100% 降到 5.48%。

**弱点**：不做归一化 → sensitivity 无界 → **没有形式化 DP 保证**。DeepInvert 明确指出它在 $\epsilon\ge 3$ 时 ASR > 0.9 是"自身误标定"而非攻击突破。

### C2. DPNR（EMNLP 2020 Findings，Lyu, He, Li）

**做法**：最早的 embedding-level DP。把表示 min-max 归一化到 [0,1]，逐坐标加 Laplace 噪声，声称 sensitivity $\Delta f=1$。

**原理**：归一化把输出范围固定住，才能算出有限的 sensitivity，才能标定 DP 噪声。

**弱点**：Maheshwari et al. (2022) 指出其算法有**根本性错误**，真实 sensitivity 是 $1\cdot n$（$n$ 为维度）→ 噪声被严重低估 → 实际保护远弱于声称。DeepInvert 评其为"坐标级而非向量级"保护，$\epsilon$ 到 10–15 时基本失效。

### C3. CAPE（EMNLP 2021，Plant, Gkatzia, Giuffrida，爱丁堡）

**做法**（两层"belt and suspenders"）：
1. **DP 层**：$x_e\leftarrow x_e/\|x_e\|_1$，加 $\tilde x_e = x_e + \mathrm{Lap}(\Delta f/\epsilon)$，采用修正后的 $\Delta f=2$（而非 Lyu et al. 的错误值）。
2. **对抗层**：加一个 attacker classifier 预测私有属性（性别 / 位置 / 年龄），用**梯度反转层 GRL** 让特征提取器对私有属性不变：
   $$\mathcal L = -\log P(y\mid \tilde x_e;\theta_r,\theta_p) - \lambda\log P(\neg z\mid \tilde x_e;\theta_a)$$

**原理**：DP 提供"普适地板"（防任意未知推断），对抗训练提供"针对性清除"（对已知敏感属性做到不变性）。二者互补。

**落地**：需要在训练管线里同时跑两个分类头，属于**训练期**方案；客户端侧只需归一化+加噪。适合"模型方自己就是数据控制方、要发布一个不带私有属性的模型"的场景（与其余八种的"客户端防云"设定其实不同）。

**数字**：Trustpilot，$\epsilon=0.1$：target F1 0.637（非隐私 0.772），attacker 性别 F1 0.751（base 0.762）。

### C4. DP-Forward（CCS 2023，Du, Yue, Chow, Wang, Huang, Sun；CUHK + OSU + UVA）

**做法**（范式级改变）：**把噪声加在前向传播，而不是反向梯度**。
- 模型切成 $f(\cdot)$（加噪前）与 $p(\cdot)$（加噪后）。
- 关键洞察：一旦 $f(X)$ 的输出满足 LDP，其后**所有计算都是该输出的后处理** → 由 DP 的后处理封闭性，fine-tuning 的梯度与参数更新自动满足 DP，无需 DP-SGD 的 per-example 梯度裁剪。
- 提出 **SeqLDP**（sequence-level local DP）。
- **aMGM**（analytic matrix Gaussian mechanism）：从矩阵高斯 $\mathcal{MN}_{n,d}(0,\Sigma,\Psi)$ 采样**非 i.i.d.** 噪声，$Z = UZ'V^\top$（$\Sigma=UU^\top,\Psi=VV^\top$），只需按 $U,V$ 的最小奇异值校准 → 噪声规模不随维度爆炸（破解维度灾难）。
- 归一化：Frobenius 归一到 $\|\cdot\|_F=C$，$S_2(f)=2C$。
- 注入位置可选：输入 embedding 后、MHA 后、FFN 后。**注意**：在第一个 MHA 输出加噪时必须**去掉残差连接**，否则 $p(\cdot)$ 会重新拿到未加噪的 $X$，破坏后处理的免费性（需相应重训一个无该残差的 BERT）。
- 缓解维度灾难的备选方案：插入两个线性映射 $M_1,M_2\in\mathbb R^{d\times d'}, d'\ll d$。
- **noisy pre-training**：用公开语料（Wikipedia）以 MLM 目标做带噪预训练，让模型先适应扰动。

**落地**：客户端本地归一化+加噪后上传；服务端拿到的是 LDP 保证下的表示，可正常 fine-tune 与推理。因为摆脱了 per-example 梯度，**可用标准 batch 训练**。

**数字**：比 DP-SGD 高最多 7.7pp；时间/显存省 3×；embedding inversion 成功率降最多 88pp，敏感属性推断降 41pp（DP-SGD 对这些推理期攻击**零防护**）。

---

## D 类：混合 / 复用 —— 「用线性组合藏起来」

### Datamix（ECCV 2020，MIT + SJTU + MIT-IBM，Liu, Wu, Gan, Zhu, Han）

**做法**：把 DNN 切成三段 —— preprocess（边缘）/ main（云）/ postprocess（边缘）。
1. 边缘跑前几层提取特征。
2. **mixing**：用**私密随机系数**（从正交矩阵群采样）把 $m$ 个输入（1 个真实 + 若干 dummy）线性混合。
3. 云在混合特征上跑主体计算，返回混合结果。
4. 边缘用私钥 **de-mixing**（一次线性代数操作）后跑后处理层。
5. **mixing-aware training**：训练时按同样方式混合（系数从正交矩阵群采样），三段端到端一起更新。
6. **inter-group shuffling**：跨组打乱混合后的中间输入，云无法把同组数据关联。

**原理（非可逆性论证，值得细读）**：攻击者可能想到"对包含同一张图的大量混合体求平均"来还原。但那是训练场景（一张图出现在很多混合里）；**推理场景下图只与固定的一组图混合，求平均后仍是系数未知的混合**。没有系数就不可逆。

**落地**：边缘只需跑前几层 + 混合/解混（极轻），云端跑主体且**100% GPU 利用率**（全是标准算术，不像 HE/MPC）。原为 CV/语音设计。

**数字**：边缘计算降到总 FLOPs 的 1–13%（论文称 <20%）；CelebA 属性分类 ~91%（与明文云推理几乎一致）；身份识别攻击成功率从 85.5% 降到 **0.6%**。

**局限**：需要 mixing-aware 重训；原论文是 CV/语音，DeepInvert 把它列入但指出其架构与标准混淆管线不完全兼容（同组的 TextMixer 也因需要特殊预训练模型而被排除在实验外）。

---

## 工程落地的关键分野

1. **谁承担改动成本**：Santext / Custext / DPNR 是"纯客户端文本预处理"，服务端零改动，最容易落地；ObfusLM / TextObfuscator / Datamix 都需要重训或微调，是一次性成本换长期效用。
2. **能否支持生成**：只有 ObfusLM 完整支持（因为连 LM head 一起混淆）。其余八种基本只覆盖分类/序列标注 —— 这恰恰是 DeepInvert 说的"任务相关张力"的根源。
3. **是否有形式化保证**：只有 ObfusLM（(k,ε)-anonymity）、Custext（ε-DP）、CAPE（DP）、DP-Forward（SeqLDP）真正站得住；SentinelLMs、Datamix 是启发式（无非可逆性证明）；TextObfuscator、DPNR 声称 DP 但 sensitivity 处理有误/无界。
4. **确定性的陷阱**：ObfusLM 对同一 token 是确定性混淆（作者自列 Token Frequency Attack 为威胁）；SentinelLMs 也是确定性变换。因此"每次请求刷新随机性"（SCX 的 one-time key、我们的 per_request ξ）是必需项而非可选项。
5. **保距是双刃剑**：SentinelLMs 靠保距保住效用，也正因保距被 KNN 一击即溃。任何"保持几何结构不变"的混淆都必须额外引入破坏对齐的随机性。

---

## 对 fastProve 的定位启示

- **我们的位置**：不属于 A/B/C/D 任何一类。A/B 改的是**离散编码本**（词表、token id），C 改的是**表示本身但会损精度**，D 靠**多输入线性组合**。我们是第 5 种：**保持计算代数结构不变、用密钥化的可逆线性变换 + 受控噪声做协变传递**——既不像 C 那样牺牲精度（exact 模式零损失），也不像 A 那样只改编码本。
- **必须避开的三个坑**（九种里已经有人踩过）：
  1. **sensitivity 无界**（TextObfuscator、DPNR）→ 我们的噪声标定必须显式给出归一化与 sensitivity。
  2. **确定性混淆**（ObfusLM、SentinelLMs）→ 必须 per-request 刷新 ξ，且要说明"同一 prompt 重复提交可平均掉噪声"这一残余风险。
  3. **保距即破绽**（SentinelLMs）→ 我们的 $M$ 若取近正交/近等距，需论证为何不会退化为可对齐的情形。
- **可借的论证**：DP-Forward 的「后处理封闭性」是我们 exact 模式最好的理论包装——一旦 checkpoint 处的表示满足某个隐私性质，其后所有计算免费继承。
