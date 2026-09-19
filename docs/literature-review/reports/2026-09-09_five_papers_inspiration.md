# 五篇论文的启发分析（fastProve 视角）

状态：CURRENT
日期：2026-09-09
上游依据：docs/literature-review/LITERATURE_INDEX.md
适用范围：五篇论文对 fastProve 的启发分析（文献调研记录）

日期：2026-09-09
对象：covariant obfuscation of Transformer inference 原型

## 0. 速览

| # | 论文 | 类型 | 与 fastProve 的关系 | 优先级 |
|---|------|------|--------------------|--------|
| 1 | DeepInvert (arXiv 2608.04477, 蚂蚁集团) | **攻击** | 最直接威胁，打的就是「混淆」；且引用了同路线的 covariant obfuscation 技术报告 arXiv 2603.01499 | **P0** |
| 2 | SCX (SIGCOMM'25, CUHK/USTC) | 防御系统 | 同赛道最强工程参照；通信复杂度论证、一次性密钥、三类攻击清单 | **P0** |
| 3 | Privacy-Aware Split Inference (arXiv 2602.16760) | 防御系统 | 部署参照系 + 深度-隐私基线曲线 (58.8%→34.8%) + SipIt 单射性动机 | P1 |
| 4 | STAMP (EACL'26, Arizona/ORNL/Duke) | LDP 文本隐私 | 机制几何学：扰动几何须匹配解码几何；分组预算 | P1 |
| 5 | Frequency-Smoothing Encryption (CCSW'17, RHUL) | 密码学原语 | 提供「可证明优势界 + 统计最优攻击者 + 实证」范式；同音编码 ≈ 连续版增广噪声 | P2（但方法论价值最高） |

---

## 1. DeepInvert: Semi-Supervised Embedding Inversion Against Obfuscated LMs

**元信息**：Zhicong Huang, Cheng Hong, Tao Wei（Ant Group），arXiv:2608.04477v1，2026-08。

### 1.1 做了什么

攻击九种「混淆类」LM 防御（ObfusLM、SentinelLMs、Santext、Custext、TextObfuscator、DPNR、CAPE、DP-Forward、Datamix）。核心论点：**这些防御提供的保护远少于此前认为的**。

先前最强的 InvBert 是纯监督的：在 shadow data 上训练反演模型，因 shadow ↔ real 的分布偏移而崩。DeepInvert 改为半监督：

- **Soft top-k labeling**（Eq.4）：不是用高置信硬伪标签，而是在 top-k 概率子集上算 KL：
  \( \mathcal L_u^{(k)} = \frac1L \sum_i D_{KL}\big(f_s(x^s)_{i,I^{(k)}} \,\|\, f_t(x^w)_{i,I^{(k)}}\big) \)
  动机（Fig.3）：top-1 不准但 top-100 已经不错，软标签比硬伪标签信息量大得多。
- **Embedding-level masked augmentation**（Eq.6-7）：随机把若干位置的 embedding 换成 `[MASK]` embedding，构造弱/强两个视图（不是对 token 做，是对 embedding 做）。
- **EMA teacher**（Eq.8-9）：\( \theta_t \leftarrow \alpha\theta_t + (1-\alpha)\theta_s \)，避免自蒸馏崩溃。
- **Mixed training 交替调度**（Eq.10）：\( \lambda_i = \lambda_0 + \frac{i}{n-1}(\lambda_{n-1}-\lambda_0) \)，每个 epoch 内先监督后无监督交替——因为联合优化存在梯度冲突（多任务学习里的老问题），固定权重 λ 会停在次优点。
- **Multi-shadow**：用多个不同随机性的 shadow model 产生 labeled 数据，逼模型学「跨随机性不变」的语义特征，缓解分布偏移。
- **Renormalization + PCA 降噪**（Lemma 1）：先 LayerNorm 统一尺度；再用大量 clean shadow embedding 估 PCA 矩阵 \(P\in\mathbb R^{H\times K}\)，做 \(PP^T x\)。
  \( g\sim\mathcal N(0,\sigma^2 I_H), z=PP^Tg \Rightarrow \mathbb E\|z\|^2 = \frac KH \mathbb E\|g\|^2 \)
  而信号侧能量保留比约为 \( \sum_{i\le K}\lambda_i / \sum_i \lambda_i \approx 1 \)（embedding 空间低秩）。于是 SNR 提升约 \(H/K\) 倍。
  作者明确声明：**这不是攻破 DP，只是说明 LM 混淆场景下 ε 必须取得更严**。

### 1.2 关键数字

- 9 防御 × 5 任务 × 4 模型（BERT-Base / RoBERTa-Base / LLaMA3-8B / Qwen3.5-27B）。
- ObfusLM (ε=0.1) + RoBERTa + SST-2：top-1 **0.735**（InvBert 0.262，TBS 0.164）。
- AG News：0.818 vs 0.267；CoNLL-2003：0.644 vs 0.358。
- Qwen3.5-27B Medical QA，ObfusLM (ε=1)：**0.852**。
- 无标注样本量：0 个时退化为纯监督 0.344；**仅 50 个**就到 0.402；随池子单调上升（transductive 设定）。
- Sensitive Top-1（GPT 标注敏感 token）：与普通 top-1 几乎一致 → 敏感信息并不更难恢复。
- CAPE / DP-Forward 在低 ε 下较抗打，但代价是 CoNLL-2003 上 **F1 = 0**（效用归零）。

### 1.3 它的核心论断（最值得引用）

> 在 token 级与生成级任务上，**保留足够效用的混淆必然保留足够可反演的结构**；能抗住攻击的方案靠的是把效用一起毁掉。这是结构性张力，不是某个方案的缺陷。在句子级分类上，部分 DP 防御可以兼得。

### 1.4 局限（它自己承认）

- transductive 假设：攻击者能拿到目标部署产生的大批无标注混淆 embedding（虽有 ablation 显示能迁移到 held-out prompt）。
- shadow 与真实 defense 的分布 mismatch（聚类数、DP 变体等）留作 future work。
- DP 噪声大到 InvBert 失效时，DeepInvert 的 teacher 也会不可靠，优势消失。

### 1.5 对 fastProve 的启发

**(a) 最直接的软肋：增广噪声是低维子空间噪声，PCA 几乎必然能剥离。**
DeepInvert 的 Lemma 1 本质是「各向同性零均值噪声 + 低秩信号 ⇒ 保留 top-K 主成分即可提 SNR」。而 fastProve 的结构恰恰是 \(c=[h,e]M\)，噪声只占 \(r\ll d\) 维子空间，信号占 \(d\) 维。攻击者只要能收集足够多 unlabeled 的 \(c\)，做一个 PCA/ICA 式子空间分离就能把 \(h\) 分量捞出来——甚至不需要知道 \(M\)（只需估计信号主子空间）。**这条必须立刻做实验验证**，因为它可能直接决定 approximate 模式的安全上限。

**(b) 把 DeepInvert 改造成 fastProve 的自适应攻击基线。**
你的威胁模型（honest-but-curious、可观测持久化张量）恰好满足它的前提：攻击者能拿到大量无标注混淆态。建议作为攻击强度上限写进 `threat_model.md` 与 `docs/archive/2026-09-19_experiment_plan.md`。改造点很小：把「拿第 l 层 hidden state 预测 token id」换成「拿 checkpoint 处的 \(c\) 预测 token id / logits」。

**(c) 评估指标体系照抄它。**
top-1 token recovery（主指标）+ Sensitive Top-1（用 LLM 标注敏感 token，作为「是否真的保护了要保护的东西」的校验）+ ROUGE-L/BLEU（语义保真度）。并且**必须同时报告效用崩塌点**：抗攻击时的任务精度是否还非零。这正是审稿人会追问的对照实验，也是它整篇文章的立论方式。

**(d) 必须划清的界限。**
DeepInvert 明确说「我们没有攻破 DP，只是说明 ε 要更严」。fastProve 的 approximate 模式同样**不能声称 DP**，只能声称「有界 logit 扰动 + 表示混淆」，否则会被它那套论据直接击穿。

**(e) 一个需要立刻确认的事实。**
它的参考文献 [53] 是：
> Y. Lin, Q. Zhang, W. Ruan, D. Zhang, J. Hong, Y. Wu, H. Xia, Y. Mao, S. Zhong, "Towards privacy-preserving LLM inference via covariant obfuscation (technical report)", CoRR abs/**2603.01499**, 2026.

（Y. Lin / S. Zhong 即 ObfusLM 作者群。）这条既是「近邻工作必须正式定位差异」，也是预警：DeepInvert 作者已知 covariant obfuscation 这条路线，下一版很可能直接来打。

---

## 2. SCX: Stateless KV-Cache Encoding for Cloud-Scale Confidential Transformer Serving

**元信息**：Mu Yuan, Lan Zhang, Liekang Zeng, Siyang Jiang, Bufang Yang, Di Duan, Guoliang Xing（CUHK + USTC），SIGCOMM'25，DOI 10.1145/3718958.3750509。

### 2.1 做了什么

提出「无状态 KV-cache 编码」：用**用户侧控制的密钥**把云上的 KV-cache 锁住，使云既不能恢复输入，也不能独立完成下一 token 预测。

三类变换：
- **Token / feature 级置换** \(P_t, P_f\)
- **冗余嵌入**（redundant embedding）：编码时拼接冗余分量，用户侧用索引 `[:d]` 减除恢复
- **Laplace 噪声** \(\eta \sim \mathrm{Lap}(0,\beta)\)，**只在第一个 block 加**（因为第一块 input embedding 还没有 token 间相关性，最容易被穷举匹配词表；后续块已有 attention 混合，穷举复杂度指数爆炸）

两个关键 insight：

1. **通信复杂度**：层层级编码是 \(O(t\cdot L)\)（t 为生成 token 数，L 为层数）。观察：Transformer 前向是顺序的，只要**首尾块被锁住**，云就无法独立完成解码 → 中间块的密钥可以上传给云。复杂度降到 \(O(t+L)\)。实验取 \(l=1\) 或 \(2\)、\(r=1\) 即足够。
2. **攻击脆弱性区分阶段**：固定密钥会被**拟合式反演**（训练 surrogate 逼近编码机制）打穿。因此 **prefill 用固定密钥**（抗穷举 + LLM 辅助恢复），**generation 每步用 one-time 密钥**（抗拟合）。

安全性：给出 DP 保证；且 Lemma 1 说明**噪声在用户侧解码时被等价消除**，因此**零推理质量损失**（数学等价有证明）。

### 2.2 关键数字

- Llama-7B TTFT：Puma（HE+MPC）**200s**，full-TEE **17s**，SCX **36ms**（明文 10ms）。
- 与 KV-cache 压缩技术叠加可再降 85% 内存搬运延迟。
- 在 7/13/47/70B 四个规模上验证。
- 抗三类攻击：穷举词表匹配、LLM 辅助序列恢复、拟合式反演。

### 2.3 对 fastProve 的启发

**(a) 冗余嵌入与增广态同构，但底座不同 —— 这个差别要写进论文。**
SCX 的「加冗余分量 → 用户侧减除」与你的 \([h,e]\) 增广在形式上是同一族技巧。区别在于：SCX 靠**用户侧参与解码**消除噪声，你靠**代数协变**让噪声随计算自动传递。它的 Lemma 1（噪声等价消除 ⇒ 零精度损失）可以作为你 exact 模式零损失主张的旁证引用，但必须点明机制不同，否则会被认为是同一件事的复述。

**(b) 「只锁首尾」的论证模板可用来省 checkpoint。**
SCX 的立场是：安全目标只需保证「云无法独立完成预测」，不必每一层都保密。这对 fastProve 是一个可借的论证框架——是否真的每层非线性都要 checkpoint？能否论证「只要首/尾断点保密，中间层由协变性自动覆盖」？若能，开销可显著下降。**这是一个值得认真推的方向**。

**(c) one-time key ≈ 你的 per_request 噪声刷新 ξ。**
SCX 给出的理由（固定密钥 → 拟合式反演可训练 surrogate）正是 AGENTS.md 里「不要把 deterministic fixed refresh 当成强隐私」的外部佐证，可直接引用。

**(d) 攻击清单可直接挪用为三条攻击基线。**
穷举词表匹配 / LLM 辅助恢复 / 拟合式反演（surrogate training）——这三条比 KNN 更有说服力，建议直接进 `threat_model.md`。

**(e) 划清界限。**
SCX 需要用户侧 TEE + 每步交互；你不需要，这是优势。但反过来，SCX 能证明「云不能独立完成预测」，而 fastProve 目前只能说「云看到的是混淆表示」。这是 claims 的硬边界，不能越。

---

## 3. Privacy-Aware Split Inference with Speculative Decoding over WANs

**元信息**：Michael Cunningham，arXiv:2602.16760v1，2026-02。（偏工程报告，非顶会论文，引用需谨慎。）

### 3.1 做了什么

- **非对称切分**：embedding 与 unembedding 留本地，中间层上云 → 原始 token 永不出本地。
- **首次把 lookahead decoding 用于 WAN split inference**，用 n-gram 投机把一次 RTT 摊到多个 token。
- **KV-cache 放云侧**（避免随序列长度线性增长的传输），支持 cache cropping / relocation / 自定义 attention mask。
- WebSocket 二进制传输（float16 裸字节 + JSON header）。

### 3.2 关键数字

- Mistral-7B @~80ms RTT：8.7–9.3 tok/s；Mistral NeMo 12B：7.8–8.7 tok/s。
- lookahead 平均接受 1.2–1.3 tokens/step（峰值 7，代码类）。
- 本地 VRAM：2.0 GB (7B) / 4.9 GB (12B)。
- **延迟分解**（顺序模式）：网络 RTT 63–64%，本地 GPU 20–22%，云 GPU 13–17%，序列化 1%。
- **反转攻击**（3 层 MLP 4096→2048→2048→32000，约 880 样本）：

| 切分位置 | 本地层数 | top-1 | top-5 |
|---|---|---|---|
| layer 1 | 2 | **58.8%** | 62.0% |
| layer 3 | 4 | 44.3% | 49.8% |
| layer 5 | 6 | 44.8% | 56.6% |
| layer 7 | 8 | **34.8%** | 46.2% |
| 随机基线 | — | 0.003% | 0.016% |

### 3.3 它的诚实之处（值得学）

- 明引 **SipIt**（Nikolaou et al.）：token 序列到 transformer hidden states 的映射是**单射**的，在有权重的前提下可线性时间精确重建输入 → 中间激活原则上可逆。
- 明确说自己的保证是 **architectural privacy**，不是 cryptographic，也不是 hardware-attested；刻意回避 "computational privacy" 这个词。
- 关键假设：本地层必须被微调/改动（LoRA 等），否则云可以下载同款权重，边界就垮了。并承认「随机 LoRA 扰动可能对自适应攻击者无效」，未验证。

### 3.4 对 fastProve 的启发

**(a) 深度-隐私曲线是你的天然对照组。**
58.8% → 34.8% 这条曲线说明「加深本地层」这条路的代价与收益。fastProve 的卖点应当是：**在同样（很浅）的本地层数下，把 recovery 压到远低于 34.8%**。这个对比数字最有说服力，建议直接放进主实验表。

**(b) SipIt 单射性是你 intro 里最好的 motivation，也是一把双刃剑。**
正面：纯 split 不够，必须改变表示本身 —— 这正是 covariant obfuscation 的立足点。
反面：**单射性对你的 exact 模式同样成立**。若 \(M\) 泄露且攻击者知道权重，\(c=zM \Rightarrow z \Rightarrow h\)（取前 d 维）是完全可逆的。因此：

> fastProve 的安全性来自 **密钥（\(M\) 与 \(\xi\) 的秘密性）**，而不是来自「计算不可逆」。

这一点必须在 threat model 中用一句话挑明，否则审稿人很容易用 SipIt 一句话把你打死。

**(c) 延迟分解方法论照搬。**
\(t_{\text{step}} = \mathrm{RTT} + \text{fixed\_overhead}\)，交叉验证误差 1.2–6.2%。你的开销评估应同样把「混淆引入的额外开销」从通信/计算里剥离出来单独报，而不是只报端到端倍数。

**(d) 「本地层必须差异化」可类比论证 \(M\) 的密钥地位。**
它要求本地权重与公开权重不同；你要求 \(M\) 保密。两者是同一类「秘密参数」论证，可以互相参照。

---

## 4. STAMP: Selective Task-Aware Mechanism for Text Privacy

**元信息**：Fengwei Tian, Payel Bhattacharjee, Heidi Hanson, Geoffrey D. Rubin, Joseph Y. Lo, Ravi Tandon（Arizona / ORNL / Duke），EACL 2026 long paper，arXiv:2603.12237。

### 4.1 做了什么

三件事：

1. **选择性预算分配**：按 (任务重要性 × 隐私敏感度) 打 2×2 网格分 4 组，每组不同 ε。任务重要性 = \(\cos(\text{token}, \text{task/query repr}) \ge \tau\)（τ 扫参，0.5 处稳定）；敏感度用 NER/PII 规则（Person / Location / Organization / Numeric ID）。多 token 实体按 span 整段同组。
2. **Polar mechanism**：把 embedding 拆成径向 \(r\) 与方向 \(u\)，**只扰动方向**（vMF 噪声，concentration \(\kappa\)），保留模长；归一化版把模长固定为 1，此时径向达到 \(0\)-metric LDP（模长本身常泄漏词频/显著性）。
3. **核心洞察 —— 扰动几何必须匹配解码几何**：解码用余弦最近邻，那么隐私度量就该定义在角度度量上。Proposition 1 证明：
   \(\arg\max_v \frac{e^\top v}{\|e\|\|v\|} = \arg\min_v \arccos(\hat e^\top \hat v) = \arg\min_v \|\hat e - \hat v\|_2\)
   即余弦解码 ≡ 球面上的 geodesic / chordal 最近邻。

安全性建立在 **metric LDP**（Definition 2，按 token 间距离缩放不可区分性）之上，Theorem 1–3 分别给出分组级、上下文级（组合）以及 vMF 机制的保证。

### 4.2 关键结果

SQuAD / Yelp / AG News，matched ε 下 Polar 一致优于各向同性 Laplace：**低预算时 Laplace 塌到 chance 水平，Polar 仍有效**。

### 4.3 对 fastProve 的启发

**(a) 「扰动几何要匹配度量/解码几何」—— 直接可用于 Softmax 噪声设计（最实在的一条）。**
你现在给 logits 加扰动，隐含的是各向同性欧氏度量。但 Softmax 实际只敏感**行内相对差值**（加行内常数不变），而你的 exact 可交换族正是「列置换 + 行内公共偏移」。因此扰动应该定义在 **logit 的差分商空间**（减去行均值后所在的空间）上，而不是原始 logit 空间。预期收益：同等保护强度下精度损失更小，或同等精度损失下排序更难被反解。**这是一个可以直接做消融实验的改进点。**

**(b) 分组预算 → 按位置分配 ξ。**
你的噪声刷新 \(\xi\) 可按 token/位置重要性分配（attention sink、首 token、敏感实体位置给更大 \(\xi\)）。注意边界：server 要算完整前向，不能像 STAMP 那样把 token 替换掉。

**(c) vMF 方向扰动 vs 增广高斯噪声 —— 与 RMSNorm 断点的潜在契合。**
你的 RMSNorm 断点依赖「正交 \(R\) 下 \(\mathrm{rms}\) 保持不变」。若噪声改为**只扰动方向、保留模长**，则与 RMSNorm 的交换性可能比各向同性噪声更好。这是把 polar 思想嫁接到噪声刷新上的一条可能路径，值得记一笔做小实验。

**(d) 归属性提醒。**
metric-LDP 框架可借来形式化「混淆后表示的不可区分性」，但你的场景不是「发布数据」，威胁模型也不是 LDP。**不要顺手声称 LDP**，那会与 AGENTS.md 的禁令冲突。

---

## 5. Frequency-Smoothing Encryption (FSE)

**元信息**：Marie-Sarah Lacharité & Kenneth G. Paterson（Royal Holloway），ACM CCSW 2017（ACM DL 2017-1068）。

### 5.1 做了什么

用**同音编码**（homophonic encoding）对抗频率分析攻击。把确定性加密（DE）推广为 "somewhat randomised" 加密：每个明文按其在分布中的频率分配多个同音密文，使密文频次分布**平坦化**。

- 两个安全概念：**FSE-SMOOTH**（密文集合与均匀不可区分）+ **privacy**（Rogaway-Shrimpton DE 安全性的推广）。
- 模块化构造：\( \mathrm{FSE} = \mathrm{DE} \circ \mathrm{HE} \)。HE 的平坦性给 frequency-smoothing，DE 的隐私给 privacy。
- 两个 HE 方案：**IBHE**（按分布把 \(r\)-bit 串切区间，消息 \(m\) 分到约 \(f_D(m)\cdot 2^r\) 个串）与一个基于随机误差的变体；均可通过调 \(r\) 权衡查询效率与抗频率分析能力。
- 用 **KL 散度 + BJV04 最优区分器框架**给出优势界：\(\mathrm{Adv} \approx \sqrt{N\cdot \mathrm{KL}/2}\)，IBHE 的 \(\mathrm{KL} \lesssim 1/(2h^2)\)（\(h\) 为每个消息的最少同音数）。数值例：\(f_D(m_1)=2^{-5}, N=2^{10}\)，目标优势 \(2^{-10}\) → 需 \(h\approx 2^{12.7}\)、\(r\ge 18\) bit。
- 还设计了 **MLE 最优攻击**（FSE 版的广义频率分析），在 Naveed 等用过的同一医疗数据集上评估：MLE 攻击者的成功率**很快降到纯猜测水平**。

### 5.2 它自己承认的局限

- 只抗 **snapshot attack**（静态快照）；不抗 query analysis（查询次数本身与频率成正比，需要 padding/batching）；不抗跨列相关性攻击。
- 承认 Grubbs et al. 对 snapshot 模型人为性的批评；但论证「抗 snapshot 是任何现实威胁模型的必要条件」。

### 5.3 对 fastProve 的启发

**(a) 最大价值是方法论范式：可证明优势界 + 统计最优攻击者 + 实证，三件套。**
你现在有 exact / approximate 双模式，但缺「最优攻击者」的刻画。FSE 的做法是：先在给定泄露下写出统计最优的攻击者（MLE），再给出其优势上界（KL + 最优区分器框架），最后在真实数据上实证。

对应到 fastProve：给定 logit 扰动分布，**写出最优的 logit 反解 / 排序恢复攻击者，给出其成功概率上界**。这比「跑一个 DeepInvert 看数字」更有说服力，两者互补（一个是经验下界，一个是理论上界）。

**(b) 最有转化价值的一条：你的方案本质上是「表示空间上的连续版同音编码」。**
FSE 平坦化的是密文频次分布；同一个 \(h\) 在你这里对应**多个可能的 \(c\)**（因为 \(\xi\) 与 \(M\) 随机）—— 这就是连续版的 homophonic encoding。

这个 framing 有两个好处：
1. 可以把 FSE 的 KL / 优势界技术搬过来，界定「给定 \(N\) 个观测，攻击者区分两个 \(h\) 的优势」；
2. 它天然回答了审稿人的经典追问：「你的混淆到底提供了多少不可区分性？」—— 你可以给一条 \((\sigma, r, N) \to\) 优势上界的曲线，而不是只给精度-噪声表。

**(c) 代价-安全三元权衡表照做。**
FSE 给 \((r, N, \text{Advantage})\) 表。你应给 \((\text{噪声幅度 }\sigma,\ \text{增广维数 } r,\ \text{观测样本数 } N) \to (\text{攻击优势},\ \text{精度损失})\) 表。

**(d) 划清界限 —— 它对 fastProve 的一记警钟。**
FSE 明确不抗 query analysis。你有完全对称的问题：**per-request \(\xi\) 只在单次请求内随机；如果同一 prompt 被重复提交（或攻击者能诱导重复），对 \(c\) 求平均即可抑制零均值噪声**。这必须写进威胁模型限制，与 AGENTS.md 里 `fixed_debug` vs `per_request` 的区分呼应。

---

## 6. 综合行动建议（按优先级）

**P0-1｜立刻验证最大软肋。**
针对「PCA/ICA 子空间分离可剥离低维增广噪声」（DeepInvert Lemma 1）做专门实验。若成立，approximate 模式的安全上限需要重新界定，可能要改噪声结构（见 P1-2 与 STAMP 的方向扰动思路）。

**P0-2｜确认 arXiv 2603.01499 的定位。**
在 related work 中正式区分 fastProve 与那篇 covariant obfuscation 技术报告的差异（可运行原型？Softmax 近似模式的误差界？系统实现？）。同时把它当作「下一版 DeepInvert 可能直接来打」的预警。

**P1-1｜补证明线：最优攻击者 + 优势上界（FSE 范式）。**
替代或补充纯经验攻击评估。经验攻击（DeepInvert 改造版）给下界，理论界给上界，两者夹逼。

**P1-2｜Softmax 噪声改在差分商空间上定义（STAMP 的几何匹配原则）。**
预计是投入产出比最高的单点改进，且消融实验很好做。

**P1-3｜开销报告采用延迟分解法（split inference 论文）。**
把混淆引入的额外开销从 RTT / 计算中剥离，单独报。

**P1-4｜攻击基线升级为三类（SCX）。**
穷举词表匹配、LLM 辅助恢复、拟合式反演（surrogate training），替代过于简单的 KNN。

**P2｜SipIt 单射性作为 motivation 引用，并明确安全来自密钥。**
一句话挑明：fastProve 的安全性来自 \(M\) 与 \(\xi\) 的秘密性，而非计算不可逆。
