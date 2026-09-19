# 2026-09-09

## 五篇论文启发分析（covariant obfuscation 相关文献）

精读并产出报告：`docs/literature-review/reports/2026-09-09_five_papers_inspiration.md`

五篇：DeepInvert (arXiv 2608.04477, 蚂蚁)、SCX (SIGCOMM'25)、Privacy-Aware Split Inference (arXiv 2602.16760)、STAMP (EACL'26, arXiv 2603.12237)、FSE (Lacharité & Paterson, CCSW'17)。

### 关键结论（对本项目有长期价值）

1. **最大软肋已被点名**：DeepInvert 的 PCA 降噪引理（Lemma 1：各向同性零均值噪声 + 低秩信号 → 保留 top-K 主成分即可提 SNR 约 H/K 倍）直接适用于我们 `c=[h,e]M` 的结构——噪声只占 r 维子空间，信号占 d 维。攻击者收集 unlabeled 混淆态做 PCA/ICA 子空间分离即可剥离噪声，甚至不需要知道 M。**必须做实验验证**。
2. **同路线近邻工作已存在**：arXiv **2603.01499** "Towards privacy-preserving LLM inference via covariant obfuscation (technical report)"（Y. Lin, Q. Zhang, W. Ruan, ... S. Zhong，即 ObfusLM 作者群）。DeepInvert 已引用它，下一版很可能直接来打 covariant obfuscation。related work 必须正式定位差异。
3. **安全性来源必须挑明**：SipIt（Nikolaou et al.）证明 token→hidden states 单射可精确反演。若 M 泄露且权重已知，`c=zM → z → h` 完全可逆。因此 fastProve 的安全来自 **M 与 ξ 的秘密性**，不是「计算不可逆」。
4. **可改进点（投入产出比最高）**：STAMP 的「扰动几何须匹配解码几何」→ Softmax 的 logit 扰动应定义在**行内差分商空间**（减行均值后的空间），而非原 logit 空间，因为 Softmax 只敏感行内相对差值，且我们的 exact 可交换族正是「列置换 + 行内公共偏移」。
5. **方法论缺口**：FSE 的「可证明优势界 + 统计最优 MLE 攻击者 + 真实数据实证」三件套可作为模板。我们应补一条 theoretically 的攻击优势上界（KL / BJV04 最优区分器框架），与经验攻击（DeepInvert 改造版）形成夹逼。 framing 上：同一个 h 对应多个 c（因 ξ、M 随机）≈ **表示空间上的连续版同音编码**。
6. **评估基线**：split inference 论文给出深度-隐私曲线（本地 2 层 58.8% → 8 层 34.8% top-1 恢复），是我们的天然对照组；卖点应为「同样浅的本地层数下压到远低于 34.8%」。DeepInvert 提供指标体系（top-1 + Sensitive Top-1 + ROUGE-L/BLEU + 效用崩塌点）。
7. **claims 边界**：approximate 模式不得声称 DP（DeepInvert 明确说它没攻破 DP，只说明 ε 要更严）；不得声称 SipIt 意义下的不可逆。

## 九种混淆防御机制梳理（DeepInvert 评测对象的原始细节）

产出：`docs/literature-review/reports/2026-09-09_nine_obfuscation_defenses.md`

四类：A 词表置换（SentinelLMs glide reflection 保距、ObfusLM (k,ε)-匿名+聚类合成）、B token 替换（Santext 全词表指数机制、Custext 定制 K 近邻候选集）、C 表示加噪（TextObfuscator 无归一化、DPNR min-max 逐坐标、CAPE L1、DP-Forward Frobenius+aMGM）、D 混合（Datamix 私密正交系数）。

### 对本项目有长期价值的几点
- **sensitivity 的标定是 C 类成败的分水岭**：TextObfuscator 无归一化→无保证；DPNR 的 Δf=1 被 Maheshwari et al. 2022 证伪（真实值 1·n）；CAPE 用修正后的 Δf=2；DP-Forward 用 Frobenius 归一 S₂(f)=2C。我们的噪声标定必须显式给出归一化与 sensitivity，否则重蹈覆辙。
- **确定性混淆必然被频次攻击打穿**：ObfusLM 作者自列 Token Frequency Attack 与 Substitution Deciphering Attack 为威胁。⇒ per-request 刷新 ξ 是必需项，且要写明「同一 prompt 重复提交可平均掉噪声」的残余风险。
- **保距是双刃剑**：SentinelLMs 靠保距保效用，也正因保距被 KNN 打到 0.91+。我们的 M 若近正交/近等距，需论证为何不退化为可对齐。
- **DP-Forward 的「后处理封闭性」是 exact 模式最好的理论包装**：一旦 checkpoint 处表示满足某隐私性质，其后所有计算免费继承。注意它要求在第一个 MHA 输出加噪时**去掉残差连接**，否则 p(·) 会重新拿到未加噪的 X。
- 九种里只有 ObfusLM 完整支持生成任务（连 LM head 一起混淆）——这正是 DeepInvert 所谓「任务相关张力」的根源。

### ObfusLM 已被攻破（DeepInvert, 2026-08）—— 三条教训

- 数字（已核对 Table 3）：RoBERTa+SST-2，ObfLM(ε=0.1) 推荐档 → DeepInvert top-1 **0.735**（InvBert 0.262）；ε=3 档 BERT 0.807 / RoBERTa 0.827。AG News 0.818、CoNLL 0.644；Qwen3.5-27B 医疗 QA (ε=1) 0.852，敏感 token S-Top1 0.769。ObfusLM 论文自评的"80% 抵抗率"是针对 KNN/InvBert/MLC 那一代攻击测的。
- **注意**：(k,ε)-anonymity 的证明本身没被推翻，被推翻的是"该保证在实用参数下够用"这一判断。

三条机制层教训（与我们的风险点结构同构，务必对照）：
1. **确定性混淆** → 作者自列的 Token Frequency Attack 成立；DeepInvert 的半监督自训练只是把它放大。
2. **簇内合成 = 加权平均 + 各向同性噪声**：加权平均是线性运算、保留语义结构；噪声零均值各向同性 + 信号低秩 → PCA 一压就提 SNR（即 DeepInvert Lemma 1）。与我们的 `[h,e]M` 同构风险。
3. **最关键**：(k,ε)-anonymity 保护的是"簇内不可区分"，而 **top-1 recovery 不是不可区分性的直接函数**——攻击者只要学到簇间可分性，再靠上下文在簇内挑，正确率可以远高于 1/k。⇒ **我们的安全论证必须直接对 top-1 recovery 建界，不能套 anonymity / LDP 式的不可区分性就宣称安全。**

### token 级替换（Santext/Custext）能否做到语义安全：不能（已论证）

三层拆解的结论：
- L1 密码学语义安全（密文不泄漏任何明文信息）：**结构性不可能** —— B 类是唯一一类「服务器看到的仍是明文自然语言」的方案，服务器可以直接读懂。
- L2 语义内容不泄漏（主题/实体类型/属性）：**设计使然，不能** —— 指数机制把替换限制在语义近邻内，语义保留正是它的效用来源，也是它的隐私死穴。
- L3 精确 token 不可指认：**ε 足够小时部分能**，代价是效用崩塌。

三条结构性理由：
1. **metric/指数机制保语义**：Pr[v|w] ∝ e^{−εd(w,v)/2}，d 是语义距离 ⇒ 替换词必在语义邻域 ⇒ I(W;Ṽ) 下界远离 0。它保护的是「哪个词」，不是「什么意思」。
2. **组合爆炸**：逐 token 独立加噪，序列级隐私损失按组合累积。per-token ε=3，20 token 的句子序列级 ε≈60，等于没有保护。
3. **停用词不扰动（CUSTEXT+）**：句子骨架完全明文泄漏。

**最有价值的反直觉证据**（DeepInvert Table 3，ε=3）：Custext 的 top-1 恢复 0.911–0.943，显著**高于**效用更差的 Santext（0.775–0.844）。⇒ **「为提升效用而收窄扰动范围」在隐私上是反向操作**。Custext 收窄到 K=20 近邻，等于只保护 identity、把语义原封不动留下。

对我们的对应警示：我们 exact 模式为保零精度损失，把可交换族收窄为「列置换 + 行内公共偏移」——同样是「收窄族」。必须论证这个收窄泄漏了什么（已知：列置换保住 attention 列结构，行内偏移完全不影响 softmax ⇒ exact 模式泄漏 softmax 的全部信息）。

### 语义安全 claims 的最终边界（重要，写论文前必读）

结论：**九种方案全部无法保证语义安全**，且这不依赖 DeepInvert，是结构性的 —— 保效用 ⟺ 保语义，是定义级别而非攻击级别。

泄漏层级（由易到难，越往上越不可能守住）：
1. 主题 / 情感 / 写作风格 —— 必然泄漏
2. 实体类型 / 敏感属性 —— **设计使然**。最强证据：CAPE 自己报告 ε=0.1 时 attacker 性别 F1 仅从 0.762 降到 0.751（降 0.011），等于没保护。
3. 语义邻域（近义词级）—— 机制保留（TextObfuscator 的功能原型就是为此设计，它 CoNLL F1 89.11 就是语义保留的证明）
4. 精确 token identity —— DeepInvert 在此层攻破（0.735）

**关键逻辑（写论文时务必用对）**：DeepInvert 提供的证据是**充分不必要**的。
- 攻破第 4 层（最难）⇒ 1–3 必然泄漏。✅ 可引用。
- 但**守住第 4 层 ⇏ 守住 1–3**。反例：TextObfuscator top-1 恢复仅 5.48%，但"这里有一个人名"这个语义被完整保留（否则 NER 做不了）。⇒ **top-1 recovery 低 ≠ 语义安全**，不能拿低 ASR 当语义安全的证据。

另外：九种方案提供的是 LDP / anonymity，其定义是"相邻输入不可区分"，**从来不是"语义不泄漏"**。用它们从未声称的标准去要求它们不算公平，但业界与使用方正是按"语义安全"来理解的，这个 gap 是真实的、且值得写。

**对我们**：exact 模式数学等价 ⇒ 语义信息 100% 保留在计算里，只是被密钥变换编码。我们的安全性**完全来自密钥（M 与 ξ）的秘密性，不来自信息破坏**。因此 fastProve 不提供、也永远不应声称语义安全——必须自己先说破，否则是最容易被审稿人一句话打死的地方。

### SCX「只锁首尾」论证的原文与对 fastProve 的适用性（重要更正）

原文（SIGCOMM'25, SCX, p.4/6/8）：
- **Definition 2 (Statelessness)**（§3.1, p.4）：*The on-cloud KV-cache is stateless if the probability that the cloud can generate the next token without access to the user's key is sufficiently low. Let k_u denote the user's key. Formally: P_cloud[f(x_t, S_{t−1}) = f(x_t, S_{t−1}|k_u)] ≤ ε, where ε ∈ [0,1].*（注意此 ε 与 DP 的 ε 重名，是原文记号冲突）
- **Insight 1 (Sharing keys for intermediate blocks)**（§3.3, p.4）：*The forward inference process in Transformers is sequential, meaning that **as long as the KV-cache of the first and last blocks remains "locked" the cloud cannot complete the decoding process independently**.* 复杂度 O(T·L) → O(T+L)。
- **§4.3 Keys Sharing**（p.6）：*we **relax the protection range to include only the first l_s layers and the last l_e layers**. The user **uploads** the encoding keys for all intermediate Transformer blocks {π^l_{l_s+1:L−l_e}} to the cloud.* 取 l_s=1 或 2、l_e=1。
- **Lemma 2 (Stateless KV-Cache)**（§5.4, p.8）：*P_cloud[f(x,S) = f(x,S|k_u)] ≤ max(1/d!, exp(ε)/V)*，其中 ε=Δx/s；d∈[1k,8k]、V≈30k、ε≈0.01 ⇒ 界 ≈ 1/V ≈ 3.4e-5。

**三条关键判读（写论文前必看）**：
1. **中间层是完全不设防的**（密钥已上传），支撑它的是**顺序依赖**（sequential dependency），不是协变性。措辞是 cannot complete ... **independently**，是"独立跑不通"，不是"学不到"。
2. **Lemma 2 的界 ≈ 1/V**，等于"在词表上均匀瞎猜"——这是一个**输出端/能力端**保证（云不能自己续写），**不是输入端保密保证**。SCX 的输入隐私走的是另一条更弱的线（Laplace (ε,0)-DP + distance correlation 实证）。
3. **不能直接照搬到 fastProve**：我们的 c_ℓ 是**直接暴露**的，破任一层 M_ℓ 即直接拿到 h_ℓ，层间不存在"缺首尾就拿不到中间"的依赖。照搬会得到假论证。
   - **成立的替代版本**：若各层 M_ℓ 由同一主密钥派生，则"破任一层"≡"破主密钥"，此时**最小充分保护集 = 主密钥**。这依赖**密钥派生结构**，不是顺序依赖。
4. **checkpoint 是正确性需求而非隐私需求**：非线性不与 M 交换 ⇒ 必须有断点才能算对。想减少 checkpoint 只有两条路：(a) 找更大的、与非线性兼容的变换族（Softmax 的可交换族目前只有「列置换 + 行内公共偏移」）；(b) 转 approximate。**SCX 减的是通信轮次（用户参与次数），不是模型里的断点数**，两者不同层面。真正能借的是**方法论**：先精确定义最小安全目标，再证明只需保护一个最小集合，其余放开换效率。

### KV-Cloak（NDSS'26，"Shadow in the Cache"）—— 最直接的近邻工作，必须正式区分

浙大+华为。结构：q^m=qM1^{-1}, k^m=kM1, v^m=vM2, W_o^m=W_o M2^{-1}；离线融合进权重 W_q^m=M1^{-1}W_q, W_k^m=W_k M1, W_v^m=M2 W_v。即**配对逆变换保 QK^T 与 Attention·W_o 不变**——与 fastProve 的协变思路同一族，只是只覆盖 attention 的 QKVO，不含 RMSNorm/FFN。

**三条对我们有直接价值的判读**：
1. **RoPE 交换性是硬约束，且他们为此牺牲了 M1 的通用性**：融合可行的充要条件是 R_{d,i}M1=M1R_{d,i}，他们的解法是把 M1 限制为 **2×2 旋转-缩放块对角**（与 RoPE 同构）。我们 AGENTS.md 里"RoPE 之后再施加公共正交 C"是另一种解法：**放弃融合、在 checkpoint 内在线施加**，换 M 的通用性。这是可以写进论文的明确 trade-off（密钥空间 vs 在线开销），且我们的选择更通用。
2. **他们自己攻破了 naive 可逆线性混淆，这个攻击适用于我们**：§V-B，CPA 下差分攻击——构造 K1、K2 使 ΔK 逼近标准基矩阵，利用 ΔK̄=S(ΔK)M 的线性逐列/逐行解出 S 与 M（至标量模糊），复杂度 **O(b²d+bd²)**。⇒ **固定 M + 无 per-request 随机性 ⇒ 必被打穿**。我们的解药是增广噪声态：`Δc=[Δh,Δe]M`，Δe 每请求随机且攻击者不可控，破坏代数可解性。这条要写成显式claim + 实验（复现他们的差分攻击，开/关 ξ 对比）。注意 KV-Cloak 的加性 mask A 是**固定 beacon**（用于 magnitude 检测隐式恢复 P̂），不提供这个保护——这是我们可以正面打的点。
3. **他们的随机性来源是"一次性块内置换 P̂"**（b! 组合屏障）+ 加性 mask 保秩，而非噪声。我们的随机性来源是 ξ。两者可比的维度是"每次请求引入的熵"，应做定量对比。

**与 SCX 的关系**：KV-Cloak 参考文献里引了 SCX，但正文未正面区分（只在参考文献出现）。两篇技术内核不同（SCX=客户端密钥+TEE 编解码+DP 噪声+通信复杂度；KV-Cloak=纯代数混淆、无需客户端在线）。写 related work 时应按"云端能否独立推理"分两大类，再把 fastProve 放在 KV-Cloak 同一类里做差异化。

### 待办（已写入报告第 6 节）
- P0-1：PCA 子空间分离攻击实验
- P0-2：定位 arXiv 2603.01499 的差异
- **P0-3：复现 KV-Cloak §V-B 的 CPA 差分攻击（O(b²d+bd²)），开/关 per-request ξ 对比，作为 fastProve 必需的反例防御实验**
- P1-1：最优攻击者 + 优势上界证明线
- P1-2：Softmax 噪声改在差分商空间
- P1-3：开销用延迟分解法（RTT + fixed overhead）剥离
- P1-4：攻击基线升级为穷举 / LLM 辅助恢复 / 拟合式反演（SCX 三类）
