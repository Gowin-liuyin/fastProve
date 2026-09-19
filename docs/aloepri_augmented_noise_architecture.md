# 基于 AloePri 升维流形的增广带噪协变混淆架构技术报告与方案论证

> **状态：已由 2026-09-16 修订方案取代。** 本文保留为历史设计稿，不再作为数学或工程验收依据。当前需求基线仍遵循本文提出的“先 AloePri 扩维、再拼接辅助噪声、最后联合混合”：
>
> $$c=[h\hat P,e]M_{aug}.$$
>
> 关键修正是：若 $D>d$，即使 $h$ 与 $e$ 的协方差均非退化，$c$ 在 $D+r$ 维环境中的协方差秩也只有 $d+r$，零空间维度仍为 $D-d$，不能称为环境满秩或据此宣称免疫 SVD。为真正消除该固定零空间，修订方案增加 `expanded_full` 候选 $c_{full}=[h\hat P+aB,e]M$，其中 $B\hat Q=0$ 且 $\dim a=D-d$；它能严格达到 $D+r$ 维总体满秩，但仍不能单凭满秩宣称云端输入保密。请以 [修订方案](../DERIVATION_PACKAGE.md)、[数学证明](../PROOF_PACKAGE.md) 和 [工程实施指南](augmented_covariant_implementation_v2.md) 为准。
>
> 当前安全目标也已修正为“在相同恢复质量下提高最便宜攻击的计算成本”，不再要求不可恢复。正式指标、摊销模型和比较协议见 [攻击工作因子](work_factor_security.md)。

**文档编号**：FP-ARCH-2026-09-16  
**版本**：v1.0 (理论推导与工程设计基线)  
**作者**：fastProve 研发团队  
**日期**：2026-09-16  

---

## 1. 来源、学术定位与研究背景

### 1.1 依托的基础工作（Upstream Work）
本项目方案直接建立并演进自以下开创性工作：

* **核心论文原件**：  
  `/Users/yin/Downloads/2603.01499v2.pdf`  
  * **标题**：*Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)*  
  * **arXiv 索引**：[arXiv:2603.01499v2 [cs.CR] 30 Mar 2026](https://arxiv.org/abs/2603.01499)  
  * **作者团队**：Yu Lin, Qizhi Zhang, Wenqiang Ruan, Daode Zhang, Jue Hong, Ye Wu (ByteDance), Hanning Xia, Yunlong Mao, Sheng Zhong (Nanjing University).  
  * **项目归属与合作方**：西安电子科技大学网络与系统安全实验室（Xidian NSS Lab，沈玉龙教授团队）与字节跳动联合研发（浙江省尖兵研发攻关计划专项配套课题）。
* **开源参考实现仓库**：  
  [https://github.com/XidianNSS/Aloepri](https://github.com/XidianNSS/Aloepri)  
  *(衍生及前置审计副本包括 `Gowin-liuyin/Aloepri` 与 `sheng1feng/Aloepri`)*。

### 1.2 fastProve 与 AloePri 的承接关系
AloePri 首次提出了面向工业级大模型推理的“**协变混淆（Covariant Obfuscation）**”机制，确立了“通过模型权重离线转换，使云端能直接处理混淆数据，同时兼容 vLLM/SGLang 等服务框架”的端云协同技术路线。

**fastProve 项目定位**：  
本项目基于 AloePri 的端云系统模型，并关注其“子空间维度”与“动态噪声”问题。本文档历史版本提出：**“在混淆后、没有降维前增广；在已经升维的矩阵/状态基础上进行增广和加噪”**。修订结论是：该路线的 exact 计算可行，也可动态化表示；单纯拼接噪声不能消除原秩亏，云端安全收益必须通过攻击实验判断。

---

## 2. AloePri 原版机制深度复盘与底层硬伤

### 2.1 AloePri 的核心构造（Algorithm 1 升维与降维）
AloePri 的权重与中间状态混淆核心依赖其 Algorithm 1 生成的密钥矩阵对 $(\hat{P}, \hat{Q})$：

1. **输入与参数**：隐状态维度 $d$，扩维大小 $h$（扩维后总维度 $d' = d + 2h$），正则系数 $\lambda$。
2. **升维矩阵 $\hat{P}$**：
   $$\hat{P} = [B, \; C, \; E] Z \in \mathbb{R}^{d \times (d + 2h)}$$
   * $B = U + \lambda V \in \mathbb{R}^{d \times d}$（$U \in \mathbb{O}_d$ 为正交基）；
   * $E \in \mathbb{R}^{d \times h}$（由两个低秩高斯矩阵 $E_1 E_2$ 乘积生成，$\text{rank}(E) = h/2$）；
   * $C \in \mathbb{R}^{d \times h}$（列向量严格采样自 $F^T$ 的零空间，即 $C F = 0$）；
   * $Z \in \mathbb{O}_{d+2h}$（全局随机正交旋转矩阵）。
3. **降维矩阵 $\hat{Q}$**：
   $$\hat{Q} = Z^T \begin{bmatrix} B^{-1} \\ F \\ D \end{bmatrix} \in \mathbb{R}^{(d + 2h) \times d}$$
   * $F \in \mathbb{R}^{h \times d}$（由低秩矩阵 $F_1 F_2$ 生成）；
   * $D \in \mathbb{R}^{h \times d}$（行向量严格采样自 $E$ 的零空间，即 $E D = 0$）。
4. **恒等正交消除性**：
   $$\hat{P} \cdot \hat{Q} = [B, \; C, \; E] Z \cdot Z^T \begin{bmatrix} B^{-1} \\ F \\ D \end{bmatrix} = B B^{-1} + C F + E D = I_d + 0 + 0 = I_d$$

### 2.2 AloePri 的三大底层缺陷（Vulnerabilities）

#### 缺陷一：维度升上去了，但“代数秩没升”（低秩刚性嵌入）
* **数学机理**：虽然 $\hat{P}$ 把状态映射到了 $d+2h$ 维，但输入向量 $x \in \mathbb{R}^d$ 的自由度只有 $d$。因此，中间激活向量 $\tilde{x} = x \hat{P} \in \mathbb{R}^{d+2h}$ 在几何上**严格处于一个 $d$ 维的线性子空间内，其协方差矩阵的秩严格为 $d < d+2h$**。
* **攻击后果**：多出来的 $2h$ 个坐标根本没有引入独立的信息论熵，它们完全是前 $d$ 个维度的确定性线性组合。攻击者只要在服务端收集若干批次的 $\tilde{x}$，通过简单的 **SVD / PCA 或核空间投影（Null-space Projection）**，就能以 $100\%$ 的确定性把多余的 $2h$ 维当成冗余线性相关项直接剔除，使扩维防护形同虚设。

#### 缺陷二：降维降得太早（Premature Reduction）
* **算子入口即降维**：在 AloePri 的模型设计中（论文第 9–10 页）：
  $$W_{\rm query} = \hat{Q}_q W_q, \quad W_{\rm key} = \hat{Q}_k W_k, \quad W_{\rm gate} = \hat{Q}_{\rm gate} W_{\rm gate}, \quad W_{\rm up} = \hat{Q}_{\rm up} W_{\rm up}$$
  $\hat{Q}$ 被直接乘在注意力与 FFN 算子的**最前端输入端**！
* **后果**：向量刚刚经过 $\hat{P}$ 升维，在进入每一个算子内部做真正的非线性/注意力计算之前，就立刻被 $\hat{Q}$ 暴力压回了 $d$ 维无噪状态。这意味着**算子内部的计算全程处于无保护的低维空间**，根本没有利用高维流形来掩蔽语义。

#### 缺陷三：确定性刚性变换引发“不变性攻击”（Invariant Attack, IA）
* AloePri 论文第 26–27 页自身披露了其难以抵御 Invariant Attack（Gate-IA 和 Attn-IA）：
  * **Gate-IA**：由于变换是确定性乘法，激活前后存在统计不变量 $\text{Avg}(e W_{\rm gate}) = \text{Avg}(\tilde{e} \widetilde{W}_{\rm gate})$。攻击者利用各层的 Gate 权重统计均值，可以列方程把置换矩阵 $\Pi$ 倒推出来。
  * **Attn-IA**：注意力头的 $Q, K$ 变换存在分块等距不变量 $e (Q^T Q)^{-1} e^T = \tilde{e} (\widetilde{Q}^T \widetilde{Q})^{-1} \tilde{e}^T$。攻击者遍历词表即可还原明文与混淆 Token 的映射。

---

## 3. 核心创新方案：升维未降维前的增广带噪协变架构

针对 AloePri 的上述根因，我们提出全新设计：  
**“在混淆后、没有降维前增广；在已经升维的矩阵/状态基础上进行增广和动态加噪。”**

### 3.1 架构拓扑对比

```text
【AloePri 原版拓扑 (过早降维与秩亏)】
h (d维) ──► 乘 P̂ (升至 d+2h维，但秩仍为 d) ──► 算子入口立刻乘 Q̂ (降回 d维) ──► 算子内部无噪计算
                                                    ▲
                                                    │ 漏洞：无真噪声，核空间直接被剥离

【本方案拓扑 (升维流形上的增广加噪)】
h (d维) ──► 乘 P̂ (升至 D = d+2h维)
                 │
                 ▼ 【尚未降维前的关键黄金区！】
         拼接独立动态噪声态 e ∈ ℝʳ ──► 联合增广态 z = [hP̂, e] ∈ ℝ^(D+r)  (秩至多 d+r)
                 │
                 ▼ 高维闭环混合基: c = z M_aug ∈ ℝ^(D+r)
                 │
                 ▼ 全程在高维带噪空间中执行 (延迟显式降维，离线吸收 Q̂ 与 W)
```

### 3.2 详细数学构造与闭环公式

#### 第一步：高维状态增广（Augmentation on Expanded Manifold）
设基础模型隐藏维度为 $d$，AloePri 密钥扩维矩阵为 $\hat{P} \in \mathbb{R}^{d \times D}$（其中 $D = d + 2h$）。  
在升维空间中，信号表示为：
$$\tilde{h} = h \hat{P} \in \mathbb{R}^D$$
**核心增广**：在 $D$ 维的基础上，不进行降维，而是引入 $r$ 维独立的辅助随机噪声状态 $e \in \mathbb{R}^r$（$r \ge 16$），形成维度为 $N = D + r = d + 2h + r$ 的**高维联合增广状态**：
$$z = [\tilde{h}, \; e] = [h \hat{P}, \; e] \in \mathbb{R}^N$$
引入高维可逆混合基 $M_{\rm aug} \in \mathbb{R}^{N \times N}$（采用正交分块基，$\kappa_2(M_{\rm aug}) \le 10$），服务端实际持有的混淆状态为：
$$c = z M_{\rm aug} = [h \hat{P}, \; e] M_{\rm aug} \in \mathbb{R}^N$$

> **秩增加命题（修订后不称环境满秩）**：  
> 设 $h$ 为任意 $d$ 维非零向量，$e \sim \mathcal{N}(0, \sigma_e^2 I_r)$ 为独立随机高斯向量。尽管 $h\hat{P}$ 局限在 $D$ 维空间的 $d$ 维子空间中，但联合状态 $z = [h\hat{P}, e]$ 以及混淆状态 $c$ 的全域协方差矩阵满足：
> $$\operatorname{Rank}(\operatorname{Cov}(c)) = d + r > d$$
> 它把总体自由度从 $d$ 提高到至多 $d+r$，但环境维度是 $D+r$，仍保留 $D-d$ 个零空间方向。若要消除这一特定零空间，需要修订方案中的补空间噪声 $aB$；即使达到满秩，也不能据此推出攻击不可恢复。

#### 第二步：延迟降维与权重的离线深度吸收（Deferred Reduction & Fusion）
在 AloePri 中，$\hat{Q}$ 被放在算子最前端强行降维。在本方案中，**绝不在线执行降维**，而是将降维矩阵 $\hat{Q} \in \mathbb{R}^{D \times d}$、明文权重矩阵 $W \in \mathbb{R}^{d \times d_{\rm out}}$，以及噪声耦合矩阵 $C \in \mathbb{R}^{D \times r}$、$G \in \mathbb{R}^{r \times r}$ 统一在离线阶段**深度吸收到高维部署权重中**。

定义逆基矩阵切片：
$$P_{\rm aug} = (M_{\rm aug}^{-1})_{[:, :D]} \in \mathbb{R}^{N \times D}, \qquad N_{\rm aug} = (M_{\rm aug}^{-1})_{[:, D:]} \in \mathbb{R}^{N \times r}$$
则有：
$$c P_{\rm aug} = h \hat{P}, \qquad c N_{\rm aug} = e$$
利用 AloePri 的恒等式 $\hat{P} \hat{Q} = I_d$，我们可以从 $c P_{\rm aug}$ 中无损提取信号 $h$ 的投影：
$$(c P_{\rm aug}) \hat{Q} = (h \hat{P}) \hat{Q} = h (\hat{P} \hat{Q}) = h$$

由此，构造高维部署权重矩阵：

1. **线性层 / 注意力投影的高维部署权重**：
   $$W_{\rm deployed} = P_{\rm aug} \cdot \hat{Q} \cdot W_{\rm base} \cdot M_{\rm next\_top} + P_{\rm aug} C_{\rm signal} M_{\rm next\_bot} \in \mathbb{R}^{N \times N_{\rm next}}$$
2. **残差流与噪声链式传播**：
   $$W_{\rm nz} = (G - I_r) M_{\rm next\_bot} \in \mathbb{R}^{r \times N_{\rm next}}$$
   在线残差更新：
   $$c_{\rm next} = c + c W_{\rm deployed} + (c N_{\rm aug}) W_{\rm nz} + \xi_{\rm out}$$
   其中 $\xi$ 为逐请求/逐层派生的动态刷新噪声。

**推论**：  
服务端在物理上执行的始终是维度为 $N \times N$ 的稠密 GEMM，**完全不发生显式的 $\hat{Q}$ 降维操作**。降维操作已经被代数折叠进离线权重中。

---

## 4. 攻防安全性收益评估

下表原为方案假设。修订后仅将其作为待验证攻击假设，不能作为安全结论：

| 攻击类型 | AloePri 原版表现 | fastProve 升维增广方案表现 | 提升机制 |
| :--- | :--- | :--- | :--- |
| **1. 核空间剥离攻击 (Null-space Peeling)** | $\tilde{x}=x\hat P$ 的总体秩至多为 $d$ | concat 后秩至多 $d+r$，仍亏 $D-d$；full 候选可填满 | 必须分别攻击 concat/full，不能由秩直接判安全 |
| **2. 不变性攻击 (Gate-IA / Attn-IA)** | 特定配置下存在可利用不变量 | 动态噪声可能扰动原坐标统计 | 云端权重可能给出等价读取，需真实 bundle 验证 |
| **3. 词表匹配攻击 (VMA)** | 存在公开权重组合攻击面 | 双重混合是否阻断对齐尚未知 | 记录公开模型、转换权重和攻击预算后实测 |
| **4. 深度学习反演攻击 (DeepInvert / IMA)** | 固定表示便于训练攻击器 | 动态表示可能提高学习难度 | 这是实验假设，不预设恢复率或分布类型 |

---

## 5. RMSNorm 范数指纹的可选端云切分候选

我们在前序研究（`docs/PROBLEMS.md` §1.1）中证实了：  
**只要云端必须算对 RMSNorm 分母 $\rho = \text{rms}(h)$，在公开基座的前提下，第 0 层静态 Embedding 行范数就会成为查表还原 Token 的天然指纹。**

将“升维增广带噪矩阵”与端侧切分结合，可以作为降低首层直接范数指纹的候选；仅计算首个 RMSNorm 尚不能维持 pre-norm 残差闭环，也不能证明深层 token 信息消失。合法候选边界和成本应按修订实施指南重新设计。

### 5.1 端云切分闭环拓扑

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 客户端 (本地端；算力、内存与通信成本待实测)                              │
│                                                                         │
│   输入 Prompt ──► Tokenizer ──► Embedding 查表 ──► 本地计算第 0 层 RMSNorm│
│                                                                         │
│   ★ 核心收益：第 0 层 Embedding 的静态行范数在本地被完全吸收，永远不出本地！│
│   ★ 输出：直接产出已经完成归一化的第 1 层纯语义信号 h₁                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼ 采用本方案的【升维增广混淆编码】
                                       c₁ = [h₁ P̂, e₁] M_aug ∈ ℝ^(d+2h+r)
 ════════════════════════════════════╪═════════════════════════════════════
   网络传输信道 (唯密文，无静态范数指纹) │
 ════════════════════════════════════╪═════════════════════════════════════
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 服务端 (不可信云端 GPU；计算占比待实测)                                  │
│                                                                         │
│   从第 1 层至第 L 层：                                                 │
│   1. 全程运行在高维增广空间 ℝ^(d+2h+r) 上；concat 路径并非环境满秩；       │
│   2. 算子内部离线吸收 Q̂ 与 W，GEMM 速度与原生模型一致；                 │
│   3. 深层 h_ℓ 是上下文相关状态；范数及反演攻击效果需按语料实测；           │
│   4. full 候选可消除固定零空间；其他不变量和权重反解仍需攻击验证；         │
│   5. 末端输出经过词表置换 τ 的混淆 Logits。                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 工程落地与代码实施规划（Blueprint）

在现有 `fastProve` 代码库中落地该方案的步骤如下：

### 阶段一：高维密钥矩阵生成器改造 (`src/fastprove/aloepri_augmented.py`)
* 结合 AloePri Algorithm 1 的 $\hat{P} = [B, C, E] Z$ 与 fastProve 的增广基生成器；
* 构造联合映射矩阵 $M_{\rm aug} \in \mathbb{R}^{(d+2h+r) \times (d+2h+r)}$；
* 单元测试检验：
  1. $\hat{P} \cdot \hat{Q} = I_d$ 恒等性；
  2. $M_{\rm aug} M_{\rm aug}^{-1} = I_{d+2h+r}$ 条件数 $\le 10$；
  3. concat 的理论秩为 $d+r$、零空间维度为 $2h$；full 候选在充分条件下理论秩为 $d+2h+r$。

### 阶段二：离线权重融合器更新 (`src/fastprove/converter.py`)
* 更新 `DeployedAttentionWeights` 与 `DeployedFeedForwardWeights`：
  将输入端的 $\hat{Q}$ 吸收进 $W_q, W_k, W_v, W_{\rm gate}, W_{\rm up}$；
  将输出端的 $\hat{P}$ 吸收进 $W_{\rm attn}, W_{\rm ffn}$；
* 确保服务端模型 Bundle 维持零解混、无明文矩阵泄露。

### 阶段三：端云切分与抗反演攻防验证
* 编写攻击复核脚本 `scripts/verify_aloepri_subspace_attack.py`：
  对比“原版 AloePri”与“本方案升维增广态”在 SVD 降维反演下的表现；
* 分别测量切分前后 Token 恢复率、基线和置信区间；不预设 $<1\%$ 或其他未经定义的安全阈值。

---

## 7. 总结结论

1. **学术传承清晰**：方案以西电 NSS / 字节跳动的顶级前沿工作 **AloePri (arXiv:2603.01499v2)** 为基石，师出有名，立论严谨。
2. **击中原版要害**：精准指出了 AloePri 原版“升维不升秩（低秩刚性嵌入）”与“算子入口过早降维”的结构性漏洞。
3. **构造可继续验证**：扩维后增广和动态刷新具有 exact 代数实现；concat 不满秩，full 候选在充分条件下满秩，但安全收益仍待攻击实验。
4. **边界仍需实证**：首层切分、RMS 几何泄漏、部署权重反解、任务精度和性能均未由本历史稿闭环，应按修订方案逐项验收。
