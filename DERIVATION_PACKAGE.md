# 增广协变推理 v2：方案、推导主线与可行性边界

日期：2026-09-16。性质：修正后的研究方案，不是安全认证或已完成实现声明。

配套：[完整数学证明](PROOF_PACKAGE.md) · [攻击工作因子](docs/work_factor_security.md) · [工程实施指南](docs/augmented_covariant_implementation_v2.md)。

## Target：要证明的可行性

建立一个严格遵循“先 AloePri 扩维、再增广噪声、最后联合混合”的 Transformer 推理系统：

$$h_\ell\xrightarrow{\hat P_\ell}x_\ell=h_\ell\hat P_\ell
\xrightarrow{\operatorname{concat}\ e_\ell}u_\ell=[x_\ell,e_\ell]
\xrightarrow{M_\ell}c_\ell=u_\ell M_\ell.$$

线性映射通过离线权重转换实现；非线性检查点只操作合法语义；exact 模式在实数模型中等于同一明文模型；approximate 模式的有效 logits 扰动具有可证明的局部误差界。

安全目标不是“永远无法恢复”，而是：在相同观察范围、恢复质量、硬件和调参预算下，使攻击者最便宜的恢复路径比 AloePri、无噪扩维和其他选定基线消耗更多计算资源。保护对象分为传输记录观察者和持有部署参数的云端执行者，两种观察范围单独评价。当前文件只建立方法与成本度量；是否产生工作因子倍率必须由 recorded run 验证。

## Status：结论状态

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**。

相对于 [v1 报告](docs/aloepri_augmented_noise_architecture.md)，保留并明确以“扩维后拼接增广”为主方案，同时保留链式刷新、离线权重转换和可选端云切分；撤回“拼接后环境空间满秩”“免疫零空间与不变性攻击”“首层 RMSNorm 切分即安全闭环”和未经运行支持的成本数字。

| 结论 | 状态 | 证明位置 |
|---|---|---|
| 扩维后拼接增广与线性转换精确 | 已证，实数运算 | P1、P3 |
| 扩维后拼接使随机自由度增加 $r$ | 已证，但环境维度也增加 $r$ | P1 |
| 扩维后拼接消除原 $D-d$ 个秩亏方向 | **不能解决**，零空间亏损保持不变 | P1 |
| 补空间噪声加拼接噪声可填满 $D+r$ 维环境 | 已证，是解决低秩问题的增强候选 | P2 |
| 噪声线性链可控 | 已证，限定传播与刷新条件 | P4 |
| exact Transformer 与明文相等 | 已证，非线性契约、残差和缓存一致 | P5–P9 |
| Top-k 集保持、Softmax 分布和输出误差有界 | 已证，限定有效位置与扰动上界 | P10–P11 |
| 多层误差传播与 greedy 一致 | 条件性已证，需层常数和输出间隔 | P12 |
| 辅助噪声提高 cloud COA 攻击工作因子 | 待同硬件攻击成本实验；便宜旁路决定上限 | P13、P15 |
| 切分首个 RMSNorm 足够保护输入 | 不成立为一般结论 | P14 |
| 真实模型精度、延迟、隐私操作点 | 待新实现与 recorded run | 实施指南 G4–G5 |

## Invariant Object：统一不变量

每个持久化的混合状态都对应同一个 AloePri 扩维信号、辅助噪声和合法语义状态：

$$x_\ell=h_\ell\hat P_\ell,\qquad
c_\ell=[x_\ell,e_\ell]M_\ell,\qquad
\operatorname{Dec}_\ell(c_\ell)=h_\ell.$$

这里解码式是证明与合法方验证的工具，不等于允许把解码矩阵或明文状态发布给攻击者。exact 的目标是逐检查点保持这一不变量；approximate 的目标是控制其语义输出相对明文计算的偏差。

## Assumptions：前提与信任边界

1. 全部公式使用行向量。数学权重 $W$ 的形状是 $[d_{in},d_{out}]$；PyTorch linear 参数保存 $W^T$。
2. 每层 $\hat P_\ell\in\mathbb R^{d_\ell\times D_\ell}$ 具有右逆 $\hat Q_\ell\in\mathbb R^{D_\ell\times d_\ell}$，满足 $\hat P_\ell\hat Q_\ell=I$；$M_\ell\in\mathbb R^{(D_\ell+r_\ell)\times(D_\ell+r_\ell)}$ 可逆且数值可控。实数证明与 FP32/BF16 数值门禁分开；不在 forward 中求逆。
3. 明文和混淆路径共用 checkpoint、样本、mask、RoPE、GQA 映射、dtype 规则及推理模式；关闭训练 dropout。适配器正确性另与官方模型实现核对。
4. 非线性操作只作用于合法信号表示。参考检查点允许内部算出该表示，production API 不返回解码状态、噪声或注意力概率。此 API 约束不保证内部秘密对云端不可推导。
5. 指定融合操作不返回内部 clean temporaries，是实现假设，不是硬件隔离。允许修改内核、dump 寄存器、任意 hook 或读取客户端秘密的攻击者不在本原型保护范围。
6. exact 使用零 logit 噪声、零 RMS 分母噪声。辅助噪声可以非零且逐请求变化，因为它在合法语义读取中精确消除。
7. 概率命题需要相应独立性和有限二阶矩；对固定输入条件分布与跨输入总体分布分别陈述。实验 PRNG 的确定性不等于密码学随机性。
8. 所有安全结论绑定实际观察范围。云端实际持有的参数、缓存、可见随机状态必须计入攻击输入；不能靠命名白名单把真实可见信息排除。
9. “更安全”只表示在预登记攻击集合与恢复目标下观测到更大的攻击工作因子；不表示不可恢复或密码学计算安全。

## Notation：记号

| 符号 | 形状/含义 |
|---|---|
| $h,x,e,c$ | 语义、AloePri 扩维信号、辅助噪声和混合态，宽度 $d,D,r,N=D+r$ |
| $\hat P,\hat Q$ | AloePri 扩维与右逆，$d\times D,D\times d$，$\hat P\hat Q=I_d$ |
| $M=[M_x;M_e]$ | $N\times N$ 可逆联合混合矩阵，上下分别为 $D,r$ 行 |
| $L=M^{-1}[\hat Q;0]$，$J=M^{-1}[0;I_r]$ | 语义、噪声读取矩阵，满足 $cL=h,cJ=e$ |
| $C,G,\xi$ | 信号到噪声耦合、噪声传播矩阵、逐记录刷新向量 |
| $B$ | 对照方案的补空间噪声基，$(D-d)\times D$，$B\hat Q=0$ |
| $R,U_j$ | RMS 合法正交基、RoPE 之后的每 KV 组共同正交变换 |
| $S,p,q,\eta,\tau$ | 单行有效 attention logits、明文/扰动分布、噪声和无穷范数上界 |
| $\epsilon_{rms}>0$ | 模型原始 RMSNorm epsilon，与误差预算分开 |
| $\mathcal V$ | 当前 query 的有效 key 位置集合 |

## Derivation Strategy：推导策略

先选有效自由度，再证明状态转换；随后把 Transformer 分解为合法非线性检查点和线性混合输出，用归纳得到 exact 网络结论。approximate 从单行 Softmax 的指数倾斜出发，推导概率分布、Value 聚合及多层传播界。最后用条件性反例明确这些正确性结论为何不能直接推出隐私。

## Derivation Map：依赖图

1. P1 确定扩维后拼接增广的读取恒等式与真实秩；P2 给出补空间填噪对照。
2. P3–P4 确定线性转换、刷新和噪声幅度控制。
3. P5 将局部语义函数提升为检查点；P6–P8 给出 RMSNorm、attention、SwiGLU/router 的具体合法实现。
4. P9 用逐层/逐 token 归纳连接到 exact LM。
5. P10 证明排序条件；P11 证明局部分布与输出界；P12 明确怎样才能向整网和生成推广。
6. P13–P14 限定安全与端云切分，不把可行性证明误用为保密性证明。

## Main Derivation：方案主体

### 1. 主方案：扩维后拼接增广

每层先使用 AloePri 映射，再拼接逐请求噪声：

$$x=h\hat P\in\mathbb R^D,\qquad
u=[x,e]=[h\hat P,e]\in\mathbb R^{D+r},$$
$$c=uM=[h\hat P,e]M,\qquad M=Q_LD_sQ_R.$$

$Q_L,Q_R$ 为正交矩阵，$D_s$ 为有界正对角矩阵。其奇异值等于 $D_s$ 对角元，因此条件数可直接控制。首个实验用 $r=8$ 或 $16$、条件数上限 $3$，同时保留 $1,10$ 作为结构与数值对照。矩阵稠密程度不作为安全评分。

合法语义与噪声读取为：

$$L=M^{-1}\begin{bmatrix}\hat Q\\0\end{bmatrix},\qquad
J=M^{-1}\begin{bmatrix}0\\I_r\end{bmatrix},$$
$$cL=h,\qquad cJ=e.$$

在 $\operatorname{Cov}(h)\succ0$、$\operatorname{Cov}(e)\succ0$ 且二者独立时：

$$\operatorname{rank}\operatorname{Cov}(c)=d+r.$$

环境维度是 $D+r$，因此零空间亏损始终为：

$$\operatorname{nullity}\operatorname{Cov}(c)=(D+r)-(d+r)=D-d.$$

这条结论非常关键：拼接噪声确实增加了 $r$ 个独立随机自由度，并使同一 token 的可见表示逐请求变化；但它同时增加 $r$ 个坐标，所以没有消除 AloePri 扩维产生的 $D-d$ 个线性约束。固定 $h$ 时条件协方差秩只有 $r$。因此“动态化”成立，“环境满秩”和“免疫 SVD”不成立。

### 2. 增强候选：补空间噪声加拼接噪声

如果旧问题明确包含“消除 AloePri 扩维后的低秩零空间”，单纯拼接不够。令 $k=D-d$，取 $B\in\mathbb R^{k\times D}$ 的行为空间为 $\ker(\hat Q^T)$，增加补空间噪声 $a\in\mathbb R^k$，同时保留拼接链式噪声 $e\in\mathbb R^r$：

$$x_{full}=h\hat P+aB,\qquad
c_{full}=[x_{full},e]M.$$

令

$$T=\begin{bmatrix}\hat P\\B\end{bmatrix}\text{ 可逆},\quad
F=\operatorname{diag}(T,I_r).$$

于是

$$c_{full}=[h,a,e]FM,$$
$$c_{full}M^{-1}\begin{bmatrix}\hat Q\\0\end{bmatrix}=h.$$

若 $h,a,e$ 相互独立且各自协方差正定，$FM$ 可逆，因此

$$\operatorname{rank}\operatorname{Cov}(c_{full})=d+k+r=D+r.$$

这确实消除了“跨输入协方差存在固定 $D-d$ 维零空间”这一特定问题。若 $D=d+2s$，补空间噪声必须有 $k=2s$ 个独立方向；少于 $D-d$ 时仍会留下零空间。拼接噪声 $e$ 继续承担跨层链式刷新，补空间噪声 $a$ 负责填充 AloePri 原扩维冗余，两者职责不同。

为避免右逆病态，优先用正交完成构造 $T$；必须兼容已有 AloePri $\hat P,\hat Q$ 时则测 $T$ 和有效基 $FM$ 的实际条件数，超限拒绝转换。这个增强候选只解决低秩统计特征；因为合法读取矩阵仍然存在，P13 的权重反解、Gram 和语义几何泄漏仍可能成立。

在线线性层不需要另写一套代数。令总辅助态 $q=[a,e]$、有效基 $M_{eff}=FM$，直接使用

$$\bar K=\begin{bmatrix}W&C_q\\0&G_q\end{bmatrix},\qquad
\widetilde W=M_{eff,i}^{-1}\bar K M_{eff,o},$$

即可得到 $[(hW+b),q']M_{eff,o}=[(hW+b)\hat P_o+a'B_o,e']M_o$。这证明增强候选和主拼接方案可以共享转换器的规范化核心。

### 3. 线性层与刷新

令输入和输出分别使用 $(\hat P_i,\hat Q_i,M_i)$、$(\hat P_o,\hat Q_o,M_o)$。对

$$h'=hW+b,\qquad e'=hC+eG+\xi,$$

构造完全位于扩维增广坐标中的矩形块矩阵：

$$K_{i\to o}=
\begin{bmatrix}
\hat Q_iW\hat P_o&\hat Q_iC\\
0&G
\end{bmatrix},\qquad
\widetilde W=M_i^{-1}K_{i\to o}M_o,$$
$$c'=c\widetilde W+[b\hat P_o,\xi]M_o.$$

于是 $c'=[(hW+b)\hat P_o,hC+eG+\xi]M_o$。在线过程没有单独输出 $h$ 或 $h\hat P_i\hat Q_i$ 张量；$\hat Q_i$ 被吸收进部署权重。它解决“显式先降成明文张量”的工程问题，却没有消灭代数上的语义读取映射。

部署保存合成权重；刷新必须进入噪声行，不能对全混合坐标任意加噪后仍标记 exact。默认 $C=0$、$G=\gamma P_e$、$\gamma=0.5$，其中 $P_e$ 为符号置换。非零 $C$ 是有形状和能量预算的扩展项，不预设它改善安全。

校准尺度取独立公开校准集的 $H=\sqrt{\mathbb E\|h\|_2^2}$。concat 定义 $\sigma_e=\beta_eH/\sqrt r$；full 另定义 $\sigma_a=\beta_aH/\sqrt{D-d}$。不能把每坐标 RMS 直接当作 $H$，也不能把 $a,e$ 两类能量混成一个未注明的 beta。线性链方差匹配只适用于 P4 的条件，attention 和残差之后必须重新记录噪声能量。

### 4. 非线性与残差

检查点实现合法 $h\mapsto F(h)$，保留旧噪声旁路，再编码为 $[F(h)\hat P_{out},e']M_{out}$。允许把连续步骤融合，但要记录实际内部量和普通输出。该契约证明存在正确计算路径，不证明服务器无法恢复信号。

同一 block 内优先固定 $(\hat P,\hat Q,M)$。若改变基，旧残差通过

$$M_i^{-1}\begin{bmatrix}\hat Q_i\hat P_o&0\\0&I_r\end{bmatrix}M_o$$

重新编码。RMSNorm 只使用 $d$ 维语义范数，不能使用 $D+r$ 维混合态范数。学习到的 gamma 吸收入随后投影；RoPE 先作用于合法 Q/K，再施加同 KV 组共同正交矩阵。

Value 采用 $c_V=[V,e_V]M_V$，attention 左乘后仍是 $[AV,Ae_V]M_V$。SwiGLU 的 gate 只做共享置换，up 做有界缩放加同一置换，down 做逆补偿；噪声从旁路接回。完整公式和偏置处理见 P6–P8。

### 5. approximate：只扰动有效 logits

对每行有效 logits 生成中心化、缩放、最终裁剪的 $\eta$，确保实际 $\|\eta\|_\infty\le\tau$。无效位置永远保持 $-\infty$。

$$\tau_{topk}=\min(\tau_{max},\alpha\Delta_k/2,\tau_{error}),\quad 0<\alpha<1,$$
$$\tau_{free}=\min(\tau_{max},\tau_{error}).$$

其中 $\Delta_k=S_{(k)}-S_{(k+1)}$。边界并列、有效位置数不超过 $k$ 时，topk-preserving 取零噪声。全 mask 行按接口约定输出零向量，单独统计，不将其称为全 $-\infty$ 的数学 Softmax。

对同一有效支持，P11 给出：

$$\|q-p\|_1\le\min(2,\tau),\quad
D_{KL}(p\|q),D_{KL}(q\|p)\le\tau^2/2,$$
$$\|(q-p)V\|_2\le\frac{D_V}{2}\min(2,\tau),\quad
D_V=\max_{i,j\in\mathcal V}\|v_i-v_j\|_2.$$

若有经过认证的 $D_V\le D_{cap}$，设置 $\tau_{error}\le2\varepsilon_O/D_{cap}$ 足以保证该 head 当前输入上的局部误差不超过 $\varepsilon_O$。只从样本分位数估计 $D_{cap}$ 是经验预算，不是全输入保证。该界不直接给出整网 PPL 或 token 恢复率。

### 6. 可选端云切分

默认研究路径不强制增加端侧计算。若选择切分，首个合法候选边界是完整 decoder block 输出，传输完整的混合残差态，并定义客户端前缀 KV cache、服务端后缀 cache 和每个生成 token 的往返协议。只传首个 RMSNorm 输出不能替代该边界。

切分深度 $k$ 是独立待测变量，不保证隐藏 token；尤其因果序列首 token 在固定位置没有前文，其深层状态仍可能形成候选指纹。客户端 embedding 的存储、前 $k$ 层计算、混合态通信都计入成本。

### 7. 它对旧问题到底解决到什么程度

| 旧问题 | 结论 | 原因 |
|---|---|---|
| 同一 token 的表示是固定确定向量 | **部分解决** | `per_request` 的 $e,\xi$ 使可见 $c$ 动态变化 |
| 原扩维信号只占 $d$ 维子空间 | concat **没有消除**；full 增强候选**可以消除该统计特征** | concat 秩亏仍为 $D-d$；加入 $D-d$ 个补空间自由度后总体秩可达 $D+r$ |
| 算子入口显式执行 $\hat Q$ 降维 | **工程上可解决** | $\hat Q$ 可离线吸收进 $\widetilde W$，不物化独立降维张量 |
| 云端能否从部署权重反解语义投影 | **没有解决** | 合成权重仍可能暴露 $L$ 或等价方程，需实际 bundle 审计 |
| Gate/Attn 等简单确定性不变量 | **可能扰动，未证明抵抗** | 噪声会改变原始坐标统计，但 exact 路径仍保留可计算语义不变量 |
| RMSNorm 范数和 QK/Softmax 几何泄漏 | **没有解决** | exact 推理仍必须计算这些语义量或等价量 |
| 模型精度 | **数学上可解决** | P3–P9 给出 exact 构造，FP32 仍须实测 |
| approximate 精度—噪声关系 | **局部可界定** | P10–P12 给出条件误差界，不等于任务精度结论 |
| 面对可修改 kernel/寄存器导出的云端 | **不能解决** | 需要 TEE/MPC/HE 或新的可信边界 |

因此，`expanded_concat` 能形成正确的“扩维后增广”基线并动态化传输表示；`expanded_full` 还能严格消除原低秩零空间这一项问题。真正的验收问题应改为：在固定观察范围和恢复目标下，相比 AloePri 与无噪对照，攻击者达到相同恢复质量所需的最小计算成本是否显著增加，同时防御端正确性与开销是否达标。

### 8. 安全目标：提高最小攻击工作因子

给定方案 $s$、观察视图 $v$、目标数 $m$ 和恢复目标 $\rho$，定义：

$$W_{s,v}(\rho,m)=\inf_{A\in\mathcal A_v}
\{C^{ref}_{s,v,A}(m):R_{s,v,A}(m)\ge\rho\}.$$

$C^{ref}$ 是固定参考硬件上的攻击总时间或 GPU-hour，原始记录同时保留 FLOPs、峰值内存、查询数、观测数和训练数据量。相对基线 $b$ 的倍率为：

$$G_{s/b,v}(\rho,m)=\frac{W_{s,v}(\rho,m)}{W_{b,v}(\rho,m)}.$$

项目目标是 $G>1$ 且置信区间下界大于 $1$；建议把 $G\ge10$ 作为首轮有意义的预登记工程目标，而非已有结果。由于实验只能覆盖有限攻击集合，报告必须写“对已评测攻击集合的观测工作因子”。

一次性恢复密钥/子空间的准备成本和逐目标成本分开：

$$C_A(E,m)=E C_{setup}+mC_{target}.$$

所以密钥轮换只有在跨 epoch 攻击不可迁移时才能提高摊销成本。任何右逆、Gram 或零空间攻击若比学习型反演便宜，就由它决定当前方案的工作因子；不能选择最慢的攻击作为安全证据。完整合同见 [攻击工作因子文档](docs/work_factor_security.md)。

## Remarks and Interpretation：预期收益与取舍

本次修正使核心工程任务变得可判定：指定的扩维后拼接路径有完整 exact 代数；补空间加拼接的 full 版本解决低秩特征并作为推荐攻击候选；approximate 扰动能与局部输出预算关联。辅助噪声是否增加攻击难度仍取决于可见参数能否消去它。

精确辅助噪声与影响语义的 logit 噪声是两个独立轴。前者可以保持模型语义，后者改变 attention；保护效果、精度代价和能量均分开记录。已有 rho-noise 工作保留，但新 v2 首轮固定 `rho_noise_bound=0`，之后单独消融，避免将两个扰动源的效果混为一谈。

## Boundaries and Non-Claims

- 不声称 LWE、FHE、端到端加密、DP 或无条件输入保密。没有归约或隐私参数证明。
- 满秩、有效秩、坐标覆盖、混合态距离变化均不是攻击恢复率的替代指标。
- 条件性权重反解、Gram 恢复、噪声投影成功时，其成本构成当前工作因子的攻击上界；只有比基线显著更贵时，才能在限定攻击集合下报告成本提升。
- 不把旧 structured bundle、随机 tiny、理想观察者实验的结果迁移为新 v2 结果。
- 软件隐藏内部临时量不保护能修改 kernel、导出寄存器、读取全部客户端秘密的攻击者。
- 正确性证明不保证 FP32 门禁、任务精度、吞吐率或有限预算反演防御一定通过。

## Open Risks：进入实现前仍需解决

1. 当前部署的公开 Q 投影存在已记录的条件性反解通道；输出侧未知变换即使阻断该特定公式，也未证明能抵抗联合权重求解。
2. 一般双侧 dense 基不能直接冒用旧 StructuredBasis 的块 Gram 快速路径。
3. 阶段 A runner 的 dtype 门禁、能量标定、域分离与配置接入仍需修复；见实施指南 G1。
4. 重复观察、公开词表、KV cache、输出 logits 和请求关联可能保留大量信息。
5. 需要真实 checkpoint 的基线一致性、accuracy 和资源检查，才能讨论实际使用点。
6. 尚无针对全部攻击的复杂度下界；当前只能构建攻击成本曲线，不能证明密码学意义的计算安全。

## 来源与证据层级

- AloePri v2：[论文](https://arxiv.org/abs/2603.01499v2)，仅用于既有方法背景；本文件的证明独立展开，不继承其安全或性能数字。
- [2026-09-15 交接](docs/AUGMENTED_NOISE_HANDOFF_20260915.md)：已有实现、观察范围及旧部署反解证据。
- [阶段 A 复核](docs/augmented_noise_stage_a_review.md)：既有数值记录的缺陷及修复要求。
- 本次可运行的数学检查与原始结果见实施指南“本次交付验证”；它们辅助检查证明，不是形式化验证器或模型评测。
