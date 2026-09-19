> 旧框架，禁止复活：生产路径是 D = d；不得引用 expanded_concat/expanded_full/P̂/a/B_comp。

状态：HISTORICAL-已归档
日期：2026-09-16
上游依据：docs/INDEX.md（归档清单）
适用范围：历史研究记录；仅存档，不作验收依据

# 增广协变推理 v2：数学证明包

日期：2026-09-16。配套：[方案](2026-09-16_DERIVATION_PACKAGE.md) · [实施指南](2026-09-16_augmented_covariant_implementation_v2.md)。

## Claim：准确命题

存在满足下述假设的“先 AloePri 扩维、再拼接增广噪声、最后联合混合”计算路径，使 exact Transformer 与同一明文 Transformer 在实数运算中逐检查点一致；有界有效 logit 扰动满足 P10–P12 的排序和误差结论。该拼接方案把随机自由度从 $d$ 增至 $d+r$，但不能消除原扩维带来的 $D-d$ 个线性约束。上述结论不包含对任意被动云端观察者的保密性保证。

## Status

**PROVABLE AFTER WEAKENING / EXTRA ASSUMPTION**，相对于 v1 的安全闭环主张。

本文件 P1–P15 的修正命题在各自显式条件下可证；“满秩即可安全”“首层归一化即可消除指纹”“原生性能无损”均不属于已证命题。

## Assumptions

沿用方案中的行向量、模型一致性和威胁模型假设。除 P12 的数值误差项外，恒等式均在实数精确运算中证明。每个矩阵维度为有限正整数；所有涉及的 logits 和模型权重有限；模型 epsilon 严格为正；attention 有效支持非空的行才能使用通常的 Softmax 定义。符号置换和正交矩阵使用精确数学定义。

涉及概率的命题另外要求所列有限二阶矩和独立性，不默认真实模型隐藏态为满秩高斯。所有条件性攻击中的“已知矩阵”均须来自攻击者实际允许的观察，不允许评分方提供目标秘密。

## Notation

$h\in\mathbb R^{1\times d}$，$\hat P\in\mathbb R^{d\times D}$，$\hat Q\in\mathbb R^{D\times d}$，$\hat P\hat Q=I_d$；$x=h\hat P$，$e\in\mathbb R^{1\times r}$，$N=D+r$；$M=[M_x;M_e]\in\mathbb R^{N\times N}$ 可逆，其中 $M_x$ 是前 $D$ 行。定义

$$L=M^{-1}\begin{bmatrix}\hat Q\\0_{r\times d}\end{bmatrix},\qquad
J=M^{-1}\begin{bmatrix}0_{D\times r}\\I_r\end{bmatrix}.$$

范数 $\|\cdot\|_2$ 对向量为欧氏范数、对矩阵为谱范数；批状态采用 Frobenius 范数时明确注明。$\operatorname{Cov}(x)=\mathbb E[(x-\mathbb Ex)^T(x-\mathbb Ex)]$。KL 使用自然对数。

## Proof Strategy

P1–P4 用矩阵恒等式与二阶矩；P5–P8 构造合法算子；P9 归纳；P10 比较成对边界；P11 用 Softmax 路径导数和对数配分函数；P12 用三角不等式及间隔条件；P13–P14 给出明确反例条件；P15 形式化攻击工作因子及有限攻击集合的结论边界。

## Dependency Map

$$\text{P1/P2}\to\text{P3/P5}\to\text{P6/P7/P8}\to\text{P9}.$$
$$\text{P10/P11}\to\text{P12};\qquad \text{P1/P6}\to\text{P13/P14}.$$
$$\text{P13 + 攻击成本记录}\to\text{P15 的工作因子上界与倍率}.$$

P4 独立给出线性噪声链的能量边界；它不是 P9 的隐含平稳性假设。

## Proof

### P1. 扩维后拼接增广：读取恒等式、总体秩与不变的秩亏

**命题。** 令

$$c=[h\hat P,e]M=hA_h+eA_e,$$
$$A_h=[\hat P,0]M,\qquad A_e=[0,I_r]M.$$

则 $cL=h$、$cJ=e$。若 $h,e$ 独立且具有有限二阶矩，则

$$\operatorname{rank}\operatorname{Cov}(c)
=\operatorname{rank}\operatorname{Cov}(h)+\operatorname{rank}\operatorname{Cov}(e).$$

若两块协方差正定，则该秩为 $d+r$；环境维度为 $D+r$，所以零空间维度恰为 $D-d$。固定 $h$ 后，条件协方差秩为 $r$。

**证明。** 根据定义：

$$ML=M M^{-1}\begin{bmatrix}\hat Q\\0\end{bmatrix}
=\begin{bmatrix}\hat Q\\0\end{bmatrix},$$

故

$$cL=[h\hat P,e]\begin{bmatrix}\hat Q\\0\end{bmatrix}
=h\hat P\hat Q=h.$$

同理 $MJ=[0;I_r]$，所以 $cJ=e$。

令 $S=\operatorname{diag}(\hat P,I_r)$，其形状是 $(d+r)\times(D+r)$，且因为 $\hat P$ 满行秩，$\operatorname{rank}(S)=d+r$。又有 $c=[h,e]SM$。独立性使 $[h,e]$ 的协方差为 $\Sigma=\operatorname{diag}(\Sigma_h,\Sigma_e)$，因此

$$\operatorname{Cov}(c)=M^TS^T\Sigma SM.$$

$M$ 可逆不改变秩。映射 $S^T$ 是单射，因为 $S$ 满行秩，所以

$$\operatorname{rank}(S^T\Sigma S)=\operatorname{rank}(\Sigma)
=\operatorname{rank}(\Sigma_h)+\operatorname{rank}(\Sigma_e).$$

两块正定时得到 $d+r$，从 $D+r$ 减去它得到 $D-d$。固定 $h$ 后随机部分是 $eA_e$；$A_e$ 有 $r$ 个线性独立行，因此正定噪声协方差给出秩 $r$。证毕。

**推论。** 增大 $r$ 会同时将协方差秩和环境维度增加 $r$，因此无法改变零空间维度 $D-d$。它打破同一输入对应同一可见向量的确定性，却不能以“满秩”证明抵抗 SVD。有限样本中心化协方差的秩还不超过样本数减一。

### P2. 补空间与拼接双增广的满秩条件及等价性

**命题。** 给定同一 $\hat P,\hat Q$。令 $k=D-d$，$B\in\mathbb R^{k\times D}$ 的行是 $\ker(\hat Q^T)$ 的基，则 $T=[\hat P;B]$ 可逆。对补空间噪声 $a\in\mathbb R^{1\times k}$ 和拼接噪声 $e\in\mathbb R^{1\times r}$，令

$$F=\operatorname{diag}(T,I_r),\qquad
c_{full}=[h\hat P+aB,e]M=[h,a,e]FM.$$

则 $FM$ 可逆，而且

$$c_{full}M^{-1}\begin{bmatrix}\hat Q\\0\end{bmatrix}=h.$$

若 $h,a,e$ 相互独立且协方差均正定，则 $\operatorname{Cov}(c_{full})$ 在 $D+r$ 维环境中满秩。

**证明。** $\hat Q$ 秩为 $d$，因此 $\ker(\hat Q^T)$ 的维度为 $D-d=k$，$B$ 存在且 $B\hat Q=0$。若行向量 $[y,z]T=0$，右乘 $\hat Q$ 得 $y=0$；随后 $zB=0$，由 $B$ 满行秩得 $z=0$。方阵 $T$ 没有非零左零向量，故可逆；$F$ 和 $M$ 也可逆。

读取时：

$$c_{full}M^{-1}\begin{bmatrix}\hat Q\\0\end{bmatrix}
=(h\hat P+aB)\hat Q=h.$$

令 $v=[h,a,e]$。独立性使 $\operatorname{Cov}(v)$ 为三个正定块组成的正定块对角矩阵，维度为 $d+k+r=D+r$。因为 $c_{full}=vFM$ 且 $FM$ 可逆，合同变换保持正定性和秩，故 $\operatorname{Cov}(c_{full})$ 满秩。证毕。

**稳定构造。** 取一个 $D\times D$ 正交矩阵 $O$ 的前 $d$ 行为 $\hat P$、后 $k$ 行为 $B$，并取 $\hat Q=\hat P^T$，则 $T=O$，因而 $F$ 也正交，$\kappa_2(FM)=\kappa_2(M)$。任意 AloePri 右逆不保证这个性质，必须检查 $T$ 和 $FM$。

若只使用 $k'<D-d$ 个补空间噪声方向，则总体秩至多 $d+k'+r<D+r$，不能环境空间满秩。满秩只消除了固定零空间这一统计特征，不证明保密性；P13 的读取映射一旦可从云端可见参数恢复，补空间噪声会被 $B\hat Q=0$ 精确消去。

**线性闭环推论。** 将 $q=[a,e]$、$M_{eff}=FM$。对 $h'=hW+b$、$q'=hC_q+qG_q+\xi_q$，P3 的基本块恒等式应用于可逆基 $M_{eff}$，得到

$$\widetilde W=M_{eff,i}^{-1}
\begin{bmatrix}W&C_q\\0&G_q\end{bmatrix}M_{eff,o},$$

其输出是 $[h',q']M_{eff,o}=[h'\hat P_o+a'B_o,e']M_o$。因此双增广不需要新的非线性理论，只需在转换层把总辅助态分成补空间噪声和拼接链式噪声。

### P3. 扩维增广坐标中的矩形线性层与仿射刷新

**命题。** 输入语义/扩维/噪声维度为 $(d_i,D_i,r_i)$，输出为 $(d_o,D_o,r_o)$。设 $\hat P_i\hat Q_i=I_{d_i}$、$\hat P_o\hat Q_o=I_{d_o}$。取

$$W\in\mathbb R^{d_i\times d_o},\ C\in\mathbb R^{d_i\times r_o},\ G\in\mathbb R^{r_i\times r_o},$$
$$K=\begin{bmatrix}
\hat Q_iW\hat P_o&\hat Q_iC\\
0&G
\end{bmatrix},\quad \widetilde W=M_i^{-1}KM_o.$$

则 $c_i=[h\hat P_i,e]M_i$ 满足

$$c_i\widetilde W+[b\hat P_o,\xi]M_o
=[(hW+b)\hat P_o,hC+eG+\xi]M_o.$$

**证明。** 消去相邻的 $M_iM_i^{-1}$ 后，左上块给出

$$h\hat P_i\hat Q_iW\hat P_o=hW\hat P_o,$$

右上块给出 $h\hat P_i\hat Q_iC+eG=hC+eG$。再加 $[b\hat P_o,\xi]M_o$ 即得结论。批量计算逐行应用，因此逐请求逐位置的不同 $\xi$ 也成立。证毕。

fixed_debug 将固定 $\xi$ 合并进扩维增广 bias；per_request 在线产生 $\xi M_{o,e}$。若直接对混合坐标注入任意 $\zeta\in\mathbb R^{D_o+r_o}$，语义误差为 $\zeta L_o$；只有它为零时保持 exact。

### P4. 噪声稳定性的成立范围

**命题 A。** 对固定 $r$、$G_\ell=\gamma P_\ell$，其中 $P_\ell$ 为符号置换、$0<\gamma<1$，若沿给定轨迹 $\|h_\ell\|_2\le H_{max}$、$\|C_\ell\|_2\le c_{max}$、$\|\xi_\ell\|_2\le X_{max}$，则

$$\|e_L\|_2\le\gamma^L\|e_0\|_2+(H_{max}c_{max}+X_{max})\frac{1-\gamma^L}{1-\gamma}.$$

**证明。** 符号置换保范数，故每步有 $\|e_{\ell+1}\|_2\le\gamma\|e_\ell\|_2+H_{max}c_{max}+X_{max}$。递推展开，有限几何级数给出结论。证毕。

**命题 B。** 对 $C=0$、各 $P_\ell$ 离线固定，若 $e_0$ 零均值且协方差 $\sigma^2I_r$，各步 $\xi_\ell$ 零均值、独立于先前噪声，协方差 $(1-\gamma^2)\sigma^2I_r$，则每步噪声均零均值且协方差 $\sigma^2I_r$。

**证明。** 均值递推为零。独立性使交叉协方差为零，故下一步协方差为 $G_\ell^T\sigma^2I_rG_\ell+(1-\gamma^2)\sigma^2I_r=\sigma^2I_r$。由初始条件归纳成立。证毕。

这只是二阶矩保持，不保证噪声全分布平稳，更不覆盖 attention 左乘、非零耦合、残差重复相加或 clipping。均匀初始噪声与均匀刷新一般不会保留同一个均匀分布。高斯噪声只有概率尾界，没有有限确定上界 $X_{max}$；有界均匀刷新才可直接用于命题 A。

### P5. 非线性检查点和跨基残差

**命题。** 给定确定性语义函数 $F$，定义参考检查点

$$\mathcal C_F(c)=[F(cL)\hat P_o,\Phi(cL,cJ,\xi)]M_o.$$

只要 $\Phi$ 输出规定的噪声宽度，则 $\mathcal C_F(c)L_o=F(h)$。对相同隐藏/噪声维度、但输入输出混合基不同的残差，有

$$R_{i\to o}=\begin{bmatrix}\hat Q_i\hat P_o&0\\0&I_r\end{bmatrix},$$
$$cM_i^{-1}R_{i\to o}M_o
+[F(h)\hat P_o,hC+e(G-I)+\xi]M_o$$
$$=[(h+F(h))\hat P_o,hC+eG+\xi]M_o.$$

**证明。** 第一式用 P1 读取出 $h,e$，最终利用 $\hat P_o\hat Q_o=I$ 且消去辅助分量。第二式第一项为

$$[h\hat P_i,e]R_{i\to o}M_o=[h\hat P_i\hat Q_i\hat P_o,e]M_o=[h\hat P_o,e]M_o.$$

与增量逐块相加后，信号成为 $(h+F(h))\hat P_o$，噪声中的 $+e$ 与 $-e$ 抵消。证毕。

这个存在性构造可能需要参考 eager 运算或指定融合操作；它不是低开销实现证明，也不赋予该操作可信执行能力。直接相加两个不同基的混合态不满足上述公式。若隐藏维度不同，必须明确原模型的残差投影，不能采用此恒等残差式。

### P6. RMSNorm 的合法正交表示、gamma 吸收及分母扰动

定义 $\rho(h)=\sqrt{\|h\|_2^2/d+\epsilon_{rms}}$。设 $R\in\mathbb R^{d\times d}$ 正交、$u=hR$、$\Gamma=\operatorname{diag}(\gamma_{learned})$，则

$$\frac{u}{\rho(u)}R^T\Gamma W+b=\frac{h}{\rho(h)}\Gamma W+b.$$

**证明。** $\|hR\|_2^2=hRR^Th^T=\|h\|_2^2$，因此分母一致；再由 $RR^T=I_d$ 消去旋转。偏置保持在除法之外。若投影结果还需右乘矩阵 $U$，应同时变换权重和偏置，得到 $bU$。证毕。

直接对 $[h\hat P,e]$ 或 $c$ 的全部 $D+r$ 个坐标归一化，分母涉及扩维尺度、$\|e\|^2$ 和 $M$ 的几何，一般不同于 $\rho(h)$。一般 $\Gamma$ 与 $R$ 不交换，因此不能把 $\Gamma$ 随意移过 $R$。

另设分母被扰动为 $\rho+\delta$，$|\delta|\le b_\rho<\sqrt{\epsilon_{rms}}$。对分子 $t=h\Gamma W$：

$$\left\|\frac{t}{\rho+\delta}-\frac{t}{\rho}\right\|_2
\le\frac{\|t\|_2b_\rho}{\rho(\rho-b_\rho)}.$$

证明来自分式相减及 $\rho+\delta\ge\rho-b_\rho>0$。当 $t\ne0$ 时相对误差不超过 $b_\rho/(\rho-b_\rho)$；$t=0$ 时绝对误差为零。该分母扰动属于 approximate，不能在 exact 名义下启用。

### P7. RoPE 后 Q/K、GQA、Value 和 mask

对 query head $a$，令 $j=\nu(a)$ 为它共享的 KV head。先得到包含正确 bias 的合法投影并施加原模型 RoPE，记为 $Q_a^r,K_j^r$。选每个 KV 组一个正交矩阵 $U_j$，定义

$$Q'_a=Q_a^rU_j,\quad K'_j=K_j^rU_j.$$

则 $Q'_a(K'_j)^T=Q_a^r(K_j^r)^T$。

**证明。** 将转置展开，中间因子 $U_jU_j^T=I$。同组所有 query 使用同一个 $U_j$ 才能对复用的 key 成立；这里没有假设 $U_j$ 与 RoPE 交换。证毕。

在同一有效 mask 上，除以相同 $\sqrt{d_h}$ 后得到相同的有效 logits，Softmax 相同。令 $A_a$ 为该分布，$c_{V,j}=[V_j,E_j]M_{V,j}$，则

$$A_ac_{V,j}=[A_aV_j,A_aE_j]M_{V,j}.$$

证明是矩阵乘法结合律和分块分配律。读取信号得到 $A_aV_j$，各 query head 按明文顺序拼接并经过相同 output projection/bias，得到相同 attention 更新。噪声可经线性映射汇入隐藏噪声，但不得污染信号。

可选 key 位置置换 $P_t$ 必须同时施加到 logits、mask 和 Value：$\operatorname{softmax}(SP_t)P_t^Tc_V=\operatorname{softmax}(S)c_V$。共同有限行偏置在非空有效支持上也会消去；无效位置继续是 $-\infty$。全 mask 行由两路径共同约定为零输出，不能把未定义的 $0/0$ 视为 Softmax 定理。

缓存成立需每个 key/value 按其绝对位置生成一次并保存。给旧 Value 在每次读取时重新采样会改变噪声语义及重放结果，因此不属于本命题。

### P8. SwiGLU 与路由排列

令 gate/up 包含各自 bias，记为 $g,u\in\mathbb R^{d_f}$，$z=\operatorname{SiLU}(g)\odot u$。取置换矩阵 $P_f$ 和可逆有界对角矩阵 $D_f$：

$$g'=gP_f,\quad u'=uD_fP_f,\quad W_d'=P_f^TD_f^{-1}W_d.$$

则

$$[\operatorname{SiLU}(g')\odot u']W_d'+b_d=zW_d+b_d.$$

**证明。** 逐坐标函数与置换交换；共享置换保持乘积对应关系。因此中间乘积等于 $zD_fP_f$。右乘 $P_f^TD_f^{-1}W_d$ 消去置换和缩放，最后加相同 $b_d$。gate 不做任意缩放，证明没有使用 SiLU 的齐次性。证毕。

可将旁路旧噪声 $e_{side}$ 刷新为 $e_z=z'C_z+e_{side}G_z+\xi_z$ 再混合。它不改变上述信号，只要遵守 P3/P5。

对 MoE，物理专家、router 列和 tie 优先级同时按同一专家置换变换，无 router 噪声时被选专家及其权重对应不变，故加权输出相同。形式上固定原模型每个专家的唯一 tie 优先级，按“分数降序、优先级升序”排序；排列后每个专家携带自己的优先级，因此总序在逆排列下相同。若优先级本身公开，计入云端泄漏，不能隐藏其可见性。仅在集合不变而 gate 权重也被扰动时，不能声称 exact 输出。router 噪声的集合保持条件由 P10 作用于专家分数得到。

### P9. exact 多层 Transformer 与缓存推理

**命题。** 初始化为 $c_0=[h_0\hat P_0,e_0]M_0$，各模块满足 P3/P5–P8、两路径使用同一 token 前缀和缓存内容语义、所有语义扰动为零。则每层混合态均可写为 $c_\ell=[h_\ell\hat P_\ell,e_\ell]M_\ell$，其解码信号等于明文隐藏态；终端归一化与 LM head 亦按同一权重计算时，合法方得到相同 logits。

**证明。** 第零层由 P1 成立。假设第 $\ell$ 层入口信号相同：P6 给出相同的归一化投影语义；P7 给出相同 attention 更新；P5 给出相同残差；第二次归一化和 P8 给出相同 FFN 更新；第二次残差仍相同。于是第 $\ell+1$ 层入口信号相同。有限层归纳后，终端按相同 norm/head 映射得到相同 logits。若公开端只返回词表置换后的 logits，合法方先逆置换，再作比较和选词。证毕。

对缓存推理，按 token 位置再归纳：已缓存的 K/V 与相同前缀的全量计算在信号上相同；新 token 的 RoPE 绝对位置、mask、GQA 映射相同；P7 给出相同更新。辅助噪声逐位重放还要求按逻辑 sample/position 域分离，且保存旧缓存不重采样。浮点归约顺序不同可能导致数值偏差，实数归纳不保证位级相同。

对 expanded-full，令 $q_\ell=[a_\ell,e_\ell]$、$M_{eff,\ell}=F_\ell M_\ell$。P2 已给出 $c_\ell=[h_\ell,q_\ell]M_{eff,\ell}$ 和合法语义读取；P2 的线性闭环推论替代 P3，检查点把输出写成 $[F(h),q']M_{eff,o}$。上述逐层与逐 token 归纳逐字成立。因此 full 版本与 concat 版本在 exact 正确性上具有同样的存在性证明。

### P10. Top-k 集保持与边界情况

对一个有 $m>k\ge1$ 个有效分数的向量 $s$，令 $I$ 为 clean Top-k，$\Delta_k=s_{(k)}-s_{(k+1)}>0$。若 $\|\eta\|_\infty\le\tau$ 且 $2\tau<\Delta_k$，则扰动前后的 Top-k 集相同。

**证明。** 对每个 $i\in I,j\notin I$，有 $s_i-s_j\ge\Delta_k$。扰动后差值至少 $\Delta_k-2\tau>0$，故所有集合内元素仍严格高于集合外元素。证毕。

取 $\tau\le\alpha\Delta_k/2$、$0<\alpha<1$ 满足严格条件。它不保护集合内部排序或概率值。当 $\Delta_k=0$ 时按原 tie 规则取零扰动；当 $m\le k$ 时本方案亦取零扰动，避免无定义的第 $k+1$ 项。free-bounded 不作集合保证。FP32 中还须验证实际加法和比较：数学上的严格不等式可能被舍入抹去，不能只检查配置中的 $\alpha<1$。

### P11. Softmax 局部分布与 Value 输出误差

**命题。** 固定非空有效支持，令 $p=\operatorname{softmax}(s)$，$q=\operatorname{softmax}(s+\eta)$，$\|\eta\|_\infty\le\tau$，则

$$\|q-p\|_1\le\min(2,\tau),\qquad
D_{KL}(p\|q)\le\tau^2/2,\quad D_{KL}(q\|p)\le\tau^2/2.$$

**步骤 1：路径导数。** 令 $p_i(t)=\exp(s_i+t\eta_i)/\sum_j\exp(s_j+t\eta_j)$，$t\in[0,1]$，$\mu_t=\sum_i p_i(t)\eta_i$。直接求导：

$$p_i'(t)=p_i(t)(\eta_i-\mu_t).$$

加权 Cauchy–Schwarz 给出

$$\sum_i|p_i'(t)|\le\sqrt{\sum_i p_i(t)(\eta_i-\mu_t)^2}\le\tau.$$

最后一步因 $\mathbb E_{p(t)}\eta^2\le\tau^2$ 且方差减去非负均值平方。对路径积分，用三角不等式得 $\|q-p\|_1\le\tau$；两个概率向量的 L1 距离又不超过 $2$。

**步骤 2：KL。** 定义 $f(t)=\log\sum_i p_i\exp(t\eta_i)$。则 $f(0)=0$、$f'(t)=\mu_t$、$0\le f''(t)=\operatorname{Var}_{p(t)}(\eta)\le\tau^2$。由 $q_i=p_i\exp(\eta_i-f(1))$：

$$D_{KL}(p\|q)=f(1)-f'(0)=\int_0^1(1-t)f''(t)\,dt\le\tau^2/2,$$
$$D_{KL}(q\|p)=f'(1)-f(1)=\int_0^1t f''(t)\,dt\le\tau^2/2.$$

**步骤 3：Value。** 令 $v_i$ 为每个有效位置的 Value，$D_V=\max_{i,j}\|v_i-v_j\|_2$。设 $a_i=q_i-p_i$；因为 $\sum_i a_i=0$，正部分和负部分的总质量均为 $t_a=\|q-p\|_1/2$。若 $t_a=0$，输出误差为零；否则将正负部分分别归一化为概率 $\alpha_i,\beta_j$：

$$\left\|\sum_i a_iv_i\right\|_2=t_a\left\|\sum_i\alpha_iv_i-\sum_j\beta_jv_j\right\|_2
\le t_a\sum_{i,j}\alpha_i\beta_j\|v_i-v_j\|_2\le t_aD_V.$$

因此 $\|(q-p)V\|_2\le(D_V/2)\min(2,\tau)$。证毕。

若 $D_V\le D_{cap}>0$ 且 $\tau\le2\varepsilon_O/D_{cap}$，则局部输出误差不超过 $\varepsilon_O$。$D_{cap}=0$ 意味所有有效 Value 相同，输出不受概率变化影响，无需该项限制。可以用 $D_{cap}=2\max_i\|v_i\|_2$ 作当前行的保守上界；若需要计算不可见信号，应在指定检查点内完成或使用有认证的离线上界，不因此公开 Value 范数。

对多 head 拼接，若每 head 的误差上界为 $\varepsilon_a$，输出投影后的误差不超过 $\|W_O\|_2\sqrt{\sum_a\varepsilon_a^2}$。这由拼接向量的平方范数相加以及谱范数定义得到。比较的 Value 与状态输入必须相同；不同层累计漂移由 P12 单独处理。

### P12. 多层传播与 greedy 间隔条件

**命题。** 将固定长度前缀的完整状态（必要时包括缓存）记为 $x_\ell$。设明文层 $F_\ell$ 在包含两条轨迹的区域上是 $L_\ell$-Lipschitz，扰动层满足同输入局部界 $\|\widehat F_\ell(x)-F_\ell(x)\|\le\varepsilon_\ell$。记录浮点及实现误差的额外界 $\nu_\ell$，则

$$\delta_{\ell+1}\le L_\ell\delta_\ell+\varepsilon_\ell+\nu_\ell,$$
$$\delta_L\le\delta_0\prod_{j=0}^{L-1}L_j+
\sum_{i=0}^{L-1}(\varepsilon_i+\nu_i)\prod_{j=i+1}^{L-1}L_j.$$

**证明。** 在 $\widehat F_\ell(\widehat x_\ell)-F_\ell(x_\ell)$ 中加减 $F_\ell(\widehat x_\ell)$，依次应用局部扰动界、Lipschitz 界和额外数值界。展开递推得到乘积和，空乘积定义为 $1$。证毕。

不能将未测定的 $L_\ell,\nu_\ell$ 默认为 $1,0$。RMSNorm 的正 epsilon 有助于局部有限导数；attention 的双线性项仍需有界状态域。若包含硬 Top-k MoE，跨专家选择边界一般不能直接假设 Lipschitz，需限制在选择集固定且有严格 margin 的区域，或另给跳变项。上述条件式不是已认证的整网数值上界。

若在同一 token 前缀上两组 logits 满足 $\|\widehat z-z\|_\infty\le\varepsilon_z$，且明文最大与次大 logit 间隔 $\Delta_z>2\varepsilon_z$，则 greedy token 相同：证明用 P10 的 $k=1$。如果每步在此前相同的前缀上均满足条件，则按生成位置归纳，整段 greedy 序列一致。任何一步前缀分叉之后，不能继续用原前缀的逐步界证明序列一致。边界并列必须使用相同 tie 规则另验。

### P13. 三类条件性泄漏；满秩不是保密性

**A：公开投影反解。** 若云端已知 $D_q=LB_q$ 与公开 $B_q\in\mathbb R^{d\times m}$，后者满行秩且列坐标对应已知，存在右逆 $B_q^+$ 使 $B_qB_q^+=I_d$。因此

$$D_qB_q^+=L,\qquad cD_qB_q^+=h.$$

这只是乘法结合律。若 $B_q$ 秩不足、存在未知输出变换或 bias 未分离，该特定恢复公式不直接成立；不据此推断整体安全。改变 $M$ 的分布或增大 $e$ 幅度不影响满足前提时的恢复。

**B：可见噪声映射。** 写 $c=hA_h+eA_e$，其中 $A_h=[\hat P,0]M$、$A_e=[0,I_r]M=M_e$。若 $A_e$ 已知，取 $V\in\mathbb R^{(D+r)\times D}$ 的列为空间 $\ker(A_e)$ 的基，则

$$cV=hA_hV,$$

且 $A_hV$ 的行秩是 $d$。

证明：$A_eV=0$，噪声项消失。若某个行向量 $y$ 满足 $yA_hV=0$，则 $yA_h$ 消灭 $\ker(A_e)$，所以 $yA_h$ 属于 $A_e$ 的行空间。另一方面，$A_h$ 与 $A_e$ 的全部行来自满行秩矩阵 $\operatorname{diag}(\hat P,I_r)M$，两者行空间相交仅有零向量。因此 $yA_h=0$，而 $A_h$ 满行秩又推出 $y=0$。故 $A_hV$ 满行秩 $d$。投影保留全部语义自由度，但未知的满行秩映射仍需对齐，不等于 token 已被恢复。

对 expanded-full，写 $c_{full}=hH+qA$，其中 $q=[a,e]$，$[H;A]=M_{eff}$ 是可逆方阵。若 $A$ 已知，取其零空间基 $V\in\mathbb R^{(D+r)\times d}$，则 $c_{full}V=hHV$，并且 $HV$ 是 $d\times d$ 可逆矩阵。证明与上段相同，只是完整 $M_{eff}$ 已满秩。因此 full 消除了协方差固定零空间，却没有在辅助映射可见时阻止噪声消除；若 $H$ 或等价公开权重还能被求出，则 $h=c_{full}V(HV)^{-1}$。

**C：可见 Gram。** 若 $A=LL^T$ 对云端可见，则

$$c_iAc_j^T=(c_iL)(c_jL)^T=h_ih_j^T.$$

于是信号的范数、内积和成对距离可由普通可见量得到。无需知道 $L$ 的某个具体分解。是否足以辨认 token 取决于候选字典、噪声/舍入精度和观测单位，须实验验证。

这三条都是具备前提时的直接代数结论，并不是所有协议的不可能性定理。另一方面，合法方持有 $M^{-1}$ 能解码本身也不是 COA 攻击；必须证明攻击者能从允许信息得到等价能力。

### P14. 首层 RMSNorm 切分与范数残留

**命题 A。** pre-norm block 的第一条残差为 $h_1=h_0+\operatorname{Attention}(\operatorname{RMSNorm}(h_0))$。只给云端归一化输出，而不定义原始残差的保存、恢复或端侧合并，不能直接替代完整 block 输入。

该结论来自模型定义：残差需要 $h_0$。例如在理想化 $\epsilon_{rms}=0$ 且正缩放时，$h_0$ 与 $2h_0$ 有相同归一化输入，但残差输出相差 $h_0$，因此归一化输入不能确定残差。真实正 epsilon 下，归一化可能可逆；若靠此恢复残差，又必须计入相应泄漏，不能声称已不可逆移除幅度。

**命题 B。** 对 gamma-free $u=h/\sqrt{\|h\|^2/d+\epsilon_{rms}}$，令 $a=\|h\|^2/d$，$q=\|u\|^2/d$，则

$$q=\frac{a}{a+\epsilon_{rms}}<1,\qquad a=\frac{\epsilon_{rms}q}{1-q}.$$

直接平方并解一次方程得证。若 $h=0$，$q=a=0$ 仍成立。该反推要求可见 $q$ 和 epsilon；$q$ 接近 $1$ 时对有限精度敏感，不能从实数恒等式直接报告攻击恢复率。已知且非零的学习 gamma 可先逆缩放；未知 gamma 或零分量需另分析。

### P15. 工作因子由最便宜攻击决定

固定方案 $s$、观察视图 $v$、目标数 $m$、恢复指标 $R$ 和目标 $\rho$。令允许攻击集合为 $\mathcal A_v$，标量参考成本为 $C_A\ge0$，定义

$$W_{s,v}(\rho,m)=\inf_{A\in\mathcal A_v}
\{C_A:R_A(m)\ge\rho\}.$$

若不存在达到目标的攻击，扩展实数约定下 $W=+\infty$。

**命题 A：已构造攻击给出工作因子上界。** 若某个允许攻击 $A_0$ 达到 $R_{A_0}\ge\rho$，则

$$W_{s,v}(\rho,m)\le C_{A_0}.$$

**证明。** $A_0$ 属于取下确界的可行集合；集合的下确界不大于任一元素。证毕。

因此，只要 P13 的右逆、Gram 或辅助空间投影有一条成本很低，昂贵的深度反演攻击不能提高该方案的工作因子。

**命题 B：有限攻击集合会乐观估计安全性。** 对实验攻击集合 $\mathcal S_v\subseteq\mathcal A_v$，定义

$$\widehat W_{s,v}(\rho,m)=\inf_{A\in\mathcal S_v}
\{C_A:R_A(m)\ge\rho\}.$$

则

$$W_{s,v}(\rho,m)\le\widehat W_{s,v}(\rho,m).$$

**证明。** 在较大集合上取下确界不会大于在其子集上取下确界。证毕。

所以实验能证明的准确说法是“对攻击集合 $\mathcal S_v$，观测到工作因子为多少”，而不是对所有攻击的复杂度下界。

**命题 C：key epoch 的摊销公式。** 假设某攻击在每个独立 key epoch 必须重新支付 $C_{setup}$，跨 epoch 产物不能迁移；每个目标另支付 $C_{target}$。对 $E$ 个 epoch、总计 $m$ 个目标：

$$C_A(E,m)=E C_{setup}+mC_{target},\qquad
\overline C_A(E,m)=\frac{E C_{setup}}m+C_{target}.$$

**证明。** 在不可迁移假设下，各 epoch 的准备任务不能共享，成本可加；逐目标任务也可加。总和除以 $m$ 得摊销成本。证毕。

若攻击器或读取矩阵可以跨 epoch 复用，第一项不能重复计算。因此密钥轮换提升攻击成本是条件命题，必须用跨密钥迁移实验验证该前提。

相对基线 $b$ 的倍率

$$G_{s/b,v}(\rho,m)=W_{s,v}(\rho,m)/W_{b,v}(\rho,m)$$

只在相同视图、恢复指标、目标数、硬件成本口径和攻击能力下有意义。$G>1$ 表示攻击成本更高，但不是密码学安全证明。

## Corrections or Missing Assumptions

- 原“全域满秩”修正为 P1 的秩 $d+r$、零空间维度 $D-d$；P2 的补空间填充仅作对照。
- 原残差公式补上 $\hat Q_i\hat P_o$ 和联合混合基变换；原线性式用明确的输入/输出扩维宽度和偏置位置替代未定义的 `M_next_top`。
- exact 是语义恒等式；浮点容差和模型适配器正确性由独立门禁负责。
- 全网络误差界与序列一致只在 P12 前提下成立，局部 Softmax 界不自动决定任务精度。
- 安全目标修正为 P15 的预算化工作因子；有限攻击实验只能给出真实最小攻击成本的乐观上界。
- 安全方面证明了必须排查的条件性泄漏，未证明 v2 能抵抗 cloud COA。不能按用户希望的结论反向选择前提。

## Open Risks

尚未给出：可抵抗所有可见部署参数联合分析的安全归约；数值认证的整网 Lipschitz/舍入常数；真实语料多密钥攻击收益；性能实测；安全的低交互端云切分协议。数学检查只验证若干有限实例，不替代以上证明，也不是机器形式化证明。
