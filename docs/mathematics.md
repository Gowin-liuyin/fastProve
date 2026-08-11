# fastProve 数学实现说明

## 1. 文档范围与当前状态

本文给出 fastProve 参考原型所实现的数学约定、转换公式和数值边界。
它说明四种推理模式为何能够在同一组基础权重上比较，但不提供尚未运行的
预训练模型实验结果。

当前代码是 PyTorch eager 参考实现。它可以验证代数恒等式、掩码语义、
确定性噪声和端到端 Tiny Causal LM 的实现正确性；它不是已经融合或经过
密码学证明的部署内核。非线性检查点在参考实现内部会临时恢复合法信号，
其安全含义见 [threat_model.md](threat_model.md)。

## 2. 统一记号与权重布局

全文采用行向量约定：

\[
Y=XW+b.
\]

若 \(X\in\mathbb R^{n\times d_{\rm in}}\)，则数学布局的权重为

\[
W_{\rm math}\in\mathbb R^{d_{\rm in}\times d_{\rm out}}.
\]

PyTorch `torch.nn.functional.linear` 存储的权重布局为

\[
W_{\rm pt}=W_{\rm math}^{T}
\in\mathbb R^{d_{\rm out}\times d_{\rm in}},
\]

因此

\[
\operatorname{linear}(X,W_{\rm pt},b)
=XW_{\rm pt}^{T}+b
=XW_{\rm math}+b.
\]

转换代码始终显式区分这两种布局。离线增广矩阵运算使用数学布局，返回给
`F.linear` 前才转置为 PyTorch 布局。非方阵往返也必须保形，防止用
“恰好是方阵”的测试掩盖转置错误。

## 3. 确定性种子与域分离

一次请求由 `RequestContext` 标识。基础种子与有类型的域标签共同派生
各层、各用途的子种子，例如：

```text
request / layer / operation / head / query-position / key-position
```

相同输入、基础种子和域标签生成相同的变换或噪声；任一域分量不同都应生成
不同的随机流。实现拒绝无序集合等容易产生歧义的域对象，避免两个语义不同
的调用意外共享随机流。

这种设计的目的仅是可复现和避免跨域复用。SHA-256 派生后使用的 PyTorch
伪随机发生器不是本项目中的密码学密钥流证明；固定种子也不提供前向安全、
不可预测性或差分隐私。

## 4. 增广混合状态

对第 \(\ell\) 层，语义信号和辅助噪声分别为

\[
h_\ell\in\mathbb R^{d_\ell},
\qquad
e_\ell\in\mathbb R^{r_\ell}.
\]

先增广，再右乘可逆基：

\[
z_\ell=[h_\ell,e_\ell],
\qquad
c_\ell=z_\ell M_\ell.
\]

生产路径传递的是 `MixedState(c, basis_descriptor)`。描述符只保存信号维度、
噪声维度、条件数和指纹，不携带逆矩阵。完整 `BasisTransform` 属于转换端
或显式调试端；`encode_debug` 和 `decode_debug` 必须由调试开关授权，并验证
描述符指纹及信号/噪声分区一致。

### 4.1 受控条件数的混合基

原型生成

\[
M=\Pi D B,
\]

其中 \(\Pi\) 为排列矩阵，\(D\) 为有界非零对角缩放，\(B\) 为正交矩阵。
排列和正交因子的 2-范数条件数为 1，因此精确算术下条件数主要由 \(D\)
控制。默认上限为可配置的

\[
\kappa_2(M)\le 10.
\]

生成、条件数检查和求解逆矩阵在 CPU FP64 离线完成；当前基生成器只接受
FP32 或 FP64 部署矩阵。模块执行时可按已验证的激活 dtype 使用副本，但
BF16 基生成本身尚未开放。FP16、整数或复数变换也不在当前生成器支持范围
内。逆矩阵只在转换/调试边界生成，`forward` 不调用 `inverse`、`inv`
或 `solve`。

“矩阵稠密”或“元素非零”既不等价于条件良好，也不构成安全证明。

## 5. ChainLinear 与仿射噪声刷新

设明文仿射层为

\[
y=hW+b,
\]

并定义输出辅助状态

\[
e'=hC+eG+\xi,
\]

其中

\[
W\in\mathbb R^{d_{\rm in}\times d_{\rm out}},
\quad
C\in\mathbb R^{d_{\rm in}\times r_{\rm out}},
\quad
G\in\mathbb R^{r_{\rm in}\times r_{\rm out}},
\quad
\xi\in\mathbb R^{r_{\rm out}}.
\]

构造数学布局的块矩阵

\[
K=
\begin{bmatrix}
W&C\\
0&G
\end{bmatrix}.
\]

离线转换为

\[
\widetilde W_{\rm math}
=M_{\rm in}^{-1}KM_{\rm out},
\qquad
\widetilde b=[b,\xi]M_{\rm out}.
\]

于是：

\[
\begin{aligned}
c_{\rm out}
&=c_{\rm in}\widetilde W_{\rm math}+\widetilde b\\
&=[h,e]M_{\rm in}M_{\rm in}^{-1}KM_{\rm out}
  +[b,\xi]M_{\rm out}\\
&=[hW+b,\;hC+eG+\xi]M_{\rm out}.
\end{aligned}
\]

这条恒等式是线性内核的直接验收条件。部署时保存
\(\widetilde W_{\rm pt}=\widetilde W_{\rm math}^{T}\)，所以
`ChainLinear.forward` 只执行 `F.linear`，再按刷新模式添加已经混合到
输出基中的请求刷新量。

稳定传播器可取

\[
G=\gamma P_e,\qquad 0<\gamma<1,
\]

其中 \(P_e\) 是排列或带符号排列。原型区分：

- `fixed_debug`：\(\xi\) 固定写入转换后的偏置，只用于恒等式和回归测试；
- `per_request`：从请求、层和用途域派生 \(\xi\)，在指定操作内部混合后添加。

固定刷新只保证确定性，不应描述为强隐私。状态字典还携带输入/输出基描述符；
缺失或不匹配的基元数据必须拒绝加载，避免将同形但不同基的权重误接。

## 6. 非线性检查点

一般的稠密增广变换不与 RMSNorm、Softmax、SiLU 或逐元素乘法交换。参考路径
在每个非线性检查点执行以下逻辑：

1. 在受控操作内部从混合状态取得合法信号和旧辅助状态；
2. 保留旧辅助状态的旁路，不将它永久置零；
3. 只在信号上计算数学上有效的非线性；
4. 由信号耦合、旁路传播和仿射刷新构造新辅助状态；
5. 重新混合后返回 `MixedState`；
6. 生产接口只返回混合结果；干净检查点和概率仅由显式 debug API 返回。

当前 eager 实现中的“受控操作”是用于正确性验证的 Python 参考边界，不是
防止进程内攻击者读取临时量的安全边界。后续安全部署需要融合内核、TEE、
MPC、HE 或其他独立可信机制。

## 7. RMSNorm 与 gamma 吸收

定义不含学习缩放的 RMS 归一化：

\[
\operatorname{rmsnorm}_0(h)
=\frac{h}{\sqrt{\operatorname{mean}(h^2)+\epsilon}}.
\]

若 \(R\) 正交，令 \(u=hR\)，则

\[
\operatorname{mean}(u^2)=\operatorname{mean}(h^2),
\qquad
\operatorname{rmsnorm}_0(u)
=\operatorname{rmsnorm}_0(h)R.
\]

设学习缩放

\[
\Gamma=\operatorname{diag}(\gamma).
\]

\(\Gamma\) 通常不与 \(R\) 交换，所以不能直接把带 gamma 的 RMSNorm 当作
协变操作。若后续投影还允许一个输出侧变换 \(T_Q\)，一般吸收权重可写为

\[
W_Q'
=R^T\Gamma W_QT_Q,
\]

则

\[
\operatorname{rmsnorm}_0(u)W_Q'
=\operatorname{rmsnorm}_0(h)\Gamma W_QT_Q.
\]

当前 Q/K 路径为避免错误地假设 RoPE 与任意 \(C\) 交换，取
\(T_Q=T_K=I\)：先用 \(R^T\Gamma W_Q\) 与 \(R^T\Gamma W_K\) 完成
投影和 gamma 吸收，再应用 RoPE，最后才使用第 8 节的共同正交变换。
Value、Gate 和 Up 投影采用各自数学上有效的输出变换。RMS 平方、均值、
\(\epsilon\) 相加和开方全部在 FP32 中进行，再转回激活 dtype。

## 8. RoPE、Q/K 正交变换与 GQA

实现顺序固定为：

1. 从归一化信号得到 \(Q,K\)；
2. 应用 RoPE；
3. 对共享一组 KV 的 Query 头和该 KV 头应用共同正交矩阵 \(C_j\)。

对第 \(j\) 个 KV 组：

\[
Q_i'=\operatorname{RoPE}(Q_i)C_j,
\qquad
K_j'=\operatorname{RoPE}(K_j)C_j,
\]

其中 Query 头 \(i\) 映射到 KV 头 \(j\)。因为
\(C_jC_j^T=I\)，所以

\[
Q_i'K_j'^T
=\operatorname{RoPE}(Q_i)
  C_jC_j^T
  \operatorname{RoPE}(K_j)^T
=\operatorname{RoPE}(Q_i)\operatorname{RoPE}(K_j)^T.
\]

代码显式保存 Query-to-KV 的 `kv_index`，不能对共享同一 KV 头的 Query
随意使用不兼容的变换。原型不假设 \(C\) 与 RoPE 交换；把 \(C\) 提前到
RoPE 之前通常不成立。

## 9. 因果掩码与安全 Softmax

有效位置由 Query token 有效性、Key token 有效性和因果关系共同决定：

\[
\operatorname{valid}(q,k)
=m_q\land m_k\land (p_k\le p_q).
\]

掩码后无效 logit 保持精确的 \(-\infty\)。Softmax 的指数、归约和归一化
使用 FP32；全掩码行显式返回全零概率，避免
\(\operatorname{softmax}([-\infty,\ldots,-\infty])\) 产生 NaN。
原型不添加有限值 dummy 位置，因为它会改变归一化分母。

## 10. Value 增广协变

每个 KV 头的 Value 可携带低维辅助状态：

\[
c_V=[V,e_V]M_V.
\]

对任意注意力矩阵 \(A\)：

\[
Ac_V
=A[V,e_V]M_V
=[AV,Ae_V]M_V.
\]

因此注意力对 Value 的左乘保持同一右混合基。参考块在上下文检查点内取出
信号 \(AV\)，将注意力后的辅助分量与旧旁路、信号耦合和刷新项合并，再把
残差结果重新混合。

## 11. 四种注意力模式

四种模式共用相同的基础权重、Q/K/Value 处理、掩码和 Softmax 接口。

### 11.1 `plaintext`

\[
S=\frac{QK^T}{\sqrt{d_h}},
\qquad
A=\operatorname{softmax}(S),
\qquad
O=AV.
\]

这是明文参考基线。

### 11.2 `exact`

先对 RoPE 后的 Q/K 使用共同正交变换，并使用混合 Value：

\[
S'=\frac{Q'K'^T}{\sqrt{d_h}}=S,
\qquad
O_{\rm mixed}=A\,c_V.
\]

取其信号分量得到

\[
O_{\rm exact}=AV.
\]

没有 logit 噪声时，精确模式与明文路径的差异只应来自有限精度、运算顺序
和 dtype 转换。是否达到“仅浮点级误差”必须由记录的层级和端到端实验
确认，不能仅凭公式宣称。

可选的位置排列协变族为

\[
\widetilde S=SP+a\mathbf 1^T,
\qquad
\widetilde V=P^Tc_V,
\]

\[
\operatorname{softmax}(\widetilde S)\widetilde V
=\operatorname{softmax}(S)c_V.
\]

当前核心原型未把它作为必需路径；即使实现，它也只重标位置，不隐藏分数
多重集合或排序结构。

### 11.3 `approximate/topk_preserving`

只对有效 logit 添加噪声：

\[
\widehat S=S+\eta.
\]

在每个有效 Query 行上排序，定义 Top-k 边界：

\[
\Delta_k=S_{(k)}-S_{(k+1)}.
\]

预算为

\[
\tau
=\min\left(
\tau_{\max},
\frac{\alpha\Delta_k}{2},
\tau_{\rm error}
\right),
\qquad 0<\alpha<1.
\]

采样后满足

\[
\|\eta\|_\infty\le\tau.
\]

于是对 \(\Delta_k>0\)：

\[
2\|\eta\|_\infty
\le\alpha\Delta_k
<\Delta_k,
\]

所以 Top-k 集不变。若有效位置数不超过 \(k\)、边界打平或 margin 非有限，
预算设为零，并统计零噪声 Query 比例。这个保证针对 Top-k 集，不保证完整
排序不变。

### 11.4 `approximate/free_bounded`

不使用 margin 上限：

\[
\tau=\min(\tau_{\max},\tau_{\rm error}),
\qquad
\|\eta\|_\infty\le\tau.
\]

该模式允许 Top-k 和完整排序改变，用来显式研究“保持重要集合”和“施加更多
排序扰动”之间的冲突。

### 11.5 噪声生成与输出误差

噪声由请求、层、head、Query 位置和 Key 位置域确定性生成；仅有效坐标参与
居中和缩放，无效坐标保持零，因此加入后掩码位置仍为 \(-\infty\)。实际
无穷范数在指标中重新测量，不能用配置值代替实测值。

\[
\widehat A=\operatorname{softmax}(S+\eta),
\qquad
\widehat O=\widehat A V,
\]

\[
\widehat O-O
=\left[
\operatorname{softmax}(S+\eta)
-\operatorname{softmax}(S)
\right]V.
\]

生产 `forward` 只返回混合注意力输出。测试用 `forward_debug` 才能捕获
clean/noisy logits、概率、margin、预算和噪声，用于 KL、JS、Top-k overlap、
rank correlation 及输出相对误差。

## 12. SwiGLU 的同步置换与补偿

明文 SwiGLU 为

\[
g=xW_g,\qquad
u=xW_u,\qquad
z=\operatorname{SiLU}(g)\odot u.
\]

选择共同神经元排列 \(P_f\) 和有界非零对角缩放 \(D_f\)：

\[
g'=gP_f,
\qquad
u'=uD_fP_f.
\]

由于 SiLU 对排列逐坐标作用：

\[
\operatorname{SiLU}(g')\odot u'
=zD_fP_f.
\]

Down 权重转换为

\[
W_d'=P_f^TD_f^{-1}W_d,
\]

于是

\[
(zD_fP_f)W_d'=zW_d.
\]

Gate 和 Up 必须使用同一排列；只有 Up 承担 \(D_f\) 缩放，Down 再做逆缩放
补偿。该公式不声称 SiLU 与任意稠密变换或任意缩放交换。

非线性后辅助状态为

\[
e_z=z'C_z+e_{\rm side}G_z+\xi_z.
\]

旧辅助状态通过旁路进入 \(G_z\)，即使信号耦合为零，仿射刷新也能重新产生
非零辅助状态，避免旧的一维纯乘法方案“一旦清零便无法恢复”的缺陷。

## 13. Router 扩展

虽然当前首选 Qwen2 1.5B 是稠密 FFN，不使用 MoE router，原型仍提供 Router
数学内核。对专家排列 \(P_E\)：

\[
r'=rP_E+\eta_r.
\]

若

\[
2\|\eta_r\|_\infty
<r_{(k)}-r_{(k+1)},
\]

则按 \(P_E\) 还原后，选中专家集合与明文一致。物理专家必须按相同排列重排；
tie 使用稳定的规范专家编号打破。生产接口只返回决策，不返回 router logits；
调试接口才返回 margin、预算和噪声门权重。

## 14. Decoder Block 与 Tiny Causal LM

明文和混淆块共享一组基础权重并保持以下顺序：

```text
RMSNorm
→ Q/K/V
→ RoPE
→ causal attention
→ output projection
→ residual
→ RMSNorm
→ SwiGLU
→ down projection
→ residual
```

每个残差点同时更新信号和辅助旁路。Tiny Causal LM 再添加共享词嵌入、多个
Decoder Block、末端 RMSNorm 和 LM head。精确模式的输出 logits 与 greedy
token 应和明文一致到验收容差；一旦 greedy token 不同，必须先定位是否为
margin 极小、dtype、掩码、权重布局或变换条件数问题，再解释近似噪声结果。

随机 Tiny LM 只验证结构、确定性和端到端代数，不是语言能力证据。真实精度
结论必须来自同一预训练 checkpoint、tokenizer 和缓存样本的公平比较。

## 15. 数值策略与验收边界

- 离线变换、逆矩阵求解和条件数检查：CPU FP64；
- RMS 统计、QK logits、mask、margin、噪声裁剪和 Softmax reduction：FP32；
- 激活 dtype：可配置，首先验证 FP32，再单独验证 BF16；
- 所有路径统计 NaN/Inf，出现非有限值即失败；
- 不允许为通过测试而改变上述数学定义。

当前 FP32 小张量回归测试冻结的容差为：

| 检查点 | 最大允许误差或 `assert_close` 容差 |
|---|---:|
| ChainLinear signal max absolute error | \(10^{-5}\) |
| exact QK score max absolute error | \(5\times10^{-5}\) |
| exact Softmax probability max absolute error | \(5\times10^{-6}\) |
| exact attention output | `atol=rtol=5e-5` |
| exact 单块最终 signal | `atol=rtol=1e-4` |
| Tiny LM exact logits | `atol=rtol=2e-4` |

这些是随机 Tiny/小张量正确性阈值，不是预训练 Qwen、BF16 或其他硬件的
自动容差。真实模型适配必须先记录明文数值基线，再单独预注册合理阈值。

数学恒等式能证明理想算术下的信号等价和配置噪声上界；它不能替代有限精度
测量、预训练模型精度测量、攻击实验或密码学归约。
