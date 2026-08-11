# fastProve 威胁模型与安全限制

## 1. 结论先行

fastProve 是“增广协变混淆 + 有界 logit 扰动”的可复现研究原型，不是生产
加密系统。它目前最多评估：一个诚实执行既定代码、但会观察持久化张量和
普通框架输出的服务方，能看到怎样的混合表示以及精度如何变化。

以下事实都不构成密码学安全：

- Python API 没有返回 attention probability；
- 激活被右乘一个未知可逆矩阵；
- 辅助噪声维度非零；
- 噪声由带 SHA-256 域分离的种子确定性生成；
- exact 模式只出现浮点级误差；
- top-k-preserving 模式保持 Top-k；
- 将来使用 ML-KEM、LWE 派生 KDF 或 LWE 风格参数。

本项目没有 IND-CPA/IND-CCA、语义安全、差分隐私、MPC/HE 安全或端到端
机密性的证明。

更强的表述见第 5bis 节：当前实数可逆构造下 \(h=c(M^{-1})_{[:,:d]}\)
对任意噪声精确成立（实测 \(\sim10^{-15}\)），因此上述各项不仅"未被证明"，
而且在已知 \(M\) 或可凑已知明文对的攻击者面前**已被证否**。

## 2. 受保护对象

原型希望研究以下对象在既定接口下的表示暴露：

- 层间语义激活 \(h_\ell\)；
- Value 和 attention 输出的信号分量；
- 近似模式下的 clean Softmax logits、概率和部分排序信息；
- 客户端持有的混合基、请求种子和域标签。

不在当前保护目标内：

- 模型权重本身的机密性；
- 输入/输出 token、长度、批大小、调用时序和访存模式；
- 客户端设备被攻破后的密钥或基；
- 训练数据成员关系、模型反演或提示词泄露的通用防护；
- 拒绝服务、完整性、回滚或恶意输出。

## 3. 参与方与可信边界

### 3.1 客户端/转换端

客户端或离线转换端持有完整 `BasisTransform`、其逆矩阵、基础种子以及
debug encode/decode 能力。生产服务不应得到这些客户端秘密。

### 3.2 参考服务路径

服务路径接收 `MixedState`、转换后的线性权重和基描述符。普通生产 API
只返回混合状态或最终 logits，不返回解码信号、辅助状态、attention
probability 或 router logits。

`ChainLinear` 及 Transformer 参考服务模块不把输入/输出逆基注册为参数或
buffer，持久化 `state_dict`/`named_buffers` 不包含这些检查点基材料，且
`forward` 不在线求逆。这是针对普通持久张量观察面的接口隔离与数值工程
属性，但不是防止进程内攻击者恢复秘密的充分条件。

### 3.3 非线性检查点

RMSNorm、Softmax 和 SwiGLU 不能通过一般稠密基纯线性搬运。当前 PyTorch
参考块把检查点矩阵/逆矩阵捕获在 designated-operation 闭包中，不写入模块
持久状态；eager 执行仍会在该闭包内物化合法信号、clean logits 和 clean
probability，再丢弃生产路径不需要的 debug 数据。

因此当前代码只能代表“指定操作不会把内部临时值作为普通 API 结果返回”的
功能假设。它尚未实现一个可抵抗 hooks、调试器、进程内存读取或恶意 kernel
的融合可信边界。把 Python 属性命名为 private、闭包捕获矩阵或不注册 buffer
都不等价于硬件隔离。

## 4. 当前考虑的攻击者

初始实验采用诚实但好奇观察者：

- 按照仓库代码和配置执行推理；
- 可读取服务端持久化的混合激活、转换权重和常规日志；
- 可观察生产 API 的普通输入/输出；
- 不修改 kernel，不注入 hook，不读取客户端秘密；
- 不直接读取 designated operation 的临时寄存器或受控内部值。

在这个模型下，实验可以比较表示扰动、排序变化和任务精度，但仍不能仅凭
经验结果推出不可恢复性或计算安全。

## 5. 明确不抵抗的攻击能力

只要攻击者具备以下任一能力，当前 eager 原型不提供相应保护：

- 修改 PyTorch、Metal、CUDA、Triton 或自定义 kernel；
- 对模块注册 arbitrary forward/backward hook；
- 使用 Python introspection 读取私有属性、闭包或对象图；
- 连接调试器，转储进程内存、寄存器、临时 buffer 或统一内存；
- 同时读取服务端状态和客户端混合基/种子；
- 替换随机流、重复/选择请求或改变 mask；
- 对大量相关明文/混合状态做已知明文、选择明文或统计恢复；
- 从访问模式、序列长度、时延、内存占用、KV-cache 大小推断信息；
- 篡改结果、跳过刷新、回滚参数或返回恶意 token。

若威胁模型需要抵抗这些能力，必须引入独立可信边界，例如经过审计的 TEE、
MPC、HE、机密 GPU 或其他有明确安全定义的协议。

## 5bis. 已量化的可恢复性上界（实测）

本节把第 1 节的定性声明「可逆混合不构成密码学安全」替换为可复核的数字。
以下结论对**当前实数构造**成立，与噪声强度、刷新模式和密钥卫生无关。

### 5bis.1 精确线性解码恒等式

当前构造为 \(c=[h,e]M\)，其中 \(M=\Pi DB\) 为实数可逆矩阵
（`src/fastprove/transforms.py:184`），无模运算、无量化、无舍入。因此：

\[
h = c\,(M^{-1})_{[:,\,:d]}
\]

\((M^{-1})_{[:,\,:d]}\) 是一个**固定的** \((d+r)\times d\) 矩阵，它把 \(e\)
所在的 \(r\) 维完全湮灭。该恒等式对**任意** \(e\) 成立，因此：

- 增广是**无损**变换，不是有损变换；
- 增大 \(\|e\|\) 不增加恢复难度；
- \(r/d\) 比例不改变结论（\(r\) 只决定被湮灭的维数）。

**精确表述（重要）**：本节陈述的是「\(e\) 对 \(h\) 的不确定性贡献为零」，
即**给定 \(M\)** 时 \(h\) 被 \(c\) 完全确定，隐藏度 \(d-\mathrm{rank}=0\)。
它**不是**说无条件可恢复——生产路径只传递 `BasisDescriptor`
（`signal_dim`/`noise_dim`/`condition_number`/`fingerprint`，不含矩阵）。
因此全部机密性都压在「\(M\) 保密」这一个假设上，而不在噪声上；
5bis.2 与 5bis.6 说明该假设在本项目条件下并不稳固。

由于 \(B\) 正交，\(M=\Pi DB\) 的行两两正交（实测 \(MM^T\) 最大非对角元
\(1.3\times10^{-15}\)），故 \(\ker(M_{\rm bot})=\mathrm{rowspace}(M_{\rm top})\)，
恢复退化为归一化内积 \(h_i=\langle c,m_i\rangle/\|m_i\|^2\)——
**不需要求逆，也不需要任何关于 \(e\) 的知识**。

实测（`generate_transform`，FP64，d=64，r=8）：

| \(\|e\|\) 量级 | \(\|cP-h\|/\|h\|\) |
|---|---:|
| \(10^0\) | \(2.1\times10^{-15}\) |
| \(10^6\) | \(5.6\times10^{-10}\) |
| \(10^{12}\) | \(5.7\times10^{-4}\) |

噪声放大 \(10^{12}\) 倍后仍可恢复到 \(5.7\times10^{-4}\)；残差来源是浮点精度，
不是噪声的遮蔽作用。d=3072 的全尺寸下，\(e\) 放大 \(10^6\) 时
\(\max|h_{\rm rec}-h|=3.2\times10^{-9}\)。

### 5bis.2 三类攻击者的实测代价

| 攻击者能力 | 恢复误差 | 代价 |
|---|---:|---|
| 已知 \(M\)（客户端泄露 / 内部人） | \(\sim10^{-15}\) | 一次矩阵乘法 |
| 不知 \(M\)，有已知明文对 \((h,c)\) | \(\sim10^{-13}\) | \(d+r\) 对样本的最小二乘；d=3072 时约 3080 对 |
| 不知 \(M\)，仅观察 \(c\) | 不直接解出 \(h\) | 但几何结构保留，见 5bis.3 |

第二行不需要"刁钻能力"：若权重公开或位于服务端，且输入 token
（进而 \(h_0=E[t]\)）对攻击者可知，则 \((h,c)\) 对可以自行构造。

### 5bis.3 条件数上界同时是安全上界

`max_condition_number: 10`（`configs/*.yaml`，`transforms.py:190`）是为数值
稳定性设的，但它同时给出距离结构的保真度上界：对任意 \(x,y\)，

\[
\frac{1}{\kappa_2(M)}\le\frac{\|(x-y)M\|}{\|x-y\|}\le\kappa_2(M).
\]

实测（d=256, r=8）：\(\mathrm{corr}(\|\Delta h\|,\|\Delta c\|)=0.66\)，
拉伸比落在 \([1.19,1.84]\)。这正是 embedding-inversion 类攻击所需的条件，
因此**即使 \(M\) 保密**，仅凭 \(c\) 也存在几何/统计泄漏通道。

> **⚠️ 本节被 5bis.8 条目 6 加强（2026-08，阶段 B 之后）。** 上面的 \(\kappa\)
> 界是「仅观察 \(c\)」情形下的结论。但阶段 B 的部署路径要求服务端持有
> Gram 块以计算 \(\rho\)，而 \(A=PP^{T}\) 使 \(h\) 的距离结构
> **精确**暴露（实测 1e-14~1e-6，非 \(\kappa\) 界内的近似）。
> 引用本节时必须同时引用 5bis.8 条目 6，不得只引用 \(\kappa\) 界。

### 5bis.4 刷新档位与可恢复性正交

设刷新为 \(e'=hC+eG+\xi\)。由于解码矩阵湮灭 \(e\) 所在子空间，
\(\xi\) 的生成方式**不改变** \(h\) 的可恢复性：

| 档 | 内容 | 对 \(h\) 可恢复性 |
|---|---|---|
| 0 | `fixed_debug` | 无影响 |
| 1 | `per_request` 浮点 \(\xi\) | **严格无影响** |
| 2 | 每层/每模块独立刷新 + 域分离 KDF | **严格无影响** |
| 3 | ML-KEM 主密钥派生刷新 | **严格无影响** |
| 4 | mod-q + 离散误差分布 | 改变代数结构（换栈，非增量） |

档 1–3 改善的是**重放抗性、跨域随机流隔离、密钥轮换**，属于工程与密钥
卫生，**不得**在报告中表述为安全等级提升或"更接近 LWE"。

另需注意：默认配置下 \(\xi\) 的量级远小于信号耦合项。以 d=3072、
\(C\) scale=0.02、\(\xi\) scale=0.02 计，\(\|hC\|\approx2.94\) 而
\(\|\xi\|\approx0.056\)，即"辅助噪声"中约 98% 是 \(h\) 的线性像。
这是独立于可逆 \(M\) 的**第二条泄漏通道**：即使将来把 \(M\) 换成有损投影，
只要保留 \(e\approx hC\) 且 \(C\) 可估，泄漏依然存在。

### 5bis.5 对 TEE / 融合内核的推论

融合内核或 TEE 保护的是非线性检查点内的**临时干净值**。但 \(c\) 本身是
对外暴露的持久张量，且按 5bis.1 是 \(h\) 的可逆编码。因此：

- 若 TEE 不包住整条 \(c\) 流，保护 Softmax 内部临时值**不足以**改变结论；
- 若 TEE 包住全链路，则"让服务器看混淆态"这一动机本身被削弱。

### 5bis.6 与 \(M\) 无关的第二、第三条泄漏通道

以下两条**不依赖** 5bis.1 的可逆性论证，即使把 \(M\) 换成有损投影也仍然成立，
因此必须独立记录：

**(a) 检查点旋转矩阵是持久 buffer。**
`attention_rotation`、`ffn_rotation`、`common_qk` 由
`register_buffer` 注册（`src/fastprove/models/obfuscated.py:320-322`，
未使用 `persistent=False`），因此**进入服务端 `state_dict`**。
对开放权重模型（本项目为 Llama-3.2-3B-Instruct，权重公开），攻击者结合
公开 checkpoint 与服务端 `state_dict` 即可闭式解出这些正交因子。
第 3.2 节"检查点基材料不进入 `state_dict`"的说法仅适用于混合基
\(M\) 及其逆，**不适用于**上述三个旋转 buffer。

**状态更新（阶段 A/C）**：`attention_rotation`/`ffn_rotation` 在阶段 B
被完全删除（其作用已吸收进部署权重，在线是恒等空转）；`common_qk` 在
任务 C4 中被重新归类为**服务端必须持有的密钥**（RoPE 之后应用，无法
吸收），在 `runtime_config.json` 中显式标注，见 5bis.8 条目 3。此条
(a) 对 `common_qk` 的结论仍然成立且是有意为之。

**(b) 辅助状态是信号的确定性仿射像，且 \(C,G,\xi\) 同样在 `state_dict` 中。**
\(e'=hC+eG+\xi\) 中 \(C,G\) 为固定的密钥派生矩阵。因此 \(e\) 不是独立
随机量，而是 \(h\) 的确定性函数加一个小扰动（默认配置下
\(\|hC\|/\|\xi\|\approx16\text{--}53\)，取决于维度）。

关键在于这些矩阵**不是进程内秘密**：`*_noise_coupling`、`*_propagator`、
`value_signal_coupling`、`attention_aux_to_hidden`、`*_fixed_refresh`
全部由 `register_buffer` 注册。实测 `configs/tiny_exact.yaml` 转换后的
服务端 `state_dict` 共 59 个键，其中 **28 个**是上述 \(C/G/\xi\) 材料，
另有 **6 个**是 (a) 中的旋转/QK 因子。因此这条通道对第 4 节的
**诚实但好奇观察者**即已成立，无需第 5 节排除的强攻击者能力
（无需 hook、无需 introspection、无需读寄存器）。

**(b') 刷新 \(\xi\) 的粒度远低于"逐请求随机"的字面含义。**
`_refresh` 返回的张量形状等于 `fixed.shape`，即每个
(请求, 层, 用途) 域**只有一个**噪声向量，再广播到整个 batch 与全部
token 位置（见 value 路径的 `[None, :, None, :]` 广播，
`obfuscated.py:751-755`）。因此同一请求内所有 token 共享同一 \(\xi\)，
攻击者做**均值中心化**即可消去它。

**(c) 密钥生命周期。** \(M\) 在一次模型转换中生成一次，被全部 28 层共享，
且不随请求或 token 重新随机化；只有 \(\xi\) 逐请求变化
（`obfuscated.py:267-284`）。与 Slalom 类方案的一次性盲化（per-input
one-time pad）相比，静态密钥复用是已知明文攻击可行的根本原因。

### 5bis.7 复核方式

上述数字可由
[`scripts/verify_recoverability_bound.py`](../scripts/verify_recoverability_bound.py)
一条命令复现（CPU、秒级、无需 checkpoint）：

```bash
PYTHONPATH=src python3 scripts/verify_recoverability_bound.py \
    --output results/raw/recoverability_bound.json
```

该脚本在恢复**失败**时以非零码退出——即它断言的是"机密性上界"，
若将来构造改变使 \(h\) 不再可恢复，脚本会失败并提示本节已过时。

这些是**已知上界**，与第 1 节 Non-Claims 一致，不是新发现的缺陷。

### 5bis.8 阶段 A–C 新增的安全后果（任务 D4 记录）

以下六条在阶段 A–C 的实现过程中被证实或引入，全部有复现命令，必须随
本文件一起维护：

**1. 块对角基降低已知明文攻击的样本代价（任务 A2）。**
`M = Π₁ D B Π₂` 中 `B` 是块宽 `b` 的块对角正交。混合态每个输出坐标只
依赖 `b` 个输入坐标，因此恢复每个块内的 `M` 只需要 `O(b)` 个
已知明文样本（稠密基为 `O(n)`）。`r ≪ d` 时大部分块不含噪声坐标，
块内信号可独立恢复。代价从 9.5M MAC 降到 49K MAC 的同时，攻击者按块
分解的代价也成比例下降。
复现：
```bash
python -m pytest -q tests/test_structured_basis.py -v
```

**2. 噪声读出矩阵 `N` 进入部署权重，观察者可恢复噪声态 `e`（任务 B1.3）。**
残差项的在线形式是 `(c @ N) @ Wnz`，其中
`Wnz = (G − I) M_bot`。已验证不可规避：`Z = N(G−I)M_bot` 的秩恒为 `r`，
且对任意分解 `Z = UV`，`M @ U` 的信号块为 0、噪声块满秩。因此观察部署
权重（`noise_read` + `deployed_*_noise_out`）者可恢复 `e`
（up to 一个可逆 `r×r` 映射）。这不额外暴露 `h`，但使 5bis.6(b) 记录的
`e ≈ hC` 通道直接可利用。唯一替代是令 `G = 0`（放弃链式衰减），本方案
保留 `G = γP` 并接受此泄漏。
复现：`tests/test_deployed_weights.py`（17 项，含秩/分解哨兵测试）。

**3. `common_qk` 服务端可见，QK 几何结构不受保护（任务 C4.3）。**
`common_qk` 在 RoPE 之后应用，无法吸收进任何部署权重，服务端必须在
运行时持有它。它现在是持久 buffer 并在 `runtime_config.json` 的
`server_visible_key_material` 字段中显式标注（不允许改名规避检查）。
后果：`Q'K'^T = QK^T` 的几何（分数多集与排序结构）对服务端可见。
复现：
```bash
PYTHONPATH=src python -m pytest -q tests/test_converter.py -v
grep server_visible_key_material <server_dir>/runtime_config.json
```

**4. `ρ` 的 FP32 精度上界：`‖e‖/‖h‖ ≤ 30`（任务 A2.4）。**
`A_gram = P Pᵀ` 靠抵消消去噪声子空间，FP32 相对误差按 `(‖e‖/‖h‖)²`
增长（比值 1 → 4e-7；100 → 1e-4；1000 → 2e-2）。`AUXILIARY_MAGNITUDE_BOUND
= 30` 是实测上界，转换期由 `validate_auxiliary_budget` 强制。想靠加大
噪声提升混淆强度会**静默污染信号路径**（`ρ` 失准），而不是只污染噪声。
不得调大该上界；要调必须重新实测并更新 `structured.py` 的 docstring 表。
复现：
```bash
python -m pytest -q \
  tests/test_structured_basis.py::test_gram_accuracy_degrades_with_noise
```

**5. tied-embedding 使词表侧峰值内存 +100%，超出手册 §69 的 5% 目标
（任务 B4.2）。**
Llama-3.2 的 `lm_head` 与 `embed_tokens` 共享权重；`untied_deployed`
模式下 `deployed_head` 是独立的 `[n, V]` 矩阵。词表侧内存从共享的
`[V, d]` 变为 `[V, n] + [n, V]`，结构比 `2n/d ≈ 2.03`（d=3072, r=16）。
3B/BF16 投影约 +0.79 GB（未实测：本地无 tied 预训练 checkpoint，
见 `results/raw/performance_D2.json` 的 `tied_embedding_memory` 节）。
`fused_norm_head` 模式需要融合 kernel（eager 会物化明文归一化态），
当前实现直接抛 `NotImplementedError`，不静默回退。
复现：
```bash
PYTHONPATH=src python scripts/record_performance_D2.py \
    --output results/raw/performance_D2.json
python -m pytest -q tests/test_secure_lm_head.py::test_fused_norm_head_raises_not_implemented
```

**6. 服务端必须持有 Gram 块，因此 `h` 的欧氏几何被精确暴露（阶段 B 的直接
后果，本条加强 5bis.3）。**
部署前向用 `A_gram = P Pᵀ` 的块对角形式计算 RMSNorm 标度 `ρ`
（`structured.py:signal_norm_squared`），所以 `deployed_gram_blocks` 和
`deployed_gram_perm` **必然**在服务端 bundle 中（已在
`tests/test_converter.py` 的密钥扫描里显式归类为服务端可见）。

后果：`A` 只把 `P` 确定到一个右正交因子——对任意满足 `A = R Rᵀ` 的
`[n, d]` 因子 `R`（例如对称特征基），都有 `c R = h Q`，`Q` 正交。因此
**不需要任何已知明文对、也不需要知道 `M`**，仅凭服务端 bundle 即可把 `h`
恢复到一个全局正交变换。实测（d=256, r=16, b=16, FP64）：

| 泄漏量 | 最大绝对误差 | 不用 Gram 的基线 |
|---|---:|---:|
| 逐 token 范数 `‖h‖` | 2.1e-14 | 11.84 |
| 所有成对内积 | 8.5e-13 | — |
| 所有成对距离 | 1.1e-6 | — |

误差在 FP64 舍入量级，即几何是**精确**暴露而非近似。这使 5bis.3 的
`κ ≤ 10` 距离保真度上界失去意义：距离不是被界住，而是可精确读出。
embedding-inversion 与语义分类类攻击所需的条件因此被完全满足。

这条**不可通过工程手段消除**：`ρ = rms(h)` 是模型语义的一部分，服务端要算
它就必须掌握足够信息确定 `‖h‖`。可选缓解只有把 `ρ` 的计算移入 TEE 或融合
kernel 并让 Gram 块不可读，那属于 5bis.5 讨论的信任边界迁移，不是本原型的
性质。

复现：
```bash
PYTHONPATH=src python scripts/verify_gram_leakage.py \
    --output results/raw/gram_leakage.json
```
原始记录：`results/raw/gram_leakage.json`

### 5bis.9 正确性强制的范数泄漏可直接恢复输入 token（实测，最严重后果）

5bis.7 说明 `ρ` 必然泄漏逐 token 的 `‖h‖`。本节给出它的**具体后果**：
这条通道可以直接恢复词表置换本应隐藏的明文输入 token ID。

**攻击链（不需要已知明文对，不需要知道 `M`）：**

1. C1 要求服务端算对 `ρ = sqrt(‖h‖²/d + eps)`，否则 RMSNorm 和模型输出都错；
2. `ρ` 代数上确定 `‖h‖`：`‖h‖² = (ρ² − eps)·d`；
3. 第 0 层 `h₀ = E[token]` 就是 embedding 表的一行，故 `‖h₀‖` 是**逐 token 常量**；
4. **行范数对行置换不变。** 词表置换 `τ` 只重排 embedding 行，不改变任何一行的
   范数，因此在这条通道上**不提供任何保护**；
5. base checkpoint 是公开的，攻击者可离线建 `范数 → token` 查找表。

**实测恢复率**（`scripts/verify_token_recovery_from_rho.py`，n=4000 随机 token）：

| 模型 | `ρ` 存储精度 | 平均候选数 | 唯一且正确恢复 |
|---|---|---:|---:|
| Qwen2-1.5B | FP64 | 1.06 | **98.7%** |
| Qwen2-1.5B | **FP32** | 1.75 | **92.5%** |
| Qwen2-1.5B | BF16 | 4360 | 0.0% |
| Llama-3.2-3B | FP64 | 1.00 | **100.0%** |
| Llama-3.2-3B | **FP32** | 1.07 | **93.4%** |
| Llama-3.2-3B | BF16 | 4491 | 0.0% |

embedding 行范数互不相同的比例：Qwen2 99.05%，Llama-3.2 **100.00%**。

端到端演示（Qwen2，8 token 句子，服务端只见混淆 ID
`[87196, 89827, 24139, 36140, 43166, 69916, 19409, 83762]`）：
**8/8 逐 token 精确恢复**，还原出原句
`", I am a patient with diabetes"`。

**核心张力：AGENTS.md R5 要求 RMS 统计用 FP32 或更高，exact 门禁也在 FP32 下
判定——而正是这个精度让攻击成立。** BF16 的 `ρ` 能挡住这条通道（恢复率 0%），
但它违反 R5，且 4.7e-03 的相对误差远超 exact 门禁的 logit 容差 1e-3。也就是说
BF16 不是解法，它只是把隐私失败换成了正确性失败。

**适用范围（不得夸大）：** 该攻击针对第 0 层，因为那里 `h₀` 恰好等于单个
embedding 行。更深层的 `h` 是上下文相关的，`范数 → token` 的一对一关系不再成立。
但第 0 层**就是输入**，而输入正是客户端要隐藏的东西；且 `‖h_ℓ‖` 序列在所有层
都仍然泄漏，只是语义解释需要另做实验。

复现：
```bash
PYTHONPATH=src python scripts/verify_token_recovery_from_rho.py \
    --model-path <本地明文 checkpoint 目录> \
    --output results/raw/token_recovery_from_rho.json
```
原始记录：`results/raw/token_recovery_from_rho.json`、
`results/raw/token_recovery_from_rho_llama.json`

## 6. 各模式实际隐藏与保留的信息

### 6.1 `plaintext`

所有合法中间量按普通模型计算，不提供表示混淆。

### 6.2 `exact`

共同 Q/K 正交变换刻意保持

\[
Q'K'^T=QK^T.
\]

因此 clean score、Softmax probability、分数多重集合和排序结构在指定
操作内部仍然存在。混合 Value 和层间状态改变表示坐标，但 exact 模式本身
不通过 logit 噪声隐藏 attention 排序。

可选位置排列只能重标 key 位置；如果攻击者能一致观察 mask、KV 布局或多次
请求，额外泄露仍可能恢复对应关系。它不隐藏分数多重集合。

### 6.3 `topk_preserving`

margin 预算的目标正是保持重要 Top-k 集，因此“Top-k 集不变”是准确性属性，
不是排序隐私属性。margin 小、有效位置不足或 tie 时预算为零，相关 Query
完全没有 logit 扰动；报告必须披露零噪声比例。

### 6.4 `free_bounded`

固定上界允许部分排序改变，但有界随机扰动不自动提供差分隐私。当前机制：

- 没有相邻数据定义；
- 没有 sensitivity 上界；
- 没有 \((\epsilon,\delta)\) 预算或组合定理；
- 使用记录种子的确定性噪声以保证复现；
- 没有攻击成功率到安全参数的归约。

所以可以报告实际 \(\|\eta\|_\infty\)、KL/JS、Top-k 变化和精度损失，
不能把它们改名为“隐私预算”或“安全等级”。

### 6.5 `structural` 与 `full` 的相对地位

`structural`（噪声注入置零）与 `full`（完整噪声链）在**第 5bis.1 意义下
安全性相同，都是 0**：两者的 \(c\) 都满足 \(h=c(M^{-1})_{[:,:d]}\)。

因此 `full` 相对 `structural` 多出的精度损失，在当前威胁模型下
**没有换到任何可度量的隐私收益**。这不构成"噪声无用"的一般结论，只说明
在实数可逆构造下噪声强度不是安全旋钮。`structural` 应作为**消融项**呈现，
不得描述为"较弱但可部署的安全模式"（其注入量为 0）。

若要让噪声强度成为真实的安全旋钮，必须先改变构造使映射**有损**
（如 \(d\to k,\;k<d\) 的投影），此时才存在真实的精度/隐私权衡前沿。

## 7. 种子、刷新和重放

域分离降低不同层/用途误用同一随机流的工程风险。`fixed_debug` 刷新只用于
测试，绝不能作为生产随机性。`per_request` 由记录 seed 派生，适合可复现
实验，但若 seed 被观察或请求标识被重用，噪声可被重放。

后续部署若需要不可预测性，必须定义熵源、nonce 唯一性、密钥轮换、前向安全、
失败恢复和 seed 生命周期；这些机制不在当前原型内。

本节所述全部机制属于**密钥卫生与可复现性**，按第 5bis.4 与 \(h\) 的
可恢复性正交。改进它们是必要的工程工作，但不得计为安全里程碑。

## 8. 权重、数据和本机工件

原型不承诺模型权重保密。评测只允许使用明确授权的公开数据或用户已有本地
模型，不得把私有材料静默加入样本。原始结果应记录模型路径或标识、权重哈希、
数据来源、样本 ID、tokenized input 哈希和软件环境，但不写入密钥、访问令牌
或不必要的主机唯一标识。

本机已审计到一个离线 Qwen2 1.5B checkpoint；“本机存在”不等于它已被
fastProve 适配或评测。标准因果语言模型数据仍缺失，任何下载必须先获得一次
明确许可。

## 9. 从实验可以和不可以推出什么

### 可以在有原始记录支持时陈述

- exact 模式在指定硬件、dtype、样本和实现上的最大/平均数值误差；
- 不同 \(\tau_{\max},\alpha,k\) 下实际噪声上界、attention 分布变化和任务
  精度变化；
- top-k-preserving 与 free-bounded 的排序、精度和运行开销差异；
- 在预先声明的精度约束和选择规则下，哪个配置是本次实验的最佳折中点；
- 参考实现相对明文的转换、prefill、decode 和内存开销。

### 不能由该实验推出

- 任意未测试模型、数据、序列长度、seed、dtype 或设备上的普遍结论；
- exact/approximate 模式具有密码学机密性；
- 观察者无法从混合表示恢复信号（**已被第 5bis.1 反向证否**：在已知
  \(M\) 或可凑 \(d+r\) 对已知明文时，恢复是平凡线性代数）；
- 不返回概率就意味着概率没有在内存中出现；
- 有界噪声满足差分隐私或能抵抗某类攻击；
- 更大的噪声、更频繁的刷新或更好的密钥派生提升了 \(h\) 的保密性
  （见第 5bis.4：档 1–3 与可恢复性正交）；
- LWE/ML-KEM 派生意味着整个系统具有 LWE 安全性；
- reference eager 性能代表融合实现或生产服务性能；
- 随机 Tiny LM 的一致率代表真实语言能力保持。

### 关于效用指标的额外约束

任务准确率与 retention（RP）**只度量效用**，不能作为隐私证据。此外，
落在 chance floor 上的任务无法登记退化（其 retention 被钉在 100% 附近），
因此：

- 每个任务必须同时报告样本量、选项数、chance level 和 95% 置信区间；
- headline retention 必须同时给出**全任务**与**仅信号任务**两个版本；
- gold label 退化（如全部标签相同）的语料必须显式跳过，不得计入平均；
- 样本选择必须是有记录种子的采样（MMLU 按 subject 分层），不得使用
  语料前缀截断——多个本机语料按 label 或 subject 排序。

## 10. 结论发布检查表

在 `results/REPORT.md` 发布任何安全或精度结论前，必须确认：

- 结论可定位到 `results/raw/` 中的成功记录；
- 所有失败、跳过和校准点均可见，没有只保留有利点；
- 明文与混淆模式使用相同 checkpoint、token、顺序、dtype、device 和生成参数；
- 报告同时给出明文绝对值、混淆绝对值、绝对变化和相对变化；
- “扰动更大”没有被直接写成“更安全”；
- exact greedy 不一致已经单独调查；
- 当前 Python 检查点边界和强攻击者限制被明确重述；
- 第 5bis 节的可恢复性上界被复述，且没有用效用数字暗示隐私；
- 明文基线已与同模型的公开参考值对账（偏差超过约 3pp 必须先排查
  harness，再讨论混淆模式的相对结果）；
- 未运行的实验标为 pending/unexecuted，不以零值或空图代替结果。
