# 双通道噪声预算重构：交接提示词与代码修改计划

状态：CURRENT（S0–S5 已执行；增量见文末「S1 之后的增量」）
日期：2026-09-19
上游依据：AGENTS.md、docs/PROBLEMS.md
适用范围：双通道噪声预算重构的提示词与分阶段计划

日期：2026-09-19。依据：[噪声放置分析](noise_placement_explained.pdf)（推导与 T1–T10 核验）、
[攻击工作因子](work_factor_security.md)、[问题分析（已归档）](archive/2026-09-13_problem_analysis.md)、
核验脚本 `docs/verification/verify_noise_ordering.py`、原始记录
`results/raw/noise_ordering_analysis_20260919.json`。

本文件两部分：**第一部分**是可直接复制给 agent 的提示词；**第二部分**是给人看的详细代码修改计划（含现状盘点、文件行号、分阶段任务、验收门禁）。

---

# 第一部分：提示词（复制此段）

```text
你在 /Users/yin/code/fastProve 工作。先读 AGENTS.md，它的约束高于本提示词。

## 背景：一个已经用数学推导 + FP64 核验确定的结论

我们的协变混淆是 c = [h, e] M，M 保密可逆，语义读取矩阵 L = M^-1 [Q_hat; 0] 满足 cL = h。
已核验的事实（脚本 docs/verification/verify_noise_ordering.py，记录
results/raw/noise_ordering_analysis_20260919.json）：

1. 「先加噪再混淆」与「先混淆再在外面加噪」生成同一个噪声族，由 n -> n M^-1 双射相连。
   顺序本身不是自由度。(T1, rel err 1.6e-15)
2. 外加噪声 n 对语义的全部影响恰好是 delta_h = nL。(T2, rel err 3.8e-15)
3. ker(L) 是唯一与非线性检查点严格交换的噪声子空间，维数 (D-d)+r；且它恰好等于
   expanded_full 的噪声空间。免费额度是饱和的。(T3, T5)
4. ker(L) 内的噪声对 h 的信息论隐藏量恒为零：放大 1e12 倍后 h 仍可恢复到 6.7e-4。(T4)
5. 已知明文下，语义解码器可由 T = d + rank(辅助) + 1 条样本闭式恢复，代价 O(T N^2)，
   无搜索无训练。填满 ker(L) 只把 T 从 d+r+1 抬到 N+1，是一个加性常数。(T5)
6. ker(L) 外的噪声等价于直接给明文 h 加噪。真正的壁垒是信噪比：要把泄漏压到每坐标
   1 bit，需要 sigma_delta = 0.577 sigma_h，即 58% 相对误差。放大系数只有约 2.5x
   （随机权重代理），不是主因。(T7, T8, T9)
7. LWE 的困难性来自 mod q 而不是噪声：实数域 64 条样本即 100% 恢复秘密，Z_q 上同一
   估计量 0%。浮点 GEMM 管线里没有 mod q，所以「LWE 增广矩阵加噪」中的 LWE 只是命名。(T10)

结论：要做的不是选顺序，而是把噪声按 R^N = ker(L) 直和 语义方向 劈成两个通道，
分别记账、分别定预算、分别定价。

## 你的任务

按本仓库 docs/IMPLEMENTATION_PROMPT_dual_channel.md 第二部分的阶段计划执行。
默认从 S0 开始，逐阶段推进，每个阶段是一个可独立验收的完整纵切。

## 硬约束

- 不要改动数学定义来让失败的测试通过。发现不一致就停下来诊断并报告。
- 不要在 forward 里求逆、不要在 forward 里构造增广矩阵。
- exact 模式的门禁不许回退：任何阶段结束时 `python -m pytest -q` 必须全绿，
  且 results/raw/exact_gate_D1.json 的门禁必须仍然 PASS。
- 通道 0（ker L 内）的噪声必须如实登记为「信息论隐藏量 = 0」。任何文档或 docstring
  里不得把它写成隐私来源。
- 通道 1（语义方向）的任何噪声必须带一个可计算的误差界，且界的来源要在代码注释里
  指明（margin Delta_k / D_V / rho 三者之一）。
- 不得新增「先混淆后加噪」这类基于顺序的配置开关。按 T1，它不是自由度。
- 每个阶段只提交可运行的代码 + 对应的确定性小测试 + recorded run。不写没有 run 支撑的数字。
- 涉及真实 checkpoint 的阶段，先确认本机模型可用，不要静默降级到随机 tiny 模型。

## 现状必读（省掉你重复摸索）

- expanded_concat / expanded_full / P_hat 升维 / 补空间噪声 a / B_comp 在 src/ 里
  完全不存在，只存在于 docs 和 docs/verification 的 numpy 脚本。生产路径是 D = d，
  也就是免费通道当前只有 r 维。
- 噪声有两套互不相连的实现：
  A 套 src/fastprove/augmented_noise.py（有 energy_ratio 即 beta，有 component_sigma 换算），
      只被 scripts/run_augmented_noise_kernel.py 和 tests/math/test_augmented_noise_kernel.py 用，
      模型 forward 和权重转换完全不调用它；
  B 套 src/fastprove/models/obfuscated.py + src/fastprove/layers/deployed.py（真正跑推理的），
      噪声幅度硬编码 scale=0.02，共 13 处。
- L 和 J 已经存在：src/fastprove/structured.py:257 signal_projection() 就是 L，
  :262 noise_projection() 就是 J。ker(L) 没有被构造出来。
- structured.py:433 AUXILIARY_MAGNITUDE_BOUND = 30.0 是通道 0 当前的真实上限，
  原因是 FP32 块 Gram 归约的相消误差随 (||e||/||h||)^2 增长，不是安全考虑。
- 攻击成本 / 工作因子记录代码不存在。唯一脚手架是 src/fastprove/coa_views.py 的
  COAViewContract（已含 key_epoch_count 和 observation_budget_per_epoch）。
- 攻击脚本是自包含单文件，无共享 util。JSON 是手写惯例：claim / requires_* 布尔前提位 /
  by_<axis> 分轴结果。最完整的范式是 scripts/verify_token_recovery_from_rho.py，
  最小模板是 scripts/verify_gram_leakage.py（126 行）。
- 唯一真正复现攻击的测试是 tests/security/test_rho_noise.py:84 和 :260，新攻击测试照这个写。

## 每阶段交付

1. 改动的文件清单；
2. 确切的测试与实验命令；
3. 通过 / 失败 / 未执行三类分开报告；
4. raw 记录文件路径；
5. 下一阶段是什么。

不要基于「代码能编译」或「测试通过」就宣称阶段完成。通过不等于有效——
tests/ 里有过一个抓不到注入泄漏却给出「2 passed」的守卫测试，见 docs/PROBLEMS.md §3.1。
```

---

# 第二部分：详细代码修改计划

## 1. 现状盘点

这一节是全部计划的事实基础。每条都在 2026-09-19 核对过。

### 1.1 两套互不相连的噪声实现

| | A 套 | B 套 |
|---|---|---|
| 位置 | `src/fastprove/augmented_noise.py`（259 行） | `src/fastprove/models/obfuscated.py` + `src/fastprove/layers/deployed.py` |
| 幅度来源 | `NoiseRefreshSpec.energy_ratio` → `component_sigma()`（:54–62） | **硬编码 `scale=0.02`，13 处** |
| 配置文件 | `configs/augmented_noise_kernel.yaml`（`load_config` 不认这个 schema） | `configs/tiny_*.yaml` |
| 消费方 | `scripts/run_augmented_noise_kernel.py`、`tests/math/test_augmented_noise_kernel.py` | 模型 forward、权重转换、全部 exact/approx 门禁 |
| 接入模型 | **否** | 是 |

13 处硬编码位置：

```
src/fastprove/models/obfuscated.py:340,362,373,385,398   coupling C / propagator G / fixed xi
src/fastprove/models/obfuscated.py:683                    _refresh_out per_request 采样
src/fastprove/models/obfuscated.py:1076                   noise_embedding 表
src/fastprove/models/obfuscated.py:1231                   _initial_refresh_out
src/fastprove/layers/moe.py:91,97,106,172                 MoE 的同类构造
```

### 1.2 升维与补空间噪声：src/ 里不存在

`grep -rn "expanded_concat\|expanded_full\|complement" src/` 零命中。生产路径是

$$c = [h,\;e]\,M,\qquad \text{signal\_dim}=\text{hidden\_size},\quad \text{noise\_dim}=\text{hidden\_noise\_dim}$$

即 $D = d$。按分析报告的定理 3，当前免费通道 $\ker L$ 的维数是 $(D-d)+r = r$，也就是
**只有拼接噪声那 $r$ 维**，补空间那 $D-d$ 维一个都没有。

`docs/archive/2026-09-16_DERIVATION_PACKAGE.md` 描述的 `expanded_full`（$c=[h\hat P+aB_c,\,e]M$）目前只是纸面方案（该旧框架已归档、禁止复活）。

### 1.3 已有的可复用件

| 需要的东西 | 已存在？ | 位置 |
|---|---|---|
| 语义读取矩阵 $L$ | 是 | `structured.py:257 signal_projection()` |
| 噪声读取矩阵 $J$ | 是 | `structured.py:262 noise_projection()` |
| $M$ 的上/下分块 | 是 | `structured.py:267 signal_rows()` / `:272 noise_rows()` |
| $\ker L$ | **否** | 只在 `docs/verification/verify_noise_ordering.py` 里以 numpy 算过 |
| $\beta\to\sigma$ 换算 | 是，但未接入 | `augmented_noise.py:54 component_sigma()` |
| 通道 0 数值上限 | 是（非安全考虑） | `structured.py:433 AUXILIARY_MAGNITUDE_BOUND = 30.0` |
| 转换期预算校验 | 是 | `layers/deployed.py:365 validate_auxiliary_budget()` |
| 观察视图契约 | 是 | `coa_views.py:49 COAViewContract` |
| 攻击成本记录 | **否** | 全仓零命中 |
| 攻击结果统一 schema | **否** | 手写惯例，`artifacts.py` 的 `fastprove.run.v1` 容不下 |

### 1.4 通道归属现状

| 旋钮 | 配置字段 | 注入点 | 通道 |
|---|---|---|---|
| 拼接噪声 $e$、刷新 $\xi$、传播 $G$ | `hidden_noise_dim`、`noise_propagation_gamma`、`refresh_mode` | `obfuscated.py:334–403`、`683` | **0**（免费） |
| Value 噪声 | `value_noise_dim_per_head` | 部署 V 路径 | **0** |
| $\rho$ 噪声 | `rho_noise_bound` | `layers/rmsnorm.py:13 perturb_rms_scale`，作为除数用于 Q/K/V/gate/up/logits | **1** |
| attention logit 噪声 | `tau_max`/`tau_error`/`alpha`/`preserve_top_k` | `layers/attention.py:99–138` 预算、`:311–346` 注入 | **1** |
| router 噪声 | router 配置 | `layers/router.py:118–175` | **1** |

通道 1 现在有**三个互不相关的上界**，没有任何统一预算对象把它们和精度合同关联起来。

### 1.5 一个必须写进预算设计的事实

`obfuscated.py:501` 把 `noise_read`（即 $J$）作为 **persistent buffer 发给服务器**，
`:464–466` 和 `:510–513` 的注释已明确承认这让观察者能把 $e$ 恢复到一个可逆 $r\times r$ 映射之内。
`coa_views.py:25–34` 的 `CLOUD_COA_EXTRA_INPUTS` 也已把 `noise_injection_mapping` 声明为
cloud 视图下攻击者可见。

所以：**通道 0 在 cloud_coa 视图下连「不可读」都不成立**，不只是「信息论隐藏量为零」。
预算文档必须同时记这两条。

---

## 2. 阶段计划

七个阶段。S0–S2 是重构，S4–S6 是测量，S3 和 S7 是候选机制。

**优先级建议（基于分析报告）**：`S0 → S1 → S2 → S4 → S6 → S7 → S3`。

把 S3（实现升维 + 补空间噪声）排在最后，是因为定理 3 和 T5 已经证明它的收益上界是
「把已知明文样本需求从 $d+r+1$ 抬到 $N+1$」，一个加性常数；而 S4/S6 是测量当前真实
暴露面，S7 是可能零代价关掉最便宜攻击通道。先做收益已知有界的大改动是错的顺序。

---

### S0：把自由噪声锥显式化（无行为变更）

**目标**：让 $\ker L$ 成为代码里的一等对象，并提供一个诊断函数回答「这个噪声向量落在哪个通道」。此阶段**不改变任何数值行为**，纯新增。

**改动**

`src/fastprove/structured.py`，在 `noise_rows()`（:272）之后新增：

```python
def free_noise_basis(self) -> torch.Tensor:
    """返回 ker(L) 的一组基（按行）。

    ker(L) = {n : n @ signal_projection() == 0}，是唯一与非线性检查点严格
    交换的噪声子空间。当前构造下 D == signal_dim，故 dim ker(L) == noise_dim，
    且该子空间恰好是 noise_rows() 的行空间。

    信息论说明：落在此子空间内的噪声对语义状态 h 的条件熵贡献为零
    （h = c @ L 是确定性函数，与噪声幅度无关）。见
    docs/noise_placement_explained.pdf 定理 4 与
    results/raw/noise_ordering_analysis_20260919.json T4。
    不得把本函数返回的子空间描述为隐私来源。
    """

def channel_decomposition(self, noise: torch.Tensor) -> dict[str, float]:
    """把噪声向量分解到 ker(L)（通道 0）与语义方向（通道 1）。

    返回 free_component_norm / semantic_component_norm /
    induced_semantic_error_norm（= ||n @ L||，见定理 2）/ free_fraction。
    诊断用，不在 forward 路径调用。
    """
```

两者都在 FP64 下算，与 `dense_inverse()` 一致。

**新测试** `tests/math/test_free_noise_cone.py`（照 `tests/math/test_structured_basis.py` 的风格）

```python
def test_free_basis_has_expected_dimension()          # dim == noise_dim
def test_free_basis_annihilates_signal_projection()   # max |basis @ L| < 1e-12
def test_noise_rows_span_equals_free_cone()           # 主角 < 1e-6（注意 arccos 在 1 附近病态）
def test_channel_decomposition_of_pure_concat_noise() # [0, e] @ M 的 free_fraction == 1
def test_channel_decomposition_of_generic_noise()     # 随机 n 的 semantic_component_norm > 0
def test_induced_semantic_error_matches_n_times_L()   # 定理 2，rel err < 1e-12
```

**验收**

- `python -m pytest -q tests/math/test_free_noise_cone.py` 全绿；
- `python -m pytest -q` 全绿（无行为变更，计数应只增不减）；
- 新增一条 recorded run：`results/raw/free_noise_cone_check.json`，记 `dim_ker_L`、
  `max_semantic_leak`、`subspace_principal_angle`，以及 `torch_version`/`platform`/`seed`。

**注意**：`arccos` 在 1 附近病态，主角 $3\times10^{-8}$ rad 对应 $1-\cos\approx5\times10^{-16}$，
是 FP64 舍入级。容差写 `1e-6` 而不是 `1e-8`，并在测试里注明理由。这个坑我在写核验脚本时踩过。

---

### S1：统一噪声幅度配置，消灭 13 处硬编码

**目标**：把 B 套的 `scale=0.02` 全部替换为配置驱动，并让 A 套的 $\beta\to\sigma$ 换算成为唯一入口。

**改动 1**：`src/fastprove/config.py` `ObfuscationConfig`（:68–95）新增字段

```python
free_noise_energy_ratio: float = 0.02      # beta_e：通道 0 噪声能量 / 信号 RMS
signal_noise_coupling_scale: float = 0.0   # C 的幅度，默认 0（与 A 套一致）
```

`__post_init__` 增加校验：两者有限且 `>= 0`；且

```python
if self.free_noise_energy_ratio > AUXILIARY_MAGNITUDE_BOUND:
    raise ValueError(...)   # 与 structured.py:433 的数值上限显式关联，不要隐式共存
```

字段名用 `free_noise_*` 而不是 `beta_e`，让通道归属在名字里就是自明的。

**改动 2**：`src/fastprove/augmented_noise.py` `component_sigma()`（:54–62）成为唯一换算入口。
B 套调用它，不再自己乘系数。若 A 套的 `NoiseRefreshSpec` 对 B 套太重，就抽一个模块级
`free_noise_sigma(energy_ratio, calibration_rms, noise_dim)` 供两套共用，
**但不要复制公式**——复制就会漂移。

**改动 3**：替换全部 13 处

```
obfuscated.py:340,362,373,385,398  -> config.signal_noise_coupling_scale 或 free_noise_sigma(...)
obfuscated.py:683                  -> free_noise_sigma(...) * refresh_noise_scale
obfuscated.py:1076                 -> free_noise_sigma(...)
obfuscated.py:1231                 -> free_noise_sigma(...) * initial_refresh_noise_scale
layers/moe.py:91,97,106,172        -> 同上
```

区分清楚：`:340` 等处的 coupling $C$ 与 `:683` 等处的刷新 $\xi$ 是两个不同的量，
不要都映射到同一个配置字段。$C$ 的默认值应是 0（A 套的 `signal_noise_coupling_scale: 0.0`），
$\xi$ 的默认值应等价于现在的 0.02。

**改动 4**：`configs/tiny_exact.yaml`、`configs/tiny_approx.yaml` 的 `obfuscation` 段
显式写出两个新字段，值取「与当前行为等价」：`free_noise_energy_ratio: 0.02`、
`signal_noise_coupling_scale: 0.0`。

**改动 5**：`src/fastprove/converter/save.py` 的 `runtime_config.json` 记录新字段。

**验收（这是本阶段最关键的一条）**

- **数值等价性回归**：改动前后，`configs/tiny_exact.yaml` 与 `configs/tiny_approx.yaml`
  在同一 seed 下的输出必须逐位一致。做法：改动前先跑一遍存基线
  `results/raw/pre_s1_baseline_tiny_exact.json`（以及 `..._tiny_approx.json`），改完再跑对比。不允许「差不多」。
- `python -m pytest -q` 全绿；
- `results/raw/exact_gate_D1.json` 门禁重跑仍 PASS；
- `grep -rn "0\.02" src/fastprove/` 只应命中 yaml 默认值的文档字符串，不再有裸魔数。

**风险**：`:340–398` 那几处 coupling 与 `deployed.py` 的 `(G-I) @ M_bot` 增量形式耦合，
改错会破坏残差恒等式。先只改 `:683`/`:1231`（刷新项，最独立），跑通回归，再动 coupling。

---

### S2：双通道预算账本

**目标**：建立一个显式对象，把「每一分噪声花在哪个通道、买到什么、付多少精度」变成可审计的记录，并把通道 1 现存三个孤立上界统一到一个预算下。

**新文件** `src/fastprove/noise_budget.py`

```python
@dataclass(frozen=True)
class ChannelBudget:
    """单个通道的噪声预算。"""
    channel: int                    # 0 = ker(L) 内，1 = 语义方向
    dimension: int
    energy_ratio: float
    accuracy_cost_bound: float | None   # 通道 0 恒为 0.0；通道 1 必须非 None
    information_theoretic_hiding_bits: float | None
    bound_source: str | None        # 通道 1 必填："margin_delta_k" / "value_diameter_D_V" / "rho"

    def __post_init__(self) -> None:
        # 通道 0：强制 accuracy_cost_bound == 0.0 且
        #         information_theoretic_hiding_bits == 0.0（定理 4），
        #         任何非零值都是记账错误，直接 raise。
        # 通道 1：强制 bound_source 非 None 且在白名单内（否则就是没有可计算的界），
        #         accuracy_cost_bound 必须有限非负。


@dataclass(frozen=True)
class DualChannelBudget:
    free: ChannelBudget
    semantic: ChannelBudget

    def manifest(self) -> dict[str, object]:
        """机读记录，落盘到 raw 结果旁边，照 coa_views.write_view_manifest 的模式。"""

    def assert_consistent_with(self, config: PrototypeConfig) -> None:
        """校验预算与实际配置一致：
        - free.dimension == basis.free_noise_basis().shape[0]
        - free.energy_ratio == config.obfuscation.free_noise_energy_ratio
        - semantic 的三个来源上界（rho_noise_bound / attention tau / router tau）
          都被 semantic.accuracy_cost_bound 覆盖
        exact 模式下 semantic.energy_ratio 必须为 0。
        """


def semantic_hiding_bits(sigma_delta_over_sigma_h: float) -> float:
    """0.5 * log2(1 + 1/ratio^2)，单位 bit/坐标。见定理 9（信噪比壁垒）。
    ratio == 0 返回 inf（无隐藏）。"""
```

**改动**：`src/fastprove/converter/save.py` 的 `ModelConverter.convert()`（:35）在写 `runtime_config.json` 时
一并写 `noise_budget.json`（`DualChannelBudget.manifest()`），并在写之前调用
`assert_consistent_with(config)`。这样预算不一致会在转换期就失败，而不是事后靠人读文档发现。

**新测试** `tests/math/test_noise_budget.py`

```python
def test_channel_zero_rejects_nonzero_hiding_claim()      # 定理 4 的守卫
def test_channel_one_requires_bound_source()
def test_channel_one_rejects_unknown_bound_source()
def test_exact_mode_forces_zero_semantic_energy()
def test_budget_dimension_matches_free_noise_basis()      # 与 S0 接上
def test_semantic_hiding_bits_matches_reference_table()   # 对齐分析报告的 SNR 表：
                                                          # 1e-5 -> 16.61, 0.577 -> 1.00
def test_manifest_round_trips()
```

**验收**

- 上述测试全绿；
- 跑一次 `configs/tiny_approx.yaml` 的转换，产出 `noise_budget.json`，人工核对
  `semantic.accuracy_cost_bound` 与 `rho_noise_bound: 1e-5` 的关系合理；
- `python -m pytest -q` 全绿。

**注意**：`test_channel_zero_rejects_nonzero_hiding_claim` 必须是一个**真的能失败**的测试。
构造一个 `ChannelBudget(channel=0, information_theoretic_hiding_bits=3.0, ...)` 并断言
`pytest.raises(ValueError)`。写完后手动把校验注释掉确认测试确实转红，再恢复。
`docs/PROBLEMS.md` §3.1 记过一个抓不到泄漏却「2 passed」的守卫测试，别重犯。

---

### S4：Tier-0 攻击——已知明文闭式解码器恢复

**目标**：把分析报告 T5 的攻击从 numpy 原型移植成对**真实导出 bundle** 的攻击，
并测出它的实际成本。这是当前缺失的最重要一条 Tier-0 攻击。

**新脚本** `scripts/verify_known_plaintext_decoder_recovery.py`

照 `scripts/verify_gram_leakage.py`（126 行，最小模板）的骨架，核心算法：

```python
def recover_decoder(h_mat, c_mat, nuisance_rank):
    """闭式恢复语义解码器。见 docs/noise_placement_explained.pdf §4.1。

    c_t = h_t A + nu_t B。步骤：
      1. Q, _ = qr(h_mat)；resid = c_mat - Q @ (Q.T @ c_mat)      # P_H^perp C
      2. V = right_null_space(resid)                              # == ker B
      3. AV = lstsq(h_mat, c_mat @ V)
      4. return V @ pinv(AV)                                      # L_hat，满足 c L_hat = h
    """
```

参考实现见 `docs/verification/verify_noise_ordering.py` 的 `recover_decoder_kpa()`，
但**不要直接 import**——那是 numpy + 合成实例；这里要用 torch + 真实导出的
`generate_structured_basis` / 真实 server bundle。

CLI（照既有约定）：`--output`（必填）、`--signal-dim`、`--noise-dim`、`--block-size`、
`--seed`、`--sample-counts`（逗号分隔，扫 $T$）、可选 `--server-model-dir`
（有则攻击真实 bundle，无则用合成 basis）。

JSON 必含（照 `results/raw/token_recovery_from_rho.json` 的惯例）：

```json
{
  "claim": "...",
  "requires_known_plaintext_pairs": true,
  "requires_knowledge_of_the_mixing_basis": false,
  "requires_public_base_checkpoint": true,
  "requires_modified_kernel": false,
  "requires_register_access": false,
  "theoretical_min_samples": "d + nuisance_rank + 1",
  "by_sample_count": [
    {"T": 39, "recovered": false, "reason": "insufficient samples for rank condition"},
    {"T": 41, "recovered": true, "heldout_rel_error": 5.1e-15,
     "decoder_rel_error_vs_true_L": 0.33}
  ],
  "cost": { "...见 S5..." }
}
```

**关键**：必须同时报 `heldout_rel_error`（在**新鲜噪声的留出样本**上）和
`decoder_rel_error_vs_true_L`。我在核验时发现二者会分叉：当前 $D=d$ 的
concat 构造下 $\hat L \ne L$（矩阵相对差可达 0.33），但在可达状态集上逐点精确。
只报后者会得出「攻击失败」的错误结论。

**新测试** `tests/security/test_known_plaintext_recovery.py`，照 `tests/security/test_rho_noise.py:84` 的范式
缩成小规模确定性单测：

```python
def test_recovery_fails_below_sample_threshold()
def test_recovery_succeeds_at_threshold()          # T = d + r + 1
def test_recovered_decoder_exact_on_heldout()      # rel err < 1e-10
def test_recovery_unaffected_by_free_noise_scale()  # 放大 1e6 倍仍恢复，定理 4
```

**验收**

- 测试全绿；
- `results/raw/known_plaintext_decoder_recovery.json` 存在，且在 $T = d+r+1$ 处
  `recovered: true`、`heldout_rel_error < 1e-10`；
- 如果攻击**没能**在理论阈值处成功，**不要调容差让它过**。停下来诊断：
  可能是 bundle 的实际辅助秩与预期不同，那本身是重要发现，要单独记录并报告。

---

### S5：攻击成本记录（`attack.v1`）

**目标**：`docs/work_factor_security.md` 定义了成本向量
$\mathbf C=(F,T,M_{peak},Q,O,D_{train})$ 和主排序标量 $C^{ref}$，但
**全仓零实现**。没有它，S4/S6 的结果无法汇总成工作因子曲线，
`work_factor_security.md` §9 的「当前判定」就永远不能推进。

**新文件** `src/fastprove/evaluation/attack_records.py`

不要去扩 `artifacts.py` 的 `fastprove.run.v1`——它的 `SUPPORTED_MODES` 白名单
（plaintext/exact/approximate）和必填 `metrics` 路径容不下攻击记录。新建平行的 schema：

```python
ATTACK_RECORD_SCHEMA = "fastprove.attack.v1"

@dataclass(frozen=True)
class AttackCost:
    """work_factor_security.md §2 的成本向量。"""
    estimated_flops: float
    wall_clock_seconds: float
    peak_memory_bytes: int
    active_queries: int            # Q
    passive_observations: int      # O
    training_data_records: int     # D_train，代数攻击为 0
    setup_seconds: float           # C_setup，每 key epoch 一次
    per_target_seconds: float      # C_target，逐目标

    def amortised_seconds(self, key_epochs: int, targets: int) -> float:
        """C_A(E,m)/m = E*C_setup/m + C_target。见 §3。"""


def build_attack_record(*, attack_id, tier, claim, coa_view, requires,
                        recovery_metric, recovery_value, cost,
                        config, environment) -> dict:
    """tier in {0,1,2}；coa_view 必须来自 coa_views 的白名单；
    requires 是 requires_* 布尔前提位的 dict。"""

def validate_attack_record(record) -> None:
    """必填键、数值有限性、tier 合法、coa_view 合法。
    右删失记录（未在预算内达到目标）必须显式标记 right_censored=True，
    不得把预算上限当成已测得成本（§6）。"""

def append_attack_jsonl(path, record) -> None:
```

复用 `coa_views.COAViewContract` 声明视图，并把
`contract.input_manifest()` 嵌进记录——`coa_views.py` 的注释本来就要求
「persist the input manifest next to raw attack results」。

**改动**：把 S4 的新脚本和 5 个既有攻击脚本接上成本记录。既有脚本
（`verify_token_recovery_from_rho.py` 等）**只新增 `cost` 字段，不改既有字段**，
以免作废已有 recorded run。

FLOPs 用解析式估算（例如 QR + SVD 的 $O(TN^2)$ 展开系数），不要靠 profiler。
在 docstring 里写明这是上界估算，不是最优复杂度下界——`work_factor_security.md` §4.1
已经这么措辞了，保持一致。

**新测试** `tests/security/test_attack_records.py`：必填键、`right_censored` 强制、
`amortised_seconds` 的三个规模点（$m\in\{1,10^2,10^4\}$，对应 §3 的要求）、
拒绝非白名单 `coa_view`。

**验收**

- 测试全绿；
- `results/raw/attack_costs.jsonl` 至少含 S4 的一条记录 + 5 个既有攻击各一条；
- 能算出 T5 攻击在参考硬件上的 $C^{ref}$，并与 `work_factor_security.md` §4.1 的
  右逆路径成本同量级对比。**如果同量级，就必须如实写：填满自由锥没有带来显著工作因子提升。**

---

### S6：否证实验——候选集 + 语言模型重排

**目标**：这是分析报告里唯一能**否证整套论证**的实验，也是
`docs/archive/2026-09-13_problem_analysis.md` §5.3 的最大证据缺口。优先级高于补测 PPL/MMLU，因为它便宜，
且它的结果决定要不要继续在 $\tau=10^{-5}$ 这个操作点上花精度预算。

**要检验的预测**

按定理 9，$\tau=10^{-5}$ 时每坐标仍保留 16.6 bit；而 `PROBLEMS.md` §4.3 实测
token 恢复率从 93.8% 掉到 3.9%。两者同时成立只能说明：掉的是查表攻击的脆弱性，不是信息量。
所以把「唯一匹配」换成「$\tau$ 邻域候选集 + 语言模型重排」或「多次观测取平均」，
恢复率应**显著高于 3.9%**。

**新脚本（从未实现）** `scripts/verify_token_recovery_candidate_rescoring.py`
（S6 未执行，该脚本在全仓库不存在；阻塞记录见
`results/raw/blocked_stages_s6_s7.json`。基础设施工件仍是已实现的
`scripts/verify_token_recovery_from_rho.py`。）

基础设施直接复用 `scripts/verify_token_recovery_from_rho.py`（它已经有
`mean_candidate_count` / `uniquely_determined_fraction` / `unique_and_correct_fraction`
这组候选集度量键，正是要扩展的地方）。三个新攻击变体：

1. **候选集 + 语言模型重排**：对每个位置取 $\rho$ 的 $\tau$ 邻域内全部候选 token，
   用公开基座模型对候选序列打分，取最大似然序列。
2. **重复观测平均**：同一 token 被观测 $T$ 次（新鲜噪声），先平均 $\rho$ 再查表。
   扫 $T \in \{1,4,16,64,256\}$，预期误差按 $\tau/\sqrt T$ 缩小。
3. **两者结合**。

JSON 按 `by_<axis>` 惯例分轴：`by_tau`（$0$、$10^{-6}$、$10^{-5}$、$10^{-4}$）
$\times$ `by_attack_variant` $\times$ `by_observation_count`。
必须包含 `baseline_unique_and_correct_fraction`（即现有 3.9% 那条）作为同表对照。

**必须同时记录**的是防御端代价：同一 $\tau$ 下的 logit max error 和 argmax 一致率，
否则无法判断「攻击变强了」是否伴随「精度也还行」。`PROBLEMS.md` §4.3 已有这几列，对齐它。

**验收**

- `results/raw/token_recovery_candidate_rescoring.json` 存在，含全部配置点或显式失败记录；
- 结论二分法写进记录的 `interpretation` 字段：
  - 恢复率显著高于 3.9% ⇒ $\tau=10^{-5}$ 不能作为隐私操作点，必须在报告里撤回；
  - 没有显著高 ⇒ 定理 9 的信息论记账在这个具体攻击面上偏悲观，记录这个边界。
- **两个结论都要写，不许只写有利的那个**（AGENTS.md：no result is silently omitted）。

**注意**：需要真实 checkpoint。先跑 `scripts/audit_pretrained_assets.py` 确认本机模型可用。
已确认可用的路径见 `results/raw/token_recovery_from_rho.json` 的 `model_path`
（deepseek-r1-distill-qwen-1.5b，vocab 151936 / hidden 1536 / 28 层）。
**不要静默降级到随机 tiny 模型**——随机模型的 embedding 范数分布没有意义。

---

### S7：embedding 等范数化（可能零代价的缓解）

**目标**：$\rho$ 泄漏的根源是「公开 embedding 行范数几乎两两不同」
（Qwen2 99.05%、Llama-3.2 100%），这是**基座模型的属性**，不是混淆构造的属性。
把范数差异离线吸收进第一层投影，可能以接近零精度代价关掉第 0 层这条最便宜的通道。

**数学**：令 $E$ 为 embedding 表，$s_i=\|E_i\|$。构造

$$E'_i = E_i \cdot \frac{\bar s}{s_i},\qquad
\text{并把逐 token 的} \frac{s_i}{\bar s} \text{补偿进第一个 block 的输入侧}$$

难点在于补偿项是**逐 token 标量**，而 RMSNorm 后的第一层投影是**逐 token 共享**的矩阵。
所以不能简单吸收进权重。要先在参考实现里确认可行性：

**先做可行性探针**，不要直接改转换器。计划新脚本
`scripts/probe_embedding_norm_equalisation.py`（从未实现：S7 未执行，脚本不在仓库中；
阻塞记录见 `results/raw/blocked_stages_s6_s7.json`）：

1. 取真实 checkpoint 的 $E$，算范数分布与 `distinct_fraction`；
2. 构造等范数化的 $E'$；
3. 在 **plaintext 参考模型**上，把 $E$ 换成 $E'$ 并加上逐 token 补偿，
   测 logits 与原模型的 `tensor_error_metrics`（复用
   `src/fastprove/evaluation/metrics.py:13`）；
4. 同时测等范数化后的「范数 → token」恢复率应降到基线（$1/V$ 量级）。

**只有探针显示精度代价可接受**，才进入第二步改
`src/fastprove/pretrained/{llama,qwen2}.py` 的加载路径和
`src/fastprove/converter/` 包的转换路径。

**验收**

- `results/raw/embedding_norm_equalisation_probe.json`：
  等范数化前后的 logit `relative_l2_error`、`max_absolute_error`、argmax 一致率，
  以及 `unique_and_correct_fraction` 从 ~93% 降到多少；
- 明确结论：精度代价是否在 exact 门禁容差内。**如果不在，如实记录并停止**，
  不要为了让它通过而放松门禁。

**边界**：这只解决第 0 层。`PROBLEMS.md` §1.3 已实测深层范数通道不归零
（稀有 token 全 28 层仍 49.4% vs 基线 8.7%）。所以 S7 的收益要写成
「关掉最便宜的那一条」，不是「关掉范数通道」。

---

### S3：补空间噪声与升维（收益已知有界，排最后）

**目标**：把 `docs/archive/2026-09-16_DERIVATION_PACKAGE.md` 的 `expanded_full` 从纸面变成实现，
即引入 $\hat P/\hat Q$ 升维和补空间噪声 $a$，把自由通道从 $r$ 维扩到 $(D-d)+r$ 维。

**先读这一段再决定要不要做。** 定理 3 + T5 已经证明：

- $\ker L$ 的维数是 $(D-d)+r$，而 `expanded_full` 的噪声空间**恰好等于** $\ker L$，
  所以它是保精确前提下的**最大**噪声族，做到它就到顶了；
- 它的全部安全收益是把已知明文样本需求从 $d+r+1$ 抬到 $N+1$——**一个加性常数**。
  核验数据：$d{=}32$、$r{=}8$ 时从 41 抬到 57；
- 而且它有一个反效果：`concat` 下攻击者恢复的 $\hat L \ne L$（可达集不满维），
  `full` 下可达集满维，攻击者恢复的**就是真解码器本身**（rel err $6.2\times10^{-15}$）。

所以 S3 是一个「工程量大、收益有数学上界、且有一项反效果」的改动。
**建议只在 S4/S5 测出「低秩 SVD 捷径确实是当前最便宜攻击」之后才做**——
`work_factor_security.md` §4.2 把这条列为 `expanded-concat` 的统计子空间攻击，
但它是否比 §4.1 的右逆路径更便宜，目前没有实测。

**若确定要做**，改动范围（仅列骨架，细节待 S4/S5 结论后再定）：

| 文件 | 改动 |
|---|---|
| `config.py` `ObfuscationConfig` | 新增 `lifted_dim`（$D$）、`complement_noise_energy_ratio`（$\beta_a$）；校验 $D \ge d$、$D-d$ 与 `basis_block_size` 对齐 |
| `structured.py` | 新增 $\hat P/\hat Q$ 生成（**用正交完成构造**，见下）、$B_c$ 基（$\ker(\hat Q^{\!\top})$）、扩展 `free_noise_basis()` 覆盖两类自由度 |
| `conversion.py:82–108` | K 矩阵改为 `docs/archive/2026-09-16_DERIVATION_PACKAGE.md` §3 的 $K_{i\to o}=\begin{bmatrix}\hat Q_iW\hat P_o & \hat Q_iC\\ 0 & G\end{bmatrix}$ |
| `layers/deployed.py:101–363` | 部署权重融合吸收 $\hat Q_i$ 与 $\hat P_o$ |
| `models/obfuscated.py` | 采样补空间噪声 $a$，与 $e$ 分开记账 |
| `noise_budget.py`（S2 建的） | `free` 通道拆成 `complement` + `concat` 两个子项 |

**必须注意**：$\hat P$ 要用正交完成构造（取 $D\times D$ 正交矩阵的前 $d$ 行，
则 $\hat Q=\hat P^{\!\top}$ 精确、$B_c$ 就是剩下的 $D-d$ 行），
**不要**用随机矩阵 + `pinv`。后者的右逆病态，会让条件数校验和 exact 门禁一起失守。
`docs/archive/2026-09-16_DERIVATION_PACKAGE.md` §2 也提了这一点。我在核验脚本的 `make_scheme()` 里就是这么做的。

**验收**：`expanded_full` 下 $\dim\ker L=(D-d)+r$；exact 门禁仍 PASS；
S4 的攻击样本阈值从 $d+r+1$ 实测抬到 $N+1$；`noise_budget.json` 正确拆分两个子项。

---

## 3. 全局验收门禁

任何阶段声称完成前，全部满足：

- [ ] `python -m pytest -q` 全绿，测试数只增不减（当前 40 个测试文件）
- [ ] `python docs/design_identity_check.py` 13 项恒等式全过
- [ ] `results/raw/exact_gate_D1.json` 门禁重跑 PASS
- [ ] `python docs/verification/verify_noise_ordering.py` T1–T10 全过
- [ ] 新增的每个数字都有 recorded run 文件路径
- [ ] 新增的每个守卫测试都手动验证过「能转红」
- [ ] 不存在裸魔数（`grep -rn "0\.02" src/fastprove/` 干净）
- [ ] 通道 0 的任何 docstring/文档都没有把它写成隐私来源

## 4. 不得声称

沿用 `AGENTS.md` 与 `work_factor_security.md` §8：

- 不声称 IND-CPA、信息论不可恢复、差分隐私、LWE 安全性、全同态、端到端加密；
- 不声称「无法恢复」或任何 bit 数安全等级；
- 不把 `expanded_full` 的满秩、有效秩、坐标覆盖当作恢复率的替代指标；
- 不把某个攻击变慢当作工作因子提高——由最便宜的攻击决定；
- 不迁移旧 structured bundle / 随机 tiny / 理想观察者实验的结果当作新结果；
- 通道 0 的隐私贡献必须如实记为 0。

## 5. 参考索引

| 内容 | 位置 |
|---|---|
| 完整推导与 T1–T10 | `docs/noise_placement_explained.pdf`（源 `.tex` 同目录） |
| 核验脚本 | `docs/verification/verify_noise_ordering.py` |
| 原始核验记录 | `results/raw/noise_ordering_analysis_20260919.json` |
| 工作因子口径 | `docs/work_factor_security.md` |
| 已记录的泄漏通道 | `docs/PROBLEMS.md`、`docs/archive/2026-09-13_problem_analysis.md` |
| 方案主体与 P1–P15（旧框架，已归档） | `docs/archive/2026-09-16_DERIVATION_PACKAGE.md`、`docs/archive/2026-09-16_PROOF_PACKAGE.md` |
| 观察视图契约 | `src/fastprove/coa_views.py` |
| 攻击脚本范式 | `scripts/verify_token_recovery_from_rho.py`（最全）、`scripts/verify_gram_leakage.py`（最小） |
| 攻击测试范式 | `tests/security/test_rho_noise.py:84`、`:260` |

---

# S1 之后的增量（P8a 于 2026-09-19 补记；不改写上文原文）

上文「现状必读」是 S1 之前的快照。截至本日已落地的增量：

1. **0.02 硬编码已收敛为具名默认。** `src/fastprove/config.py:86` 的
   `free_noise_energy_ratio: float = 0.02` 就是原先 13 处 `scale=0.02` 的单一来源；
   `signal_noise_coupling_scale` 的 dataclass 默认是保守的 `0.0`，而
   `configs/tiny_exact.yaml`/`tiny_approx.yaml` 显式写入历史值 `0.02` 以保持逐位一致。
   现在 `rg "0\.02" src/fastprove/` 只剩该默认值与注释
   （`config.py:90`、`models/obfuscated.py:184`），不再有裸魔数。
2. **`ker(L)` 已在 `structured.py` 构造。** `StructuredBasis.free_noise_basis()`
   （`src/fastprove/structured.py:254`）用 FP64 SVD 返回 `ker(L)` 的正交行基；
   `channel_decomposition()`（同文件 `:287`）给出通道 0/1 能量分解。
   通道 0 的信息论隐藏量为 0 的口径不变。
3. **`noise_budget.py` 已落地并在转换器接线。** `DualChannelBudget`
   （`src/fastprove/noise_budget.py:127`）在 `src/fastprove/converter/save.py:103-104`
   构造并做一致性校验，在 `:187-201` 写出 `noise_budget.json` 并记录 sha256。
4. **测试规模与布局已变。** P7 把测试移入
   `tests/{math,layers,models,deployment,security,evaluation,cli,pretrained}/`，
   现有 48 个 `test_*.py`（`tests/` 下合计 59 个 `.py`），
   `python -m pytest tests -q` 为 **399 passed**；上文「当前 40 个测试文件」已过期。
5. **S6/S7 状态。** 两阶段均未执行，阻塞记录在
   `results/raw/blocked_stages_s6_s7.json`；S6 的
   `scripts/verify_token_recovery_candidate_rescoring.py` 与 S7 的
   `scripts/probe_embedding_norm_equalisation.py` 从未实现（上文对应处已加注）。
