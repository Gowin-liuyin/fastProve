# 架构：生产路径 / 参考路径 / 侧路实验

状态：CURRENT
日期：2026-09-19
上游依据：docs/CLEANUP_PLAN.md §P2、AGENTS.md
适用范围：src/fastprove、scripts 的路径归类；完整模块地图见本文末与 P8 的补充

本文件回答一个新人最容易踩的问题：**哪些代码是推理时真正执行的，哪些只是验收门与数学
等价性测试的参考实现，哪些是 forward 从不调用的侧路实验。**

## 三分表

| 分类 | 模块 | 说明 |
|---|---|---|
| 生产路径（forward 实际执行） | `models/obfuscated.py` | `ObfuscatedDecoderBlock` / `ObfuscatedTinyCausalLM` 的无解码 forward；调用部署权重、SecureEmbedding、Secure LM Head、attention 与 rmsnorm 的混合态算子 |
| | `layers/deployed.py` | `build_deployed_attention` / `build_deployed_feed_forward` 及其 Gram 产物（rho 统计的因子张量） |
| | `layers/embedding.py`、`layers/head.py` | 预混词表嵌入与混淆 LM head（不物化明文 embedding / logits 列） |
| | `layers/attention.py` | `ObfuscatedAttention`、`compute_topk_budget`、噪声采样、`safe_masked_softmax_fp32` |
| | `layers/rmsnorm.py` | RoPE、Q/K 后正交变换、`perturb_rms_scale`、`rms_norm_fp32` |
| | `layers/swiglu.py::generate_swiglu_transform` | 生产 SwiGLU 的神经元置换与缩放生成 |
| | `augmented_noise.py::deployed_free_noise_sigma` | 唯一的生产用噪声标度函数（逐位等于 energy_ratio） |
| | `structured.py` | `StructuredBasis` 因子、Gram 产物、`free_noise_basis`（ker(L) 账本） |
| | `codec.py`、`state.py`、`seed.py`、`config.py` | 词表置换、混合态容器、域名分离种子、配置校验 |
| | `converter.py`、`noise_budget.py` | 部署打包与转换期双通道预算账本；`noise_budget` 同时参与转换门 |
| 参考路径（验收门 / 数学等价性测试，forward 不调用） | `layers/linear.py`（`ChainLinear`） | 一般增广仿射恒等式；被 `tests/test_chain_linear.py`、`test_deployed_matches_chain_linear.py` 使用 |
| | `conversion.py` | `convert_affine_chain` 的离线参考转换（生产融合见 `layers/deployed.py`） |
| | `transforms.py` 的 dense 族 | `generate_transform` / `generate_two_sided_transform` 仅用于验收门、等价性测试与侧路核实验；生产使用 `generate_orthogonal` / `generate_signed_permutation` |
| | `layers/moe.py`（`ObfuscatedMoE`）、`layers/router.py`（`StableRouter`） | 专家路由与 MoE 的正确性参考；当前 tiny LM 生产 forward 未接入 |
| | `layers/swiglu.py::convert_swiglu_weights`、`refresh_swiglu_noise` | SwiGLU 协变与刷新参考；测试专用 |
| 侧路实验（独立脚本，不接入推理） | `augmented_noise.py` 其余函数 | 采样、传播子、`record_generator`、覆盖度指标；服务侧路核实验与单元测试 |
| | `scripts/run_augmented_noise_kernel.py` 及 `results/raw/augmented_noise_kernel/` | 噪声链平稳性/覆盖度实验，不是推理路径 |
| | `scripts/verify_*.py`、`scripts/record_*.py` | 一次性探针与记录脚本（自包含单文件约定，见 docs/implementation_notes.md 与后续 docs/scripts_index.md） |

## 补充说明

- `coa_views.py` 是**字符串契约**，用于声明各威胁视图允许/禁止的输入，不是隔离机制；真正的
  隔离来自部署形态（服务端只有融合权重）与调试 API 门禁。
- `src/fastprove/evaluation/` 是原型自带的评测与报告流水线；`evals/` 是第二套独立 harness，
  依赖 `fastprove`，并有 3 个脚本反向引用它。归位方案见 docs/CLEANUP_PLAN.md §10-①（已决：
  保留独立顶层 + 补打包 + 统一指标定义）。
- 模块顶部带 `REFERENCE PATH` / `参考路径说明` / `侧路实验` banner 的文件即上表对应条目；
  删除它们会丢失验收门，故只标注、不删除。

## 模块地图（简版，P8 补齐依赖方向图）

```text
src/fastprove/
├── seed.py  state.py  codec.py  config.py  coa_views.py
├── transforms.py  structured.py  conversion.py  converter.py  noise_budget.py
├── augmented_noise.py
├── layers/{attention,rmsnorm,deployed,linear,swiglu,router,moe,embedding,head}.py
├── models/{plain,obfuscated}.py
├── pretrained/{llama,qwen2}.py
└── evaluation/{runner,metrics,accuracy,artifacts,report,...}.py
```
