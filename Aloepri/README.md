# AloePri 研究代码

这个仓库包含论文机制的研究实现和可保存为普通 Hugging Face 权重的工程变体。两者需要分开理解：

- `AloePriEngine` 保留扩维隐藏向量和自定义运行模块，便于检查各项数学运算。
- Qwen2、Llama 的标准权重导出采用**不改变隐藏宽度的符号置换方阵**。它可以用普通 Transformers 模型加载，但不是论文算法 1 的完整扩维复现，也没有因此获得论文的隐私或准确率结论。
- BERT 是额外的本地研究扩展。它需要自定义 LayerNorm 恢复模块，不能作为无需额外计算的普通服务端发布。

部分函数和目录保留旧名 `paper_consistent`，以便旧脚本继续导入。判断产物实际做了什么，应查看 `manifest.json` 中的 `implementation_variant`、`paper_algorithm1_reproduced` 和 `requires_custom_runtime`，不能只看名称。

## 安装和检查

先从 fastProve 根目录执行 `cd Aloepri`，以下命令均在这个子目录内执行。代码调用了 Transformers 内部接口，因此先使用已验证的版本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pytest
```

也可以安装项目及测试依赖：`.\.venv\Scripts\python.exe -m pip install -e ".[test]"`。`requirements-tested.txt` 记录的是本次本地小模型验证环境；它不意味着已经测过所有 CUDA、vLLM 或 SGLang 版本。

迁移来的历史文档、旧产物测试与当前独立仓库不相符，测试配置会明确说明相应跳过原因。数学、导出和评测测试仍应执行，不能把历史文件缺失当作数学检查通过。

## 从原始 Qwen2 生成标准权重

先准备本机原始模型目录，包括 config、tokenizer 和 safetensors 权重（可以分片）。下面先关闭噪声，检查改写前后的计算是否一致；它是正确性实验，不能用来宣称隐私保护有效。

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_qwen_checkpoint.py --model-dir model/Qwen2.5-0.5B-Instruct --export-dir artifacts/qwen/square_zero_noise --dtype float32 --device cpu --alpha-e 0 --alpha-h 0
.\.venv\Scripts\python.exe scripts/release/export_qwen_bundle.py --source-dir artifacts/qwen/square_zero_noise --export-dir artifacts/qwen/release_zero_noise --materialize
.\.venv\Scripts\python.exe scripts/evaluation/run_qwen_correctness.py --bundle-dir artifacts/qwen/release_zero_noise --baseline-model-dir model/Qwen2.5-0.5B-Instruct --output-dir outputs/qwen/zero_noise --dtype float32 --device cpu --correctness-atol 0.0001
.\.venv\Scripts\python.exe scripts/release/infer_bundle.py --bundle-dir artifacts/qwen/release_zero_noise --profile default --prompt "请用一句话介绍你自己。" --max-new-tokens 16 --device cpu
```

输出目录需要为空。导出程序不会删除已有目录来覆盖结果。默认种子来自系统随机源，实际种子只保存在客户端秘密文件中。`--seed 123` 用于可复现的实验；不要把公开实验种子当作部署秘密。后续变换仍由 PyTorch 随机数生成器产生，这不等于已经实现密码学安全的密钥生成方案。

输出中的文件这样使用：

```text
checkpoint/
  server/                    # 只把这里的普通推理文件交给服务器
  client/client_secret.pt    # 客户端保留，不能上传服务器
  stage_i_metadata.json      # 已去除秘密种子的公开说明
  manifest.json              # 实现范围和结构检查结果
```

release bundle 是模型持有者在本地使用的完整包，包含客户端秘密。**不能把整个包上传服务器**，只能交付 `profiles/<name>/server`。发布过程会检查全部权重名称、形状、分片索引和配置，并拒绝调试文件、需要自定义运行程序的 BERT 产物、非法 profile 名称及源目录重叠。结构检查不能代替数值、任务和隐私评测。

## Llama

同样只导出全部层。共享或不共享 Embedding/输出头的原模型都会分别处理输出头；两个位置可以采用不同的噪声参数。

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_llama_checkpoint.py --model-dir model/Llama-3.2-3B-Instruct --export-dir artifacts/llama/square_zero_noise --dtype float32 --device cpu --alpha-e 0 --alpha-h 0
.\.venv\Scripts\python.exe scripts/release/export_llama_bundle.py --source-dir artifacts/llama/square_zero_noise --export-dir artifacts/llama/release_zero_noise --materialize
.\.venv\Scripts\python.exe scripts/evaluation/run_llama_correctness.py --bundle-dir artifacts/llama/release_zero_noise --baseline-model-dir model/Llama-3.2-3B-Instruct --output-dir outputs/llama/zero_noise --dtype float32 --device cpu --correctness-atol 0.0001
```

`--num-layers` 不能用于部分层的标准导出。部分层的实验模块会额外恢复隐藏向量，保存后不再是普通模型的完整结构。

## 本地推理和检查结果如何理解

推理入口把输入 token 编号置换一次，随后由模型在混淆编号中生成，并使用 KV cache 保存已有计算、遇到 EOS 停止。最后客户端把输出编号恢复。可传 `--do-sample --temperature 0.65 --top-k 20 --top-p 0.95` 开启随机采样。

这是同一进程里的 token-ID 协议模拟，还没有实现远程服务，也没有证明普通文本 API 的“解码后再编码”能保留任意混淆编号序列。不能用这段代码的耗时替代论文集群延迟。

小样本等价检查必须满足：有测试样本，全部数值有限，所有决定性生成编号和文本相同，最大 logits 误差不超过明确给出的容差。默认 float32 容差为 0.0001。检查程序会记录误差及容差；NaN、Inf、只匹配一个样本、调用者自行填入 status=pass 都不能使它通过。

加入噪声后本来就可能改变输出，严格等价检查失败不自动说明实现错误。应另用与论文相同的数据和协议测任务准确率，同时测攻击成功率；当前仓库没有完成论文的大模型、完整数据集和集群结果复现。

攻击评测需要什么数据、如何区分真实答案与攻击者可见内容，见 [攻击评测条件和评分说明](docs/security_attack_protocol.md)。

## 历史研究产物

`--source-dir` 保留给已有中间产物的转换。它不能凭空生成仓库外的历史目录；一般使用上面的 `--model-dir` 路线。

历史调试格式保存了恢复矩阵、原始记录权重或重建种子。保存这些格式必须在 Python API 中显式设置 `allow_unsafe_debug_export=True`，仅用于本地研究，不能进入标准发布包。将它们改成普通文件名，并不会自动消除已有的秘密暴露或数学近似问题。

## 代码位置

| 内容 | 文件位置 |
| --- | --- |
| 扩维研究运行程序 | `src/aloepri/engine.py`、`layers/` |
| Qwen2/Llama 方阵导出 | `src/aloepri/_implementation/export/standard_square_checkpoint.py` |
| 完整权重结构检查 | `src/aloepri/_implementation/analysis/standard_weight_proof.py` |
| 单文件和分片读取 | `src/aloepri/_implementation/export/tensor_io.py` |
| 公开说明与秘密分离 | `src/aloepri/_implementation/export/export_safety.py`、`checkpoint_io.py` |
| 发布与本地推理 | `src/aloepri/release/`、`src/aloepri/private_inference.py` |
| 攻击评测 | `src/security_qwen/` |

旧架构文档保留了历史阶段名称，具体行为以当前代码和本 README 的限制说明为准。
