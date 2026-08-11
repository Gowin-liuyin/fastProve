# Raw results

当前状态：`not_run`。

`experiment_readiness.json` 是不含推理指标的就绪交接清单；它记录当前可执行的
候选范围、资源警告和仍需外部解决的标准数据集阻塞，不应被当作实验结果。

本轮未执行噪声扫描或预训练模型评测。后续每个配置在此保存一条独立 JSONL
记录；相同文件中不允许重复 `run_id`。一次 attempt 使用一个独立子目录和
全新的空 JSONL，当前原型不做 partial resume。随机 Tiny 记录必须标为
`correctness_only_random_tiny`，不能作为真实语言模型精度证据。

目录中现有的 `eval_*.json` 是早期多层结构化 smoke/protocol 工件，不是当前
`fastprove.run.v1` JSONL 记录；它们不会被 `scripts/build_report.py` 自动纳入
当前报告，也不能被解释为本轮四模式语言模型评测结果。

`pretrained_evaluation_status.json` 是不启动推理的本地资产审计结果。当前状态为
`skipped_external_dependency`：Qwen2 checkpoint/tokenizer 已找到并完成哈希，
标准 causal-LM 评测语料仍未获准。`flickr30k_caption_eval_inputs.pt` 是已准备
的 405 条非标准 caption-only cache；只有显式使用
`--accept-nonstandard-caption` 才会被 sweep 或 preflight 标为可执行候选。该文件包含搜索根、
缺失项和最小外部运行建议。

对应的显式限定范围 manifest 是
`pretrained_evaluation_status_caption_candidate.json`；它的
`evaluation_scope` 仍明确为 `caption_only_nonstandard_lm_candidate`，不等价于
标准 causal-LM 证据。

显式 device 的当前
`flickr30k_caption_preflight_cpu_fp32.json`、
`flickr30k_caption_preflight_mps_fp32.json`、
`flickr30k_caption_preflight_mps_bf16.json` 也只是无权重资源/身份预检；它们不含
困惑度、accuracy 或性能结果。MPS 清单额外记录无权重 matmul/Softmax smoke 和
checkpoint 的 FP32 运行算术，避免把 MPS 不支持 FP64 误当成可执行路径。
目录中的旧 generic `flickr30k_caption_preflight*.json` 仅保留作历史诊断，不能替代
当前显式 device/范围 opt-in 的 preflight。

校准 JSONL 经过 `scripts/select_sweep_candidates.py` 生成的
`candidate_manifest.json` 是 full 子集的唯一准入清单；它不会修改校准 raw，且
exact gate 失败时拒绝作为 `--spec-ids-file` 输入。
