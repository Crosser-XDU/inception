# 2026-09-16 下午至晚间：推理加速修改与实验记录

整理日期：2026-09-17。本文中的“下午”指前一轮对话及日志中的 **2026-09-16**；覆盖当日下午至晚间的连续推理优化工作，避免跨零点后日期混淆。

本文依据当前本地代码、相对 Git HEAD 的差异、已有四档实验报告，以及用户回传的三档重复测速日志整理。由于修改尚未逐项提交，章节按功能组织，不把每一行未提交差异都断言为某个具体时刻新增。

## 1. 本轮解决什么问题

目标是在同一模型、checkpoint 和数据区间下，降低 T/boundary 投机解码的实际耗时，并判断收益究竟来自草稿接受、缓存复用，还是目标模型本身变快。

本轮工作有三类：

1. **新增实验入口和审计**：自适应/位置调度、固定起始位置三组对照、逐项优化档位、重复汇总及实际执行计数。
2. **接入已有底层能力**：纠正缓存复用、延迟纠正、原地 KV、T/目标 LoRA 合并、批量 boundary、并行草稿、边界 hidden hook。它们并非全部在本轮重新实现；本轮主要让这些能力可以受控组合、直接测速并核查是否真正执行。
3. **新增或调整底层逻辑**：轻量严格验证路径、草稿 argmax 快速读出、优化执行计数、参数兼容性检查、可选预热，以及吞吐模式派生计时口径修正。

这些推理修改没有更改训练 loss、训练阶段或 checkpoint 文件。此前联合训练、GSM8K 答案提取修复属于前序工作，不计为本轮新改动。当前修改仍在本地，未因本轮整理而推送 GitHub。

## 2. 术语和一轮解码的含义

- **目标模型**：负责最终验证和纠正的完整语言模型，也是 greedy 使用的模型。
- **T**：预测后续位置中间 hidden 的递归模块；验证时仍运行完整目标模型，没有用 T 替换目标模型的 mid 层。
- **boundary 头**：把预测 hidden 映射为草稿 logits，避免每枚草稿都经过目标模型后续 tail 层。
- **BLOCK**：一个验证块的 token 上限，包含一枚已经由目标模型确定的 anchor token。

例如 `BLOCK=3` 表示：

```text
验证输入：[a, d1, d2]
a：已确定 token
d1、d2：最多两枚草稿
```

目标模型在输入 a 后的 logits 验证 d1，在输入 d1 后的 logits 验证 d2。只接受从 d1 开始连续正确的前缀；第一枚错了，不能因为第二枚碰巧匹配就单独接受第二枚。

因此要区分：草稿接受数、输出 token 数、验证调用次数。纠正 token 是目标模型的结果，不计入接受草稿。

## 3. 文件和职责

以下链接相对于本报告所在的 reports 目录。

| 文件 | 本轮职责 |
|---|---|
| [benchmark_decode_adaptive.sh](../scripts/benchmark_decode_adaptive.sh) | 新增自适应及按位置调整上限的便捷入口 |
| [benchmark_decode.sh](../scripts/benchmark_decode.sh) | 接收并转交策略、分段上限、门控、冷却和计时参数 |
| [run_decode.py](../scripts/run_decode.py) | 将包装器参数转换为实际解码器参数；检查策略配置是否合法 |
| [common.py](../scripts/common.py) | 审计策略、路由、分段接受统计和有效 T 执行证据 |
| [benchmark_decode_start.sh](../scripts/benchmark_decode_start.sh) | 新增三组对照的环境变量入口 |
| [benchmark_decode_start.py](../scripts/benchmark_decode_start.py) | 参数解析、三组实验编排、资产核查、重复汇总、优化执行审计 |
| [benchmark_decode_speed.sh](../scripts/benchmark_decode_speed.sh) | 新增可复现优化档位，控制组合及默认值 |
| [recurft_speculative_generate.py](../LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py) | 实际解码；新增严格验证快速分支、计数、预热及计时修正 |
| [test_adaptive_benchmark.py](../tests/test_adaptive_benchmark.py) | 自适应参数与统计检查 |
| [test_start_benchmark.py](../tests/test_start_benchmark.py) | 三组编排、真实解码器参数解析、档位及汇总检查 |
| [test_strict_decode.py](../tests/test_strict_decode.py) | 严格验证和真实解码控制流的张量/小模型测试 |
| [README.md](../README.md) | 使用方式、同步依赖、实验解释；其中“尚未实测”描述保留了各次实现时点，本报告第 11 节补充后续回传结果 |
| [MANIFEST.sha256](../MANIFEST.sha256) | 更新已有清单项目的校验值；不是性能测试结果 |

已有专项结果：[四档测速复核](speed_review_20260916.md)及[原始统计与来源哈希](speed_review_20260916.json)。

## 4. 自适应、按位置调度和起始位置对照

### 4.1 自适应入口

`run_decode.py::draft_policy_settings()` 将三种策略显式映射到底层：

| DRAFT_POLICY | 底层 mode | 行为 |
|---|---|---|
| fixed | fixed | 固定上限，关闭自适应门控及失败冷却 |
| schedule | schedule | 根据生成位置调整上限，不叠加置信度门控 |
| adaptive | heuristic | 分段上限内，使用置信度门控、提前停止和失败冷却 |

adaptive 的默认门控参数为：目标 margin 跳过阈值 1.0、缩短阈值 3.0、草稿最低 margin 1.0、连续失败 3 次后冷却 4 轮。margin 是 logits 前两名差距，不是经过校准的“正确概率”。

这些阈值只控制是否尝试投机和尝试多长，不改变目标验证的严格性。目标模型对当前 token 有把握，并不保证 T 对下一位置的预测正确。

分段参数 `SHORT_BLOCK/MEDIUM_BLOCK/LONG_BLOCK` 默认都等于 BLOCK；位置默认 96/192。因此只开启 adaptive，并不自动得到“1→2→4 枚草稿”。该调度需要显式设置块上限 2/3/5，且总 BLOCK 至少为 5。

参数检查包括：分段上限位于 1..BLOCK；位置阈值严格递增；fixed 不允许传入不同分段上限；margin 有限且非负；冷却次数必须同时启用或同时关闭。

新增 early/mid/late 接受统计按**块起始位置**归段；同时报告进入该段的样本数。后段只包含实际生成到那里的一部分题，不能直接推断所有题的后期都更容易。

### 4.2 固定起始位置三组对照

`benchmark_decode_start.py::make_run()` 生成以下实验：

- greedy：目标模型逐 token 生成。
- fixed：从头使用固定长度投机。
- after_start：生成满 N 枚 token 后，允许相同固定长度投机。

`SPEC_START_POSITION/--spec-start-position` 不包含 prompt；`START/--start` 是数据起始行。两者不是同一个参数。

例如 N=128：最早的投机块从输出的 0-based 位置 128 开始，首枚 anchor 已确定，首枚草稿最早位于 129。提前 EOS 或预算不足的样本保留在汇总中，并标为未触发投机。

两组投机均开启 `--lazy-t-sync` 和 `--defer-t-init-until-latent`：尚未需要 T 时推迟其初始化；允许延后的真实 hidden 同步先排队，再在需要时批量执行。起始位置前仍经过现有目标/KV 维护路径，不能视为零额外开销的纯 greedy 前缀。

### 4.3 实验公平性

- 每个 repeat 的 fixed 进程同时测量 greedy；after_start 使用 `--no-baseline`，共用同一份 greedy 结果。
- 奇偶 repeat 交替 fixed/after_start 顺序；解码器还传入 `--decode-order alternate`。
- 每个进程默认用首题预热一次 greedy 和投机；加载与预热不计时。
- 计时包含 prefill、T 初始化和输出生成。
- 记录 checkpoint 相关文件及数据文件 SHA256；运行前后检查资产是否发生变化。
- 核对题目、标准答案、顺序、prompt 长度及共享参数。
- 环境变量设默认，CLI 优先；speed 档位先重置开关，避免继承上次实验残留。
- 重复汇总用总 token / 总秒数，而非简单平均样本加速比。32 题重复 3 次是 96 次运行，不是 96 道独立题目。

## 5. 纠正前向和缓存：reuse、defer、cache

这些是已有底层路径，本轮加入受控档位和更清楚的计数。

### 5.1 reuse：保留验证产生的正确前缀 KV

若验证 [a,d1,d2] 后 d1 正确、d2 错误：

```text
保留 old_prefix + [a,d1] 的 KV
裁掉错误 d2 的 KV
目标模型仅对纠正 token c 做一次前向
同步相应真实 hidden 到 T
```

它避免重算已经验证的前缀，但**仍有纠正 token 的完整目标前向**，所以 correction_s 不会自动消失。实现关注点是 `target_cache.crop(old_length + len(accepted_tokens))`，以及相同范围的边界 hidden。

### 5.2 defer：将纠正 token 的计算并入下一轮

拒绝后，保留正确前缀 KV 和确定纠正 token 的目标 logits，使纠正 token 成为下一轮 anchor，再与下一轮草稿一起送入目标模型。

```text
reuse：验证 → 单独纠正前向 → 下一轮验证
defer：验证 → 下一轮以纠正 token 为 anchor 的验证
```

纠正 token 的上下文计算仍要完成，只是并入下一轮；不是跳过验证，也不是零成本增加输出。需要保留正确的缓存长度、位置及 EOS/预算处理，不能把旧 correction_s 全部从新总耗时里直接扣除。

defer 会改变轮次划分、后续草稿所处状态和调用结构，因此即便逻辑都严格，接受统计也不必与 reuse 相同。

### 5.3 cache：原地使用并回退 KV

`--inplace-draft-cache` 避免为投机反复显式 clone T/验证缓存，拒绝后裁掉不应保留的后缀。缓存从“复制一份试算”改为“原地试算，再按接受长度回退”。

这要求目标缓存与 T 缓存都对应真实已提交前缀，不能让错误草稿的 KV 污染后续轮次。测速入口限定它配合 reuse/defer 使用。

**原地缓存仍是 DynamicCache。** 内部追加可能分配新内存，不等同于预分配静态 KV，也没有因此自动支持 CUDA Graph。

## 6. 合并 T LoRA 与目标 LoRA

对于线性层，以列向量记号表示：

```text
y = W x + s B A x
W_merged = W + s B A
y = W_merged x
```

其中 s 是 LoRA 缩放系数。合并将额外低秩分支折入权重，推理时减少独立矩阵运算及算子调度。

- `tmerge / --merge-recurrent-lora`：只合并 T 的 LoRA。
- `target_merge / --merge-target-lora`：在 batch_strict 上额外合并目标模型 LoRA。

底层目标模型调用 `merge_and_unload(safe_merge=True)`；本轮在三组入口中加入 `MERGE_TARGET_LORA`、CLI 和结果审计。T 合并检查实际合并模块数，目标合并检查 summary 的 `target_lora_merged`。

合并只作用于推理进程中的模型，不写回 checkpoint。FP/BF16 下加法和矩阵乘法顺序变化可能影响舍入、argmax 和生成轨迹；代数等价不能代替实际 token 一致性检查。

**目标合并必须同时用于 greedy 和投机。** 当前两者使用同一个合并后目标模型；不能拿合并后的投机与未合并 greedy 比较，再把全部收益称作投机收益。

## 7. 草稿读出：batch 和 parallel

### 7.1 batch：先得到多个预测 hidden，再一次读出

已有逐步路径可抽象为：

```text
预测 hidden1 → boundary(hidden1)
预测 hidden2 → boundary(hidden2)
```

批量路径先递归得到 hidden 序列，再拼接为块，一次调用 boundary/LM head，减少小矩阵调用和调度次数。T 的时间依赖仍然存在，**不是把递归 T 各步都改成并行**。

要求 boundary 路线且没有 token conditioning；否则下一步可能依赖当前读出的 token，不能任意把读出全部后移。当前 checkpoint 的 `token_conditioning_rank=0` 符合条件。本轮强化不兼容检查，避免参数写了 batch，实际却静默回退。

### 7.2 parallel：当前读出与下一步 T 重叠

利用已有双 CUDA stream 路径，让同一 hidden 的 boundary 读出与下一步 T 计算重叠；明确建立流间等待并记录张量/缓存的 stream 使用，避免生命周期问题。

理想上重叠部分由“两个时间相加”接近“较长一个的时间”，但实际可能受资源争用、同步及小任务开销影响，不能预报加速。

batch 和 parallel 是不同路径，脚本与解码器拒绝同时启用。BLOCK=2 只需要一枚草稿，可能没有额外的下一步 T rollout，不能仅因 t_step_s=0 就判定未使用 T。

## 8. 本轮新增：轻量严格验证 fast_strict

主要位置：解码器的 `validate_fast_strict_verification()`、`strict_verify_prefix()`、`speculative_decode()`。

### 8.1 旧路径的额外工作

通用验证路径兼容多种接受策略，会计算相对支持度、概率或 entropy、逐 token 诊断记录、margin 和策略分支。对于固定长度、lambda=1 的严格 argmax 匹配，其中部分工作不影响最终接受结果。

### 8.2 新路径如何计算接受前缀

令 m_i 表示第 i 枚草稿是否匹配对应目标 argmax，则接受草稿数为：

```text
A = sum_i product_{j=1..i} m_j
```

实现与当前代码对应：

```python
targets = logits[0, :draft_count].argmax(dim=-1)
matches = targets.eq(block[0, 1:])
accepted_drafts = matches.to(torch.int64).cumprod(dim=0).sum()
corrective = targets.gather(
    0, accepted_drafts.clamp(max=draft_count - 1).reshape(1)
)[0]
prefix, token, matched = torch.stack(
    (accepted_drafts, corrective, matches.sum())
).cpu().tolist()
```

draft_count=0 有单独分支。函数返回的 accepted_len 包含 anchor，所以为 `1 + prefix`；全部接受时没有当前块的纠正 token。

`matches.sum()` 只是各位置匹配数，**不等于连续前缀接受数**。例如 matches=[0,1]，接受数仍为 0。

新路径在 GPU 上算连续前缀，只把接受长度、纠正 ID 和匹配计数打包回传一次；保留必要的接受计数、block_records 和纠正/KV 分支，省掉通用逐草稿诊断路径。

### 8.3 草稿读出也做了精简

无须 margin 时直接 argmax，省去 FP32 转换和 top-k=2。batch 路径一次回传整块 token ID，不再每枚分别 `.item()`。但当前仍有 `.cpu().tolist()`，并未实现草稿 token 全程留在 GPU。

### 8.4 保持适用边界

要求 compact stats、fixed/schedule、神经草稿、全位置严格 target_match/lambda=1；拒绝 n-gram、宽松/类别接受、未验证提前提交、preemptive lookahead、相关串行回放等不支持的组合。

strict + diagnostic 时仍保留 compact stats，以便测同一个快速路径；仅恢复组件同步计时。

该改动精简的是**验证决策和统计**，没有减少目标模型层数，也没有用轻量头代替完整目标前向。块内严格规则成立，不意味着浮点环境下已经证明与串行 greedy 完全一致。

## 9. target_hook：只保留所需边界 hidden

在目标 decoder 的 `anchor_idx - 1` 层注册 forward hook，捕获该层输出作为 T 所需的输入边界；目标 forward 设置 `output_hidden_states=False`，随后仅提供所需边界状态。

原理是减少返回全部层 hidden 的引用保留和容器管理。完整目标模型的所有层仍照常计算，未跳过前后层，也不是省去了所有 hidden 的计算。

`target_hook` 档位 = `target_merge` + `--anchor-hook-hidden-states`。这个底层 hook 已存在，本轮将其组合为可直接对照的档位。hook 也存在自身管理成本，需实测净收益。

## 10. 计时与执行审计

### 10.1 两种计时口径

| 模式 | 组件计时 | 整段耗时 | 用途 |
|---|---|---|---|
| diagnostic | 每个组件前后 CUDA synchronize，记录同步墙钟时间 | 起止同步 | 定位开销；包含同步扰动及主机部分，不是纯 kernel event 时间 |
| throughput | 不逐组件同步，分项是主机提交时间 | 仍然起止同步 | 报告端到端吞吐 |

吞吐模式不能将 verify_s、T 等主机分项当成 GPU 时间相加，也不能拿这些数从整段时间中扣除 prefill，推算精确 decode-only 速度。本轮将无法可靠计算的派生项留空。

### 10.2 新增执行证据

- `batched_boundary_blocks`：批量 boundary 块数。
- `parallel_draft_steps`：实际并行草稿步数。
- `fast_strict_blocks`、`fast_strict_draft_tokens`：快速严格验证覆盖范围。
- 原有纠正复用/延迟纠正计数也进入三组报告。

`audit_runtime_optimizations()` 检查逐样本计数与 summary 一致、请求的路径实际覆盖相应块，并拒绝缺失计数的旧解码器结果。

`common.py` 的路由审计同时识别 T 初始化、同步和并行路径的执行证据，避免把 BLOCK=2 或重叠计时下 t_step_s=0 误判为未使用 T。

GPU 审计沿用已有监控。`provisional_overlap_or_monitor_error` 不能单凭名称认定抢卡：容器 PID 与宿主 GPU PID 不一致也可能导致误判。本轮未解决这一 PID 映射问题。

## 11. 已回传的实测结果和解释

### 11.1 前四档：单次运行

详细资产核查见已有 [speed_review_20260916.md](speed_review_20260916.md)。

| 档位 | fixed 秒 | 耗时加速 | 吞吐加速 |
|---|---:|---:|---:|
| reuse | 196.634 | 1.001× | 0.994× |
| defer | 195.768 | 1.002× | 0.977× |
| cache | 185.696 | 1.039× | 1.013× |
| tmerge | 180.302 | 1.070× | 1.043× |

defer 减少纠正调用，但增加了草稿轮次；不是减少调用就必然更快。cache/T 合并带来改善，同时必须看各自 greedy 是否漂移。不同生成长度使耗时比与吞吐比不同。

### 11.2 最新三档：同一 32 题、每档重复三次

配置：checkpoint-60375、GSM8K [0,32)、boundary、BLOCK=3、BF16、max_new_tokens=1024、defer、throughput，after_start=128。

| 档位 | greedy 总秒数 | fixed 总秒数 | greedy tok/s | fixed tok/s | fixed 吞吐加速 | after_start 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| batch_strict | 545.937 | 498.856 | 34.400 | 36.696 | 1.067× | 0.996× |
| target_merge | 267.922 | 265.621 | 68.830 | 69.358 | 1.008× | 0.976× |
| target_hook | 254.243 | 244.945 | 72.533 | 75.213 | 1.037× | 1.021× |

原始运行目录：

```text
timing_speed_batch_strict_20260916_182559_8269
timing_speed_target_merge_20260916_185451_9920
timing_speed_target_hook_20260916_191016_10960
```

结论及边界：

1. 目标合并后绝对吞吐接近翻倍，greedy 也同样受益；不能称为投机相对优化后 greedy 已加速两倍。
2. target_hook 绝对吞吐最好，但该组 greedy 同样比 target_merge 快约 5.4%；fixed 吞吐增加约 8.4%，按各自基线归一后的额外改善约 2.9%。没有隔离运行漂移前，不能将全部变化归因于 hook。
3. 合并后的 fixed 每轮平均接受 0.459 枚，零接受约 64.43%，全接受约 10.63%。接受长度仍是核心限制。
4. target_hook 每 32 题平均约 81.65 秒。保持相同输出量和当前 greedy 吞吐，1.3× 目标对应约 65.13 秒，还需减少约 20% 耗时；这是条件估算，不是承诺。
5. after_start 没有超过同档 fixed，不支持将“后期一定更快”作为当前结论。
6. 合并组 greedy 答对 22/32，fixed 答对 21/32，完整输出一致率 50%。重复三次不扩大独立质量样本；需要定位首次分叉，不能仅凭准确率接近就认定无损。
7. 所有结果仍标为 provisional，未确认 GPU 全程独占。该批无同步分项结果，不能套用旧配置的验证耗时占比。

另一次 adaptive + reuse 的同步分项为 verify 112.22 秒、correction 54.04 秒、总计 189.53 秒，两项约占 87.7%。它与本表的 defer/合并配置不同，只能解释那次运行。

## 12. 测试覆盖及实际验证范围

最近一次本地回归命令：

```bash
python -m unittest discover -s tests -q
```

记录结果为 **66 项测试，62 项通过，4 项因当前 Python 缺少 PyTorch/CUDA 跳过**。同时完成两个更新 Bash 入口的语法检查及 `git diff --check`。这是当时回归记录，本次文档整理未重新执行模型测试。

测试代码的覆盖包括：

- 真正调用原生解码参数解析，验证 shell 环境变量、CLI 优先级和各档开关。
- 构造解码结果验证共享 baseline、重复聚合、数据区间、资产/策略审计、起始位置与实际执行计数。
- 严格验证张量测试枚举连续匹配模式、首错后再匹配、argmax 并列及多种 dtype。
- 小模型执行实际 speculative_decode 控制流：3 种错误模式 × 3 种纠正方式 × 3 种块长 × 2 种批量设置 × 6 种起始位置/EOS 组合，共 324 个子场景，对照旧路径和快速路径。
- 具备 CUDA 时的双 stream 与串行对照。

“测试覆盖存在”与“本次环境执行通过”是两回事：上述被跳过的张量/GPU 检查不能计作本次通过。小模型测试也不能证明真实 Llama/BF16/H200 的逐 token 等价或性能收益。集群收益以第 11 节用户回传结果为据。

## 13. 档位总表和复现实验

| 档位 | 在哪一档上增加什么 |
|---|---|
| reuse | 验证 KV 复用，单独纠正前向 |
| defer | 改为纠正并入下一轮 |
| cache | defer + 原地 KV |
| tmerge | cache + T LoRA 合并 |
| batch | tmerge + 批量 boundary |
| parallel | tmerge + 双 stream 草稿 |
| strict | tmerge + 轻量严格验证 |
| batch_strict | tmerge + batch + strict |
| parallel_strict | tmerge + parallel + strict |
| target_merge | batch_strict + 目标 LoRA 合并 |
| target_hook | target_merge + 边界 hidden hook |

使用已填写的 MODEL/CHECKPOINT/DATA/GPU，可复测：

```bash
# 预检参数，不执行模型推理
SPEED_PROFILE=target_hook bash scripts/benchmark_decode_speed.sh --dry-run

# 既有三档对照
for profile in batch_strict target_merge target_hook; do
  SPEED_PROFILE="$profile" bash scripts/benchmark_decode_speed.sh \
    --samples 32 --block 3 --repeats 3
done

# 下一项低成本实验：最多 1、2、3 枚草稿
for block in 2 3 4; do
  SPEED_PROFILE=target_hook bash scripts/benchmark_decode_speed.sh \
    --samples 32 --block "$block" --repeats 3
done

# 当前配置的小样本分项诊断，不把诊断速度当最终吞吐
SPEED_PROFILE=target_hook bash scripts/benchmark_decode_speed.sh \
  --samples 4 --runtime-mode diagnostic
```

不能把 adaptive.sh 当作 speed.sh 的等价替代：两者的默认策略、计时和已接入优化参数不同。尤其 fast_strict 不直接支持当前 adaptive 的 heuristic 路径。

## 14. 同步依赖与尚未实现的方向

已经安装 batch_strict 完整版本时，最后一轮 target_merge/target_hook 增量只需更新：

```text
scripts/benchmark_decode_speed.sh
scripts/benchmark_decode_start.sh
scripts/benchmark_decode_start.py
```

对应包：[decode_verify_increment_20260916.zip](../artifacts/decode_verify_increment_20260916.zip)。该包依赖此前已经同步的解码器、common.py、run_decode.py，不是完整仓库。

更早的 batch/parallel/strict 包为 [decode_speed_123_20260916.zip](../artifacts/decode_speed_123_20260916.zip)，其后仍需上述最新增量。若从旧仓库整体补齐本轮入口，应同步第 3 节所有运行文件；测试文件和报告按需同步。修改过集群脚本顶部路径时，覆盖前应保存参数或转用环境变量。

以下仅提出过，**本轮没有实现**：

- 草稿 token 全程保留 GPU、合并多次主机决策回传。
- 固定预分配 KV、专门的回退位置管理。
- 编译目标/T 路径、按块长捕获 CUDA Graph。
- 自动解决 GPU 宿主/容器 PID 审计差异。
- 解决全部真实模型输出与 greedy 的不一致。
- 保证 1.3× 或 2× 相对同等级优化 greedy 的加速。

下一步建议先在 target_hook 上比较 BLOCK=2/3/4，再根据当前配置的分项或 GPU 时间线决定同步优化与静态缓存的投入；同时继续做首次分叉定位。最终配置在开发区间选定后，还需换未参与选参的数据区间验证。
