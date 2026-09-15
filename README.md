# LayerLoop / RecurFT Executable Code Export

完整的修改版 LLaMA-Factory 源码、递归训练实现和可运行入口。不是仅包含几个补丁的补充材料，也不需要重新拉取上游仓库。此包是 2026-09-12 工作区快照，不声称恢复了七月逐字节一致的代码和环境。

## 先明确执行路径

| 入口 route | 实际草稿路径 | 定位 |
| --- | --- | --- |
| `boundary`，默认 | 隐状态 -> 递归 T -> 低秩残差适配 -> final norm + LM head -> token | 真正的递归方法；384 样本受控实验采用此读出 |
| `tail` | 隐状态 -> 递归 T -> 边界后剩余 Transformer 层 -> norm + LM head -> token | 真正的递归方法；较贵的原始读出 |
| `ngram` | 文本历史查找 -> token -> 目标模型验证 | 明确命名的非递归对照，不作为 recurrent 收益 |

两条神经草稿入口均显式设置 `--ngram-draft-mode off` 和严格 target-match，不开放 unchecked commit、宽松接受或自动切到 n-gram 的参数透传。

2026-09-11 审计已确认：旧跨模型图的 24 个加速点全部是 n-gram-only，另 3 格是 greedy fallback，不能用其 1.295x 支持递归加速。真正的 N=384 递归受控实验使用 boundary，实测约 1.040x wall / 1.288x target-call reduction。详见 `provenance/execution_route_audit.json`。本代码包没有改写或重标那些结果。

## 内容

```text
LLaMA-Factory/src/                         完整框架源代码，含模型、LoRA、trainer、loss
LLaMA-Factory/experiments/recurft_math/    解码、rollout、数值审计及其直接依赖
LLaMA-Factory/tests/                      递归、损失、冻结参考与草稿相关单元测试
scripts/run_decode.py                    安全检查 + 显式路由 + 结果审计
scripts/benchmark_decode.sh              集中改参数、检查训练步数、重复测速并生成耗时简报
scripts/reproduce_headline.sh             N=384 / 两遍 / FP16 / K=3 协议入口
scripts/train.py                         四阶段训练入口，生成路径可移植的 YAML
scripts/self_test.py                     离线 CPU 单元与真实小模型端到端测试
scripts/check_assets.py                  checkpoint 文件和 SHA256 检查
scripts/fetch_checkpoint.py              可选：从自己的服务器取回推理 checkpoint
configs/reference/                      历史超参数配置
requirements-linux-cu121.lock.txt        当前服务器实际环境的可移植版本清单
provenance/                              代码来源、审计、历史指标、资产清单
MANIFEST.sha256                          全部交付文件校验和
```

核心文件：模型模块 `src/llamafactory/model/model_utils/recurft.py`；训练损失 `src/llamafactory/train/sft/recurft.py`；参数与加载逻辑 `hparams/finetuning_args.py`、`model/adapter.py`；训练入口 `src/train.py`；推理入口 `experiments/recurft_math/recurft_speculative_generate.py`。

## 环境和离线自测

正式运行目标为 Linux x86_64 / Python 3.11 / NVIDIA CUDA。当前服务器实测环境为 PyTorch 2.4.1+cu121、Transformers 4.57.1、PEFT 0.17.1。依赖清单来自 2026-09-12 的现有环境，不代表七月原始环境快照。

在已有 `otv311` 环境上，不必重装模型依赖。若没有 pytest，用继承现有包的独立测试环境，避免修改原环境：

```bash
source scripts/server_paths.example.sh
"$PYTHON" scripts/verify_package.py
"$PYTHON" -m venv --system-site-packages /tmp/layerloop-test-env
/tmp/layerloop-test-env/bin/python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES="" /tmp/layerloop-test-env/bin/python scripts/self_test.py
```

自测不下载模型、不访问数据集、不使用 GPU；现场创建一个随机初始化的微型 Llama 和 adapter/checkpoint，实际调用完整解码器跑通 boundary、tail 和 n-gram 对照。随机模型不用于质量或性能结论。

新机器可用 `bash scripts/install.sh` 创建本地 `.venv`。安装会访问 Python 包索引；不会修改已有服务器环境。仓库的 `pyproject.toml` 与实际环境在 PEFT、AV 等版本范围上有冲突，所以安装脚本采用观测版本、`--no-deps` 和独立 venv；运行入口设置 `DISABLE_VERSION_CHECK=1`。这不是通用版本兼容保证。源代码完整性和 CPU 实际执行测试比绕过声明本身更重要。

## 模型与数据不在代码 ZIP 内

需要已有的可信本地基座模型目录、训练过的 checkpoint 和数据文件。包内没有基座权重、训练/测试语料、密钥或 `.env` 文件，也不会自动下载它们。Meta-Llama 权重需要使用者自行取得授权。

对于已存在的 checkpoint-60375：

```bash
"$PYTHON" scripts/check_assets.py "$CHECKPOINT"
```

在自己的另一台机器上，需要取回推理 adapter、T/head 和 tokenizer 时：

```bash
python3 scripts/fetch_checkpoint.py --host 117_jump_vpn --output assets/checkpoint-60375
```

该可选操作约 163 MB，逐文件校验哈希；不包含基座模型和训练恢复状态。继续训练还需要来源 checkpoint 的 `trainer_state.json`，严格恢复优化器还需要对应 optimizer/scheduler/RNG 状态；原历史 `save_only_model` checkpoint 并不提供完整优化器恢复保证。

评测 JSONL 每行必须有 `question` 和 `answer`，不允许空行，选取区间内问题必须唯一。训练数据为 MetaMath 的 `query` / `response` JSON。不要把自造小样本文件当正式测试集。

## 先预检，再运行

```bash
source scripts/server_paths.example.sh
"$PYTHON" scripts/run_decode.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --data "$DATA" --output "$PWD/runs/boundary_smoke" --gpu 5 \
  --route boundary --samples 2 --max-new-tokens 64 --dry-run
```

`--dry-run` 只检查文件、数据覆盖并打印命令。确认后移除它才会执行。比较后半网络读出时，使用新的输出目录并把 `--route boundary` 换成 `--route tail`。基座架构必须与 checkpoint 一致；不能用 Llama 的 adapter 跑 Qwen。该轻量入口固定为数学任务提示；其他任务需要显式适配任务协议，不能直接混入主表。

输出含 `result.json`、`run.log`、`inputs.json`、启动/完成/失败 marker 和 GPU 观察日志。已有输出目录绝不覆写。完成 marker 表示样本与执行路径审计通过，不代表质量通过；wall < 1 或质量退化仍会如实保留。

GPU 入口要求空闲显存至少 51200 MiB、util <= 20、无 compute PID（包括未澄清的驱动残留 PID）。使用同用户/设备协作锁防止本包重复启动；不杀任何外部进程。运行中每五秒采样，外部重叠或监控失败标为 provisional。`sampled_exclusive` 仅表示采样时未见重叠，不是连续独占或正式 clean wall 的证明。

## 自定义 checkpoint 测速

在 Linux GPU 节点修改 `scripts/benchmark_decode.sh` 开头的参数，至少填写 `MODEL`、`CHECKPOINT`、`GPU`，并将 `DATA` 指向已有的 GSM8K 测试 JSONL。也可以通过环境变量覆盖参数：

```bash
export MODEL=/path/to/Meta-Llama-3-8B-Instruct
export CHECKPOINT=/path/to/checkpoint-xxxxx
export GPU=GPU-your-allocated-device-uuid
export DATA=/path/to/gsm8k_test.jsonl
bash scripts/benchmark_decode.sh --dry-run
bash scripts/benchmark_decode.sh
# 增加题数、比较单枚草稿，重复两遍：
SAMPLES=128 BLOCK=2 REPEATS=2 bash scripts/benchmark_decode.sh
```

默认参数为 `ROUTE=auto`、`SAMPLES=32`、`MAX_NEW_TOKENS=256`、`BLOCK=3`、`DTYPE=bf16`、`REPEATS=1`。`BLOCK=3` 为一枚目标模型已确定的 token 加最多两枚草稿。`auto` 根据 `boundary_head_rank` 选择 `boundary` 或 `tail`；头部配置存在不代表已经充分训练。core / multistep 权重通常使用 `tail`，完成头部训练后可显式设置 `ROUTE=boundary`。启动前打印 checkpoint 的累计 `global_step`；没有 `trainer_state.json` 的推理权重仍可评测，但无法从该文件确认步数。

默认优先使用项目 `.venv/bin/python`，其次使用当前环境的 Python；可显式设置 `PYTHON`。`OUT` 留空时自动生成实验目录，每次重复写入 `repeatN` 子目录；已有 `OUT` 会被拒绝。结果包括原有 `result.json`、`run.log` 和新增的 `timing_summary.txt`，简报打印速度、输出长度、草稿接受率、输出一致率和降序排列的分项耗时。每遍都比较同一 checkpoint 的 greedy 与投机解码，保持严格 target-match、n-gram off 和已有 GPU 检查。使用默认诊断计时，包含 prefill 和生成，不包含模型加载；该便捷入口不等同于下面的历史 N=384 协议。

`--dry-run` 只做文件和参数预检，不下载资产、不占用 GPU，也不创建输出目录。该脚本尚未在用户集群上完成 GPU 测速。

## 启用缓存优化的测速

新入口 `scripts/benchmark_decode_cache.sh` 默认启用 `CORRECTION_MODE=reuse`，支持已有 core/multistep 的 tail 路线和带头部 checkpoint 的 boundary 路线，不需要为缓存优化重新训练。填写脚本顶部的 `MODEL`、`CHECKPOINT`，确认数据和分配的 GPU；其他参数也可用同名环境变量覆盖。

```bash
export MODEL=/path/to/Qwen3-8B
export CHECKPOINT=/path/to/checkpoint-xxxxx
export DATA=/path/to/gsm8k_test.jsonl
export GPU=0
bash scripts/benchmark_decode_cache.sh --dry-run
bash scripts/benchmark_decode_cache.sh

# 分别测试延迟纠正和原路径，保持模型、样本、长度等条件一致：
CORRECTION_MODE=defer bash scripts/benchmark_decode_cache.sh
CORRECTION_MODE=off bash scripts/benchmark_decode_cache.sh
```

- `reuse`：传递 `--reuse-verify-cache-for-correction`，保留验证阶段的前缀 KV，只运行纠正 token。
- `defer`：仅传递 `--defer-correction-to-next-verify`，保留验证前缀并将纠正 token 留到下一轮验证，避免额外的独立纠正前向。
- `off`：不传递上述两个开关，使用原纠正路径。原 `benchmark_decode.sh` 默认仍为 `off`。

三种模式互斥，默认一次只跑所选模式。`ROUTE=auto` 按头部配置选择 tail/boundary；默认 32 题、block 3、BF16、重复 1 遍。继续使用严格 target-match、n-gram off、相同 checkpoint 的 greedy 基线、原 GPU 检查和同步分项计时，不混入其他运行优化。自动输出目录包含纠正模式；显式 `OUT` 必须为新目录。

`timing_summary.txt` 增加纠正模式、`fast_correction_cache_reuses`（缓存复用次数）和 `deferred_corrective_tokens`（延迟纠正次数）。`inputs.json` 和 `complete.marker.json` 保存模式；原始 `result.json` 的 `summary.args` 保存实际开关。入口会核对开关及每条样本的计数之和；`correction_optimization_observed: false` 表示开关配置正确但本次没有触发相应分支，不代表已经获得优化收益。

可直接使用底层包装器，例如 `python scripts/run_decode.py ... --correction-mode reuse`。新脚本依赖更新后的 `benchmark_decode.sh`、`run_decode.py` 和 `common.py`，同步时应拉取整个提交。缓存路径改变可能影响浮点计算结果；比较速度时继续检查输出一致性和任务质量。12 项新增测试覆盖开关传递、计数检查，以及真实 Bash 到解码命令的 dry-run；未在用户集群验证 GPU 性能或数值等价性。

## N=384 受控协议

```bash
source scripts/server_paths.example.sh
export GPU=5
export OUT="$PWD/runs/headline384_new"
DRY_RUN=1 bash scripts/reproduce_headline.sh
# 检查路径与数据后，以下命令才会执行两遍实验：
bash scripts/reproduce_headline.sh
```

此入口使用 GSM8K 物理行 8..391、384 个样本、最多 256 新 token、prompt cap 1024、K=3、FP16、non-thinking、strict target-match、boundary 和 n-gram off。默认 baseline 是同一个适配后 target 的逐 token greedy，不是未适配基座。当前代码/内核版本与硬件可能使结果不同；旧数值只是参考，不是启动脚本预期必须得到的答案。

## 训练完整链

```bash
# 新输出目录，不会覆盖旧训练。
python3 scripts/train.py --stage core --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --output runs/core --gpu 5 --dry-run
python3 scripts/train.py --stage multistep --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-49375 \
  --output runs/multistep --gpu 5 --dry-run
python3 scripts/train.py --stage boundary-warmup --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-59375 \
  --output runs/boundary_warmup --gpu 5 --dry-run
python3 scripts/train.py --stage boundary --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-59600 \
  --output runs/boundary --gpu 5 --dry-run
```

移除 `--dry-run` 才启动对应阶段。入口会为每个训练单独生成 dataset_info 与 YAML，不改共享仓库数据注册表。`--steps N` 是新增训练步数，会改变历史训练协议，不能直接称为原实验复现。

历史晚层链：core 49375 步 -> multistep 59375 -> boundary warmup 59600 -> boundary 60375。core 配置来自此前基于保存参数重建的补充材料，非原始 YAML 字节副本；另三个配置直接保留当前工作区历史文件。默认仅适用于该 Llama-3-8B 晚层链，其他模型须重新确定层索引、模板、adapter 与 checkpoint。

## T 与 boundary 联合训练（实验阶段）

`joint` 从已有 boundary checkpoint 继续，冻结目标模型及其 LoRA，开放 recurrent 模块中原本可训练的参数（默认架构为 T LoRA 和 boundary 头）。两步 rollout 使用预测 hidden；`detach_rollout: false` 让第二步损失也能沿状态路径反向更新第一步。当前配方是待验证的实验起点，不代表已经提高接受率或达到某个加速倍率。

配置位于 `configs/experimental/recurft_joint_boundary.yaml`。它叠加在该 checkpoint 对应的 `train.yaml` 上，保留模型结构参数，并检查循环层、T rank、boundary rank 与 checkpoint 元数据是否相符。只支持带 `trainer_state.json`、不含旧 optimizer/scheduler 状态的 checkpoint；历史 `save_only_model: true` 保存格式符合要求。原 checkpoint 保留，新阶段重新建立优化器；`--steps` 表示新增优化器更新次数，预热起点自动设为加载权重的 global_step。

```bash
MODEL=/path/to/Qwen3-8B
DATA=/path/to/MetaMathQA-395K.json
CKPT=/path/to/boundary_run/checkpoint_output/checkpoint-60375
SOURCE_CONFIG=/path/to/boundary_run/train.yaml

python scripts/train.py --stage joint \
  --model "$MODEL" --data "$DATA" --checkpoint "$CKPT" \
  --source-config "$SOURCE_CONFIG" --template qwen3_nothink \
  --output runs/joint_trial --gpu 0 --steps 2000 --dry-run
# 检查输出的 YAML 后，移除 --dry-run 执行。
```

Qwen3 非思考模式使用 `qwen3_nothink`；入口会拒绝 Qwen3 搭配 `llama3`。已使用错误模板训练的权重并不会因为改了模板就恢复，需要在新模板下重新检查目标模型质量与草稿匹配率。不要在联合阶段更换基座模型。

如果现有权重只有 core/multistep、`boundary_head_rank: 0`，先运行头部预热。以下示例适用于当前 29–30 层、pre/post rank 8、T rank 128 的配置；其他结构需先匹配对应配方：

```bash
python scripts/train.py --stage boundary-warmup \
  --model "$MODEL" --data "$DATA" --checkpoint /path/to/core_checkpoint \
  --template qwen3_nothink --output runs/head_warmup --gpu 0 --steps 225 --dry-run
# 完成实际预热后，joint 的 --checkpoint 和 --source-config
# 分别指向此次预热保存的 checkpoint 和 runs/head_warmup/train.yaml。
```

联合阶段的监督来自同一训练文本前缀下、固定的适配后目标模型分布。当前默认不启用数据标签 CE，优先用 KL 训练草稿匹配目标模型；这仍是基于训练文本的 hidden rollout 蒸馏，不等于完整的模型生成轨迹训练，也没有直接优化两枚 token 的联合接受事件。

关键参数：`recurft_stage1_heads_only: false`、`recurft_recurrent_trainable_only: true`、`recurft_multistep_steps: 2`、`recurft_multistep_detach_rollout: false`。`recurft_multistep_loss_weight: 0.1` 会乘到 rollout 内所有损失上，因此 boundary KL 内部权重设为 `10.0`，预热后的有效权重为 `1.0`；基础 rollout hidden 损失权重为 `0.1`。单步 recurrent 权重也保留 `0.1`：当前实现用它作为整个 recurrent/rollout 分支的入口，不能直接设零。两项普通 boundary CE/KL 设零，以免混入真实 hidden 上的旧头部训练目标。

先观察 `recurft_multistep_boundary_teacher_top1_k1`、`..._k2`、`recurft_multistep_boundary_logit_kl_k1`、`..._k2`。这些是训练前缀上的分步指标，不是在线两步连续接受率。每个候选 checkpoint 仍应使用 `ROUTE=boundary` 在独立验证集上测接受率、输出差异、任务质量和 wall speed；最终评测集不要用于反复选择超参数。2000 步只是首轮实验长度，不保证训练充分。

此入口已通过 11 项配置/实际 CLI dry-run 检查，未在集群执行联合训练，也未重新运行完整 ML 梯度测试。原四阶段配置保留，旧 CPU 测试报告不代表这项新实验已经验证。

## 验证范围

本次实际通过 87 项单元/参数化测试，以及 boundary、tail、ngram 三条 CPU 小模型端到端解码；完整框架源码已成功构建 wheel。真实 checkpoint 的 8 个推理文件哈希、GSM8K 的 384 样本输入覆盖、训练入口 dry-run 均已检查。

`provenance/cpu_test_report.json` 记录测试结果、环境版本和对应代码 SHA256，导出器拒绝把报告绑定到修改后的代码。该测试不是全量 8B 训练/评测，也没有重新测量论文 wall。没有在全新机器上重装并验证整套 CUDA 依赖；下载与安装仍取决于网络、wheel 可用性和 NVIDIA 驱动。本包保留上游 Apache-2.0 LICENSE；个人路径与历史审计存在于 provenance，此包是研究代码交付，不是匿名投稿附件。
2026-09-14 评分修复：数学评分现在从官方 GSM8K 原始 `answer` 的 `####` 后提取标准答案，仍兼容已经只保留最终答案的数据。修复前直接使用官方原始 JSONL 会把完整推导当作标准答案，从而错误判零；这个问题不影响已记录的生成 token 或耗时。修复通过 5 项独立评分回归检查（`python tests/test_math_scoring.py`），未重新执行 GPU 实验。上面的历史 CPU 报告仍绑定导出时的代码快照，不代表已验证本次修改后的全部模型路径。
