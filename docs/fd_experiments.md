# FD 在线筛选：运行与分析

## 执行边界

本项目运行我们的方法及精确余弦、不过滤、随机匹配对照，不重跑 LESS。
复用 LESS 的数据来源和测试任务。Qwen3.5-4B 的结果不能直接当作原论文
Llama/Mistral 上的同模型受控比较。已发表的 LESS 数字应注明来源及模型/协议差异。

所有训练使用相同初始化种子 `--model-seed 0`，候选顺序种子分别为 3/6/9。
LoRA r=16、alpha=64、dropout=0，基座 bf16，LoRA FP32；在 eval 模式保留 autograd。
每条样本先对监督 token 求平均，更新再按全局保留样本数等权平均。
AdamW lr=1e-5，无 weight decay、无调度，梯度裁剪 1.0。

## 数据准备

先执行只读预检，保存数据检查和可复用产物清单：

```bash
python -m less.data_selection.fd_preflight --data-dir data --artifact-root out \
  --model-path "$MODEL_PATH" --output out/preflight/data_audit.json
```

`--strict` 会在保存检查结果后，对数据缺失或参考/测试重复返回非零退出码。
这也能在服务器上自动查找旧的 metrics/checkpoint 元数据；不会据说明文件猜测实验完成。

原始数据仍使用 LESS 处理后的 Flan V2、CoT、Dolly、OpenAssistant 四个来源。
本地的三个三行文件是 Git LFS 指针，不能用于正式实验。
实际数据来自 https://huggingface.co/datasets/princeton-nlp/less_data 。
从原数据仓库获取正文时可用 Git LFS；下载后应与指针中的 SHA256/size 对照，
不要把指针文本当作三条训练样本。准备完整数据后：

```bash
python -m less.data_selection.prepare_fd_pool \
  --data-dir data --output-dir data/train/pool40k --size 40000 --seed 2026
```

`pool_manifest.json` 保存全部源文件行数、SHA256、输出哈希及每条样本的来源行号。
抽样为均匀无放回抽样，落盘顺序保持源文件顺序；每个运行再按候选顺序种子打乱。
工具不自动下载或替换数据。若使用全池，`--train-files` 可以依次列出四个源文件。

## 标定 epsilon 与判定保真度

先跑单卡 20 批 cosine/fd 冒烟，再跑一次精确 cosine 轨迹以保存中间参数。
全局 batch=16、40k 池一轮为 2500 批，可在 500、2000 批取快照。
20k 池一轮只有 1250 批，应事先登记改用 250、1000 批，不能请求不存在的快照。

```bash
FILTER=cosine OUTPUT_DIR=out/online/mmlu/cosine/s3 \
  SAVE_EVERY_BATCHES=500 bash less/scripts/online_proxy_train.sh

python -m less.analysis.fd_fidelity \
  --model-path "$MODEL_PATH" --task mmlu --data-dir data \
  --train-files "$TRAIN_FILE" --output-dir out/E1/initial --reference-mode fresh

python -m less.analysis.fd_fidelity \
  --model-path out/online/mmlu/cosine/s3/snapshot-batch-500 \
  --base-model-path "$MODEL_PATH" --task mmlu --data-dir data \
  --train-files "$TRAIN_FILE" --output-dir out/E1/b500 --reference-mode fresh

python -m less.analysis.fd_fidelity \
  --model-path out/online/mmlu/cosine/s3/snapshot-batch-2000 \
  --base-model-path "$MODEL_PATH" --task mmlu --data-dir data \
  --train-files "$TRAIN_FILE" --output-dir out/E1/b2000 --reference-mode fresh

python -m less.analysis.fd_fidelity --output-dir out/E1 \
  --select-from out/E1/initial/fidelity.json out/E1/b500/fidelity.json out/E1/b2000/fidelity.json
```

输出长表 `fidelity.csv` 和各配置指标 `fidelity.json`。同时记录三值符号一致率、
实际保留判定一致率、零差分比例、强信号一致率、Spearman、相对误差、AUC、kappa、Jaccard。
重复前向的噪声底只反映非确定性，不能代替舍入误差检查。
`choice.json` 至少需要三个不同参数点，实际选中的配置须达到 90%/97% 门槛。
不通过时停止主实验；可单独用 `--dtype float32` 重新标定，不静默改变精度。
如果主实验改 FP32，其他受控在线方法也必须采用相同 dtype，并重新报告成本。

E2 读取快照缓存的参考梯度；候选从快照之后的真实 batch 取，但精确梯度和 FD
都在该固定参数点重新计算，不使用历史日志在其他参数状态下的 cosine：

```bash
python -m less.analysis.fd_fidelity \
  --model-path out/online/mmlu/cosine/s3/snapshot-batch-500 \
  --base-model-path "$MODEL_PATH" --task mmlu --data-dir data \
  --train-files "$TRAIN_FILE" --reference-mode cached \
  --batches-log out/online/mmlu/cosine/s3/batches.jsonl --after-batch 500 --num-batches 32 \
  --epsilon 0.003 --scheme forward --output-dir out/E2/b500
```

其中 epsilon/scheme 必须替换为 E1 的选择。默认的标定抽样与 E2 分别记录候选 ID，
任务与参数点不会混同。零梯度只由离线精确诊断识别；FD 主运行仅报告零差分。

## 正式训练与随机匹配

对每个 task、seed 使用独立的空输出目录，逐条执行。先完成 FD 再运行 random。
不用固定 5% 保留率，也不根据测试结果调整阈值。

```bash
TASK=mmlu SEED=3 FILTER=none OUTPUT_DIR=out/online/mmlu/none/s3 \
  bash less/scripts/online_proxy_train.sh

TASK=mmlu SEED=3 FILTER=fd FD_CHOICE=out/E1/choice.json OUTPUT_DIR=out/online/mmlu/fd/s3 \
  bash less/scripts/online_proxy_train.sh

TASK=mmlu SEED=3 FILTER=random OUTPUT_DIR=out/online/mmlu/random/s3 \
  RANDOM_MATCH_LOG=out/online/mmlu/fd/s3/batches.jsonl \
  bash less/scripts/online_proxy_train.sh
```

精确版 s3 可复用标定所用完整轨迹，其他种子另跑。
random 会校验源数据哈希、种子和逐批候选顺序；保留数必须逐批一致。
none/random 不需要参考梯度，实际省略该开销，并在 run.json 中明确记录。
FD 与 cosine 共享相同缓存、每步参考熵检查、过期全拒绝刷新重筛规则。

## 单卡与分布式

默认仅使用 `CUDA_VISIBLE_DEVICES=0`。多卡改为 `0,1,...` 并设置 `NPROC`。
也可直接 `torchrun --standalone --nproc_per_node=2 -m less.train.online_proxy ...`。
多机可由外部 torchrun 设置 rendezvous，Python 入口使用标准 RANK/WORLD_SIZE/LOCAL_RANK。

全局候选 batch 固定为 16，微批 `FD_BATCH_SIZE=2` 只控制每卡前向和更新占用。
每卡独立计算梯度和，all-reduce 后除以全局保留数；不使用 DDP 的 rank 均值。
没有参考样本或没有保留样本的 rank 也参与同步，失败信息在后续 collective 前传播。
浮点归约顺序可能导致接近零边界的差异，GPU 精度仍须实测。

## 测试与 LESS 协议对齐

原入口 `evaluation.eval.qwen` 默认仍是 `qwen-native`，保持旧命令语义。
新脚本默认选择独立的 `--protocol less`：

| 项目 | LESS 兼容配置 |
|---|---|
| MMLU | 原始 subject/Answer 提示、5-shot、超过 2048 token 时减少 shot、候选空格前缀最后一个 token、主分数按题目加权；同时保存学科宏平均 |
| BBH | 原 cot-prompts、3-shot、每任务最多 40 条、greedy 最多 512 token、双换行停止、原式答案提取与忽略大小写/标点 EM |
| TyDiQA | 每语言最多 200 条、1-shot gold passage、512-token passage 截断、原语言模板及 Tulu 包装、greedy 最多 50 token、换行停止、F1/EM |

**历史样本完全一致不能仅凭 seed 证明。** 原脚本依赖 glob/set 遍历顺序，且当前
只提供固定 TyDiQA one-shot 文件，没有原始 train 抽样记录。该配置采用可重现的排序
抽样、提供的 one-shot，并明确写入 protocol，不能声称逐条复现原论文的未知抽样。
与已有 LESS 结果严格比较时应取得其评测清单；否则结果仅标为相同 benchmark 设置的参考比较。
这是数据来源限制，程序不补造历史清单。所有直接比较的模型使用同一导出清单：

```bash
python -m evaluation.eval.qwen --protocol less --task mmlu --data_dir data \
  --check-data --export-eval-manifest out/protocols/mmlu.json

RUN_DIR=out/online/mmlu/fd/s3 TASK=mmlu EVAL_MANIFEST=out/protocols/mmlu.json \
  bash less/scripts/online_proxy_eval.sh
```

每次评测保存逐题预测、gold、few-shot ID、提示哈希、生成长度、停止原因与逐题得分。
缺题、重复题、超长提示会报错，不生成部分指标。
原论文按验证集选择 checkpoint，本项目当前主协议评测完成的一轮训练终点，
这一模型选择差异也必须在跨论文比较中注明，不能声称整个实验条件完全相同。

当前数据的 MMLU dev/test 有 5 条问题文本重复。`less` 配置保留原 14,042 题及原口径
的 `score`，并同时保存 `reference_overlap_ids`、`clean_n=14037` 和排除这些题的
`clean_score`；不能把原口径结果写成零重叠评测。`qwen-native` 原有去重行为仍保留。
如正式报告要求参考/测试零重叠，使用去重口径并明确其与原 LESS 测试集的五题差异。

## 保存结构与字段

```text
out/online/<task>/<method>/s<seed>/
  run.json                配置、版本、代码哈希、数据哈希、参考 ID、环境和 GPU
  status.json             initializing/running/completed/failed 与失败 traceback
  console.log
  batches.jsonl           每批候选/决策/重筛、保留数、更新数、耗时、显存、前反向样本数
  decisions.jsonl         每样本每次尝试一行，标记最终尝试，适合 pandas/SQL 分析
  accepted_ids.jsonl      真正参与更新的样本及来源行号
  refreshes.jsonl         初始/熵漂移/间隔/全拒绝刷新及参考 loss、熵、梯度范数
  checkpoints.jsonl
  snapshot-batch-500/     adapter、tokenizer、online_state.pt、checkpoint.json
  model/                 最终 adapter 和完整在线状态
  train_results.json      本执行片段汇总，区分冒烟/完整池与续训
  eval/                  eval_manifest.json、predictions*.jsonl、metrics.json
```

核心决策字段为 `pool_index,id,accepted,reason,cos,dot,gradient_norm,delta,dir_deriv,
loss_base,loss_perturbed,loss_plus`。方法无法直接测得的量保存为 null，不伪装为零。
`batches.jsonl` 的 batch_index 从 0 开始；快照名称 N 表示已处理 N 批，下一批 index=N。
中心差分为保存可分析的 base loss 额外计算 θ 处前向，计入真实成本。

阶段时间同步 CUDA 后测量，取各 rank 最大值；前反向样本数全 rank 求和；显存报告
各 rank 的最大 allocated 峰值（十进制 GB）。训练 wall time 包含初始化、日志和保存。
参考熵/参考梯度开销分别记录，none/random 的缺省开销不会被人为补齐。
基座记录配置哈希及权重文件大小/mtime，未声称计算完整数 GB 权重的内容哈希。

## 断点续训

续训写入新空目录，保留原始失败/完成日志。快照包含 Adam 状态、缓存参考梯度、
参考熵、有效更新数、候选位置与 RNG。要求输入哈希及轨迹配置完全匹配。

```bash
python -m less.train.online_proxy --model-path "$MODEL_PATH" --task mmlu \
  --data-dir data --train-files "$TRAIN_FILE" --filter fd \
  --resume-from out/online/mmlu/fd/s3/snapshot-batch-500 \
  --output-dir out/continuations/mmlu/fd/s3
```

同时传入与原运行一致的 epsilon/scheme/seed 等配置。汇总保留每个片段，
不会把续训片段成本当作完整运行成本或把片段当作独立 seed。

## 汇总与统计

```bash
python -m less.analysis.summarize_fd --root out --output out/report
```

产出 runs、main_results、batches、decisions、refreshes、paired_differences、
paired_summary、question_comparisons、speedups、audit、fidelity_grid、fidelity_buckets、
fidelity_batches CSV，及 summary.json、PNG 图。逐样本 CSV 流式导出，不在 summary.json 中复制全部样本。
主表要求完整池一轮完成、完成评测且非续训片段；没有评测的运行只进成本表。
不同数据哈希、模型、评测协议不合并，同配置重复 seed 会报错。
主比较为 FD vs cosine/random/none；三 seed 报均值和样本标准差。
逐题 bootstrap 按子任务分层、保持该协议的 micro/macro 口径；另报二元 McNemar 与 Holm。
不把不同任务的九个点盲目视为独立同分布样本。三 seed 的区间只作为有限证据。
