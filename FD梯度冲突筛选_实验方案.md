# 实验执行规格：基于前向有限差分的低成本在线梯度冲突筛选

> **2026-10-18 执行口径更新（按用户后续要求）**
> - 当前设备为单张 A800，Python 训练入口必须兼容 torchrun 多卡；全局候选 batch 不随卡数改变。
> - **不运行 LESS 实验**：下文旧阶段中 offline random/transfer/LESS 命令、M5–M7 不属于本次执行范围。LESS 仅作为测试协议/已发表参考结果的来源，不能生成未测量的本地对比分数。
> - 测试新增独立 `--protocol less`，与现有 Qwen-native 入口分开。原脚本 MMLU `average_acc` 实为题目加权平均；不再将学科宏平均当作相同主指标。具体抽样、提示与原历史清单缺失的限制见 `docs/fd_experiments.md`。
> - 原文“在线精确余弦已实现/已核实”的前提与提供代码不符；本次补建 `online_proxy_core.py`，以新增 CPU/分布式测试作为正确性依据。GPU Qwen 冒烟与 E1 门槛仍须实际执行。
> - E2 在同一快照参数和同一缓存参考梯度下重新计算精确/FD，不使用后续训练日志的历史 cos 作真值。20k 子池的快照改为预登记的 250/1000 批。
> - FD 主运行记录零差分，真正零梯度仅在离线精确诊断中识别。E1 门槛检查实际选中的配置。
> - 每次运行保存配置/代码/数据哈希、归一化逐样本决策、逐阶段计时、刷新事件、checkpoint、失败状态与逐题评测。详细可执行命令、续训与汇总口径以 `docs/fd_experiments.md` 为准。

> 本文件是给执行智能体的完整任务书。按第 7 节阶段顺序执行，每个阶段的“验收门槛”不通过就不进入下一阶段。
> 代码仓库：`https://github.com/chengdie/gradient`（LESS 的 fork，已含精确余弦在线筛选实现）。
> 编写日期：2026-10-18。文中“已核实”指已对照仓库源码确认；“待执行方确认”指需要在 GPU 服务器上实测。

## 0. 执行方必读

1. **有限差分（FD）筛选尚未在仓库实现**，需按第 3 节实现；仓库现有的 `less/train/online_proxy_core.py` 是**精确余弦版本**（逐样本反传），在本实验中作为对照组和正确性参照，不得改变它的行为。
2. 不得使用任何测试集答案构造参考集；参考集只来自 dev / few-shot 示例（仓库 `less/data_selection/target_data.py` 已按此实现）。
3. 所有方法共享同一模型、LoRA 配置、候选池、数据顺序种子和评测协议，只改变“筛选规则”。
4. 每次运行的输出目录必须是空目录；所有原始日志（`run.json`、`batches.jsonl`、`metrics.json`）都要保留，最终汇总只能从这些原始日志计算，不得手填数字。
5. 遇到下列情况立即停止并上报，不要自行改动实验设计：参考梯度为零/非有限；FD 与精确余弦符号一致率低于第 5 节门槛；训练 loss 出现 NaN；评测样本数与预期不符（评测脚本会拒绝部分预测，不要绕过）。

## 1. 研究问题与预注册假设

**研究问题**：在线训练中，用“参考集梯度方向上的两次前向损失差”判断候选样本是否与目标任务梯度冲突，能否以显著更低的成本，复现逐样本反传余弦筛选的选样决策与下游效果？

| 编号 | 假设 | 对应实验 | 判定方式 |
|---|---|---|---|
| H1 保真度 | FD 判定与精确余弦符号高度一致，且不一致集中在 \|cos\| 接近 0 的样本 | E2 | 符号一致率、分桶一致率、AUC |
| H2 效果不劣 | FD 在线筛选的下游分数不劣于精确余弦在线筛选 | E3 | 配对差值 95% CI 下界 > −1.0 分（非劣界值，预注册，可由用户调整） |
| H3 有效性 | FD 筛选优于“同保留数随机丢弃”与“不过滤” | E3 | 配对差值及 95% CI |
| H4 成本 | FD 每批筛选成本显著低于精确余弦 | E4 | 墙钟时间、GPU·h、峰值显存、前/反向次数 |

无论结果是否支持假设，都如实报告；不得在看到测试结果后修改非劣界值、阈值或丢弃种子。

## 2. 方法定义

### 2.1 共同部分（与已实现的精确版本一致，已核实）

- 可训练参数 θ：全部 LoRA 参数（r=16，alpha=64，dropout=0，作用于仓库 `QWEN_PROJECTIONS` 指定的投影层）；LoRA 参数以 FP32 存储，基座 bfloat16。
- 计算模式：`model.eval()`（关闭 dropout）但保留 autograd，保证打分与更新面对同一确定性目标。
- 样本损失 ℓ_i：只在监督（回答）token 上计算交叉熵，**先在样本内 token 平均**，样本间等权。
- 参考梯度 g_ref：固定参考集（mmlu / bbh / tydiqa 的 dev 或 few-shot 示例）各样本梯度的算术平均，FP32 累加，不做随机投影。
- 刷新策略：每次有效更新后在参考集上测 teacher-forced 全词表归一化条件熵；`drift = mean_j|h_cur[j]-h_refresh[j]| / max(mean_j|h_refresh[j]|, 1e-3)`，`drift > 0.05` 刷新；无论熵如何，每 20 次有效更新强制刷新（`max_cache_updates=20`）。
- 全批冲突且缓存已过期：在当前参数上刷新一次 g_ref 后重筛同一批；仍全冲突则跳过该批，不调用优化器 step。
- 更新：保留样本梯度取算术平均 → 梯度范数裁剪 1.0 → AdamW（lr=1e-5，weight decay 0）一步；不跨 batch 累积。

### 2.2 精确余弦版本（对照，已实现）

对每个候选样本单独反传得到 g_i，计算 `cos_i = <g_i, g_ref> / (||g_i|| ||g_ref||)`；`cos_i < 0` 排除，`cos_i >= 0` 保留；无监督 token 或零梯度样本排除。保留样本的梯度直接复用做更新，不二次反传。每批成本 ≈ B 次前向 + B 次逐样本反传。

### 2.3 有限差分版本（本文方法，需实现）

记单位方向 `u = g_ref / ||g_ref||`（全 LoRA 参数向量，FP32），步长 ε>0。对候选样本 i 定义

```
Δ_i = ℓ_i(θ − ε·u) − ℓ_i(θ)
```

一阶泰勒展开：`Δ_i = −ε·<g_i, u> + O(ε²) = −ε·||g_i||·cos_i + O(ε²)`，因此 `sign(Δ_i) = −sign(cos_i)`（ε 足够小且数值噪声可忽略时）。

判定规则（与精确版的“cos<0 排除、cos>=0 保留”对齐）：

- `Δ_i > δ`：沿参考下降方向走一小步，该样本损失**上升** → 冲突 → 排除。
- `Δ_i <= δ`：保留。主实验 `δ = 0`；`δ > 0`（噪声死区）只作为消融。
- 无监督 token 的样本排除（与精确版一致）。FD 无法廉价检测“零梯度”样本，此类样本若 Δ_i=0 会被保留，需在日志中单独计数。

辅助诊断量（只记录，不参与判定）：方向导数估计 `d_i = −Δ_i / ε ≈ <g_i, u>`。

### 2.4 两种 FD 实现路径

| 路径 | 每批计算 | 说明 |
|---|---|---|
| A（主实现，必须做） | ① θ 处无梯度批量前向 B 条 → ② 原地扰动到 θ−εu，无梯度批量前向 B 条 → 精确恢复 θ → ③ 只对保留的 K 条做一次带梯度的批量前向 + 反传 | 简单、显存安全；③ 用“保留样本逐样本损失的算术平均”求梯度，数学上等于逐样本梯度的平均，与精确版的更新规则一致 |
| B（可选优化，E4 中测） | ① θ 处带梯度批量前向并保留计算图 → ② 用 `torch.func.functional_call` 传入扰动后的参数字典做无梯度前向（不得原地改参数，否则破坏 ① 的计算图）→ ③ 在 ① 的图上对保留样本损失均值反传 | 省掉一次前向，但要为全部 B 条保留激活，显存更高；必须先证明与 A 的判定和梯度一致 |

成本对比（每个全局 batch，不含两种方法共有的参考集熵前向与刷新）：精确版 = B 次单样本前向 + B 次单样本反传；FD-A = 2 次 B 条批量前向 + 1 次 K 条批量前向反传；FD-B = 1 次带图前向 + 1 次无图前向 + 1 次反传。FD 的收益主要来自“去掉逐样本反传、改为批量前向”。

### 2.5 实现约束（必须满足）

1. **扰动必须精确恢复**：扰动前 `backup = [p.detach().clone() for p in params]`，前向后 `p.data.copy_(backup)`；不得用“再加回 εu”恢复（浮点误差会累积）。恢复后断言逐元素完全相等（开发与冒烟阶段开启，正式运行可每 100 批抽查一次）。
2. **批量前向的逐样本损失必须等于单样本前向损失**：padding 位置 label=-100，attention_mask 正确，按样本内监督 token 求均值；需在单测和冒烟中核对（容差见第 6 节）。
3. u 在所有 rank 上必须相同（直接复用已 all-reduce 的 g_ref）。多卡时候选按位置分片（与现有 `_filter` 相同：`range(rank, len(records), world_size)`），每卡算本地 Δ_i 与保留样本的损失和梯度和，`all_reduce` 梯度和与保留数，按**全局保留数**求平均；没有保留样本的卡也必须参与 all_reduce。
4. 判定只用当前缓存的 g_ref（可能已过期），与精确版语义一致；“全批冲突且缓存过期 → 刷新一次重筛”逻辑原样复用。
5. 前向的 dtype 与精确版相同（基座 bf16、LoRA FP32）。FD 的数值噪声在 E1 中实测，不得预先假设可忽略。

## 3. 需要实现的代码改动

不修改 `OnlineProxyTrainer` 现有行为；新增功能通过子类与新参数接入，默认参数下旧命令的结果必须不变。

| 文件 | 改动 |
|---|---|
| `less/train/online_proxy_fd.py`（新） | `FiniteDifferenceProxyTrainer(OnlineProxyTrainer)`：重写筛选与更新；支持 `scheme=forward|central`（central：`Δ_i = [ℓ(θ−εu) − ℓ(θ+εu)]/2`，需两次扰动前向，此时 θ 处前向只在需要记录 loss 时做）、`deadzone=δ`、`epsilon=ε`、`fd_batch_size`（批量前向的微批大小，防 OOM）。另实现 `filter_mode=none`（全部保留）和 `filter_mode=random`（按给定日志逐批复现保留数量，随机选样本，独立随机种子） |
| `less/train/online_proxy.py` | 新增参数：`--filter {cosine,fd,none,random}`（默认 `cosine`，即现有行为）、`--fd-epsilon`、`--fd-scheme {forward,central}`（默认 forward）、`--fd-deadzone`（默认 0）、`--fd-batch-size`（默认 16）、`--random-match-log <batches.jsonl>`、`--random-seed`、`--save-every-batches N`（每 N 个全局 batch 额外保存 adapter + `online_state.pt` 到 `snapshot-batch-N/`，供 E2 使用）。全部写入 `run.json`，FD 运行的 `protocol_version` 写为 `online-proxy-fd-v1` |
| `batches.jsonl` 字段扩展 | 每条 decision 增加：`delta`、`dir_deriv`（=−Δ/ε）、`loss_base`、`loss_perturbed`；每批增加：`timing`（filter / update / entropy / refresh 各阶段秒数，用 `torch.cuda.synchronize()` 后计时）、`peak_mem_gb`、`n_forward_examples`、`n_backward_examples`、`update_displacement_norm`、`filter_mode` |
| `less/analysis/fd_fidelity.py`（新） | 离线保真度工具：输入 adapter 路径、参考集任务、候选 id 列表、ε 列表、scheme；在同一参数上输出每个候选的精确 `cos`、`dot=<g_i,u>`、`||g_i||`，以及各 ε 下的 `delta`。用于 E1、E2，不训练 |
| `less/analysis/summarize_fd.py`（新） | 读取各运行的 `run.json`、`batches.jsonl`、`train_results.json`、`metrics.json`、`predictions.jsonl`，生成第 8 节规定的表格、统计检验和图 |
| `tests/test_online_proxy_fd.py`（新） | 见第 6 节单测清单 |
| `less/scripts/online_proxy_train.sh` | 透传新增环境变量：`FILTER`、`FD_EPSILON`、`FD_SCHEME`、`FD_DEADZONE`、`FD_BATCH_SIZE`、`RANDOM_MATCH_LOG`、`SAVE_EVERY_BATCHES` |

## 4. 实验设置

### 4.1 固定配置（全部在线方法共享，已核实为仓库默认值）

| 项 | 值 |
|---|---|
| 基座模型 | Qwen3.5-4B（`MODEL_PATH`，脚本默认路径是用户服务器路径，执行方按实际改） |
| LoRA | r=16，alpha=64，dropout=0，目标层 `QWEN_PROJECTIONS` |
| 优化器 | AdamW，lr=1e-5，weight decay=0，grad clip=1.0，无学习率调度 |
| 全局候选 batch | 16（`--batch-size 16`） |
| 长度 | 候选 `--max-length 1024`，参考集 `--reference-max-length 4096` |
| 刷新 | `--entropy-threshold 0.05 --max-cache-updates 20`，开启过期全拒绝重试 |
| epoch | 1（候选池每条恰好看一次） |
| 候选顺序 | `random.Random(seed + epoch).shuffle`，同一 seed 下所有方法看到相同顺序（配对设计的基础） |
| 参考集 | `--task mmlu|bbh|tydiqa`：MMLU 每学科 dev 前 5 条，BBH 每子任务 3 条示例（27 个子任务），TyDiQA 每语言 1 条 one-shot |
| 评测 | `evaluation.eval.qwen`：MMLU accuracy（5-shot）、BBH exact_match（3-shot）、TyDiQA F1（1-shot）；子任务宏平均；`--seed 42 --max_length 8192 --max_new_tokens 512 --dtype bfloat16` |

### 4.2 候选池（待执行方确认）

完整 LESS 训练池需从 Hugging Face `princeton-nlp/less_data` 下载，放到 `data/train/processed/{flan_v2,cot,dolly,oasst1}/<name>_data.jsonl`（在线入口按此路径拼接，已核实）。当前仓库副本中 cot、flan_v2、oasst1 只是占位文件（3 行），dolly 有 15,011 条。仓库 `data/selected_data/` 已有 LESS 原版 top-5% 选择结果（mmlu/bbh/tydiqa × seed 3/6/9，各 13,533 条），按 5% 反推完整池约 27 万条。下载后必须核对行数：四个文件合计应与 13,533/0.05 ≈ 270,660 量级一致，差异超过 1% 时停止上报。

**算力预算决策（执行前由用户确认，否则按默认）**：在线精确余弦版需对每条候选逐样本反传，27 万条单 epoch 成本很高。默认方案：
- 主实验在**固定随机子池 P**上进行：从完整池按 `seed=2026` 均匀抽取 `N_pool = 40,000` 条（约 15%），落盘为 `data/train/pool40k/pool.jsonl` 并记录 sha256；所有在线方法使用同一子池（`--train-files` 指向该文件）。
- 若 E4 测得精确版在子池上的单次运行超过 24 GPU·h，再降到 20,000 条，并在报告中写明；不得各方法用不同子池。
- 离线基线（random / transfer / less）保持仓库原协议（完整池），另外在子池 P 上补一组 random-5% 与 base，确保在线方法有同池对照。

### 4.3 种子

- 在线方法的候选顺序种子 `--seed ∈ {3, 6, 9}`（与 LESS 选择种子对齐，便于叙述）。同一 seed 下所有在线方法看到完全相同的候选序列，构成配对设计。
- `random-matched` 的选样种子：`--random-seed = 1000 + seed`。
- 评测种子固定 `--seed 42`。
- 离线方法沿用 `less.experiments` 的 `--selection-seeds 3 6 9 --training-seeds 0`。

## 5. 实验矩阵

### E0 环境与基线复现（门槛实验）
- `base`：不训练的 Qwen3.5-4B 在三项评测上的分数。
- 用已有精确余弦实现跑 `--max-batches 10` 冒烟，确认日志字段完整。

### E1 ε 与数值精度标定（不训练，决定主实验 ε）
在三个参数点上测：初始 LoRA（θ0）、精确版运行 seed=3 的 `snapshot-batch-500/`、`snapshot-batch-2000/`。每点固定抽取 512 条候选（`seed=7`），用 `less/analysis/fd_fidelity.py` 计算：
- ε 网格：`{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`（单位：全 LoRA 参数向量的 L2 距离）；scheme：forward 与 central。
- 输出：符号一致率 `agree = mean[sign(Δ_i) == −sign(cos_i)]`；`d_i` 与 `<g_i,u>` 的 Spearman 相关；相对误差 `|d_i − <g_i,u>| / |<g_i,u>|` 的中位数；“噪声底”：同一 θ 上重复两次前向的 `|ℓ − ℓ'|` 最大值（应为 0，非 0 说明存在非确定性 kernel，需记录）。
- **选择规则（预注册）**：选使三个参数点平均 agree 最高的 (ε, scheme)；若最高者与次高者相差 < 0.5 个百分点，选更大的 ε（更不受 bf16 舍入影响）。若 forward 与 central 差距 < 1 个百分点，主实验用 forward（成本更低）。结果写入 `out/E1/choice.json`，后续实验只读该文件。
- **门槛**：最优配置在三个参数点上的平均 agree ≥ 90%，且在 |cos| ≥ 0.05 的样本上 ≥ 97%。不达标时先尝试：基座改 FP32 前向（`--dtype float32`）复测；仍不达标则停止上报（这是方法本身的负结果，需要用户决策）。

### E2 训练过程中的选样保真度
用精确版 seed=3 运行，每 `--save-every-batches 500` 保存快照；对每个快照，取紧随其后的 32 个真实训练 batch（512 条，按 `batches.jsonl` 中的 `pool_index`），用 E1 选定的 (ε, scheme) 计算 FD 判定并与精确 cos 对比：
- 总体符号一致率、Cohen's κ、以 −Δ 为分数预测 `cos ≥ 0` 的 ROC-AUC；
- 按 |cos| 分桶 `[0,0.01), [0.01,0.03), [0.03,0.1), [0.1,1]` 的一致率；
- 每批保留集合的 Jaccard 相似度、保留率差异；
- 随训练步数的一致率曲线。

### E3 端到端下游效果（主实验）

| ID | 方法 | 入口 | 说明 |
|---|---|---|---|
| M0 | base | `less.experiments --methods base` | 不训练 |
| M1 | online-none | `online_proxy --filter none` | 不过滤，在线逐批全部更新；与在线方法同池同顺序 |
| M2 | online-random-matched | `--filter random --random-match-log <M4 的 batches.jsonl>` | 每批保留数与 M4 同 seed 运行完全相同，样本随机选；隔离“少用数据”本身的效应 |
| M3 | online-cosine（精确） | `--filter cosine`（现有实现） | 上界参照 |
| M4 | **online-fd（本文）** | `--filter fd --fd-epsilon <E1> --fd-scheme <E1>` | 主方法 |
| M5 | offline-random-5% | `less.experiments --methods random` | 仓库原协议 |
| M6 | offline-transfer-5% | `less.experiments --methods transfer` | 使用仓库已有 LESS 原版选择文件（seed 3/6/9） |
| M7 | offline-LESS-5%（Qwen 复现） | `less.experiments --methods less` | warmup → 梯度库 → top-5% → 重训练；成本最高，预算不足时列为可选，报告中注明 |

每个方法 × 每个目标任务（mmlu / bbh / tydiqa，参考集与评测任务一致）× 3 个 seed。优先级：P0 = mmlu 上 M0–M4；P1 = bbh、tydiqa 上 M0–M4；P2 = M5、M6；P3 = M7 和 E5。预算不足时按优先级截断，并在报告中写明缺失格子，不得用单 seed 结果冒充三 seed。

### E4 成本测量
对 M1、M3、M4（以及 FD 路径 B，若实现）在 mmlu、seed=3 上各统计：
- 每全局 batch 中位数墙钟时间（分 filter / update / entropy / refresh），去掉前 20 批热身；
- 单次完整运行墙钟时间与 GPU·h（= 墙钟 × 卡数）；
- 峰值显存（`torch.cuda.max_memory_allocated`）；
- 每批前向/反传样本数（日志计数）；
- 刷新次数及刷新占比。
全部在同一型号 GPU、同一卡数上测；记录 GPU 型号、驱动、CUDA、PyTorch 版本。

### E5 消融（mmlu，seed 3/6/9；预算不足时只跑 seed 3 并注明）

| 消融 | 设置 |
|---|---|
| ε 敏感性 | E1 最优 ε 的 ×0.3、×3（只改 ε） |
| 差分格式 | forward vs central |
| 死区 δ | `δ ∈ {0, 0.5·σ_noise, 1·σ_noise}`，σ_noise 取 E1 中同 θ 两次前向差的标准差；若噪声底为 0，用 E1 中 |Δ| 的 10% 分位 |
| 刷新策略 | `--max-cache-updates 1`（每步刷新）；熵阈值 0.01 / 0.05 / 0.10；固定间隔（`--entropy-threshold 1e9 --max-cache-updates 20`） |
| 过期重试 | 默认 vs `--no-retry-stale-empty` |

## 6. 测试与正确性门槛

`tests/test_online_proxy_fd.py` 至少包含（用 CPU、FP32、小型随机 Transformer 或现有测试中的玩具模型）：
1. **扰动恢复**：FD 打分后所有 LoRA 参数与打分前逐元素相等。
2. **批量 = 单样本**：不同长度、含 padding 的 4 条样本，批量前向的逐样本损失与逐条前向一致（FP32 下 `atol=1e-6`）。
3. **符号正确性**：FP32、ε=1e-4 时，对随机构造的 g_ref，`sign(Δ_i) == −sign(<g_i,u>)` 对 |cos|>1e-3 的样本全部成立；`d_i` 与 `<g_i,u>` 相对误差 < 1e-2。
4. **更新等价**：给定同一保留集合，FD 路径的更新梯度与精确版“保留梯度算术平均”一致（`atol=1e-6`），优化器 step 后参数一致。
5. **边界**：全批冲突 + 缓存过期 → 刷新一次并重筛同一批；仍全冲突 → 跳过且不调用 `optimizer.step`；无监督 token 样本被排除并记录原因。
6. **默认行为不变**：`--filter cosine`（默认）时现有 `tests/test_online_proxy.py` 全部通过。
7. **random-matched**：每批保留数与匹配日志一致，相同 seed 结果可复现。
8. **分布式**（若有多卡环境）：`torchrun --nproc_per_node=2` 与单卡在同一 batch 上的保留集合和更新后参数一致（bf16 下 `atol=1e-3`）。

GPU 冒烟门槛：Qwen3.5-4B 上 M3 与 M4 各跑 `--max-batches 20`，检查无 NaN、日志字段齐全、显存未溢出、扰动恢复断言通过。

## 7. 端到端执行流程

所有命令在仓库根目录执行。以下变量先设定（路径按服务器实际修改）：

```bash
export REPO=/path/to/gradient                 # git clone https://github.com/chengdie/gradient
export MODEL_PATH=/path/to/qwen3.5-4b
export DATA_DIR=$REPO/data
export OUT=/path/to/out/fd-thesis             # 每个运行一个空子目录
export NGPU=8                                 # 实际卡数
```

### 阶段 1：环境
1. `git clone` 后记录 commit hash；新建分支 `fd-filter`，所有改动在该分支提交。
2. 按 `requirements-qwen3_5.txt`（若不存在则按 README）安装依赖，记录 `python -m pip freeze > $OUT/env/pip-freeze.txt`、`nvidia-smi > $OUT/env/gpu.txt`。
3. `python -m pytest tests/test_online_proxy.py -q` 通过。
4. `PYTHON_BIN=python bash less/scripts/online_proxy_train.sh --dry-run` 能打印配置。

验收：现有测试全部通过；记录文件齐全。

### 阶段 2：数据
1. 从 `princeton-nlp/less_data` 下载训练池，替换 `data/train/processed/` 下占位文件；核对行数（见 4.2）。
2. 校验评测数据：`python -m less.experiments run --preflight --tasks mmlu bbh tydiqa --model-path $MODEL_PATH --data-dir $DATA_DIR`，记录各任务评测条数。
3. 构造子池 P：写一个小脚本，按行读四个源文件，`random.Random(2026).sample` 抽 `N_pool` 条，原样写入 `data/train/pool40k/pool.jsonl`，同时写 `pool_manifest.json`（每行来源文件与行号、sha256）。
4. 参考集与评测集去重检查：确认参考样本（dev / few-shot）不出现在评测记录中（按问题文本精确匹配），结果写入 `$OUT/data/leak_check.json`。

验收：行数一致；leak_check 为 0 重叠；manifest 存在。

### 阶段 3：实现 FD（第 3 节）与测试（第 6 节）
完成后运行 `python -m pytest tests -q`，再做 GPU 冒烟：

```bash
for F in cosine fd; do
CUDA_VISIBLE_DEVICES=0 python -m less.train.online_proxy \
  --model-path $MODEL_PATH --data-dir $DATA_DIR \
  --train-files $DATA_DIR/train/pool40k/pool.jsonl \
  --task mmlu --output-dir $OUT/smoke/$F --filter $F \
  --fd-epsilon 1e-3 --max-batches 20 --seed 3
done
```

验收：第 6 节全部测试通过；冒烟两项无 NaN、字段齐全。

### 阶段 4：E0 + E1
1. base 评测：`python -m less.experiments run --methods base --tasks mmlu bbh tydiqa --model-path $MODEL_PATH --data-dir $DATA_DIR --output-dir $OUT/offline --eval-nproc $NGPU --execute`。
2. 精确版 seed=3（mmlu）带快照运行（同时作为 E3 的 M3 seed3 和 E2 的数据源）：

```bash
torchrun --standalone --nproc_per_node=$NGPU -m less.train.online_proxy \
  --model-path $MODEL_PATH --data-dir $DATA_DIR \
  --train-files $DATA_DIR/train/pool40k/pool.jsonl \
  --task mmlu --filter cosine --seed 3 --save-every-batches 500 \
  --output-dir $OUT/online/mmlu/cosine/s3
```
3. 在 θ0、`snapshot-batch-500`、`snapshot-batch-2000` 上运行 `less/analysis/fd_fidelity.py` 完成 ε 网格，输出 `$OUT/E1/grid.csv` 与 `choice.json`。

验收：E1 门槛（5 节）满足；否则停止上报。

### 阶段 5：E2
用阶段 4 的快照和 `batches.jsonl` 中 decision 的 `id`/`position` 定位后续 32 个 batch，跑保真度分析，输出 `$OUT/E2/fidelity.csv` 和分桶表。

验收：每个快照都有结果；总体一致率与 E1 同量级（相差 > 5 个百分点需在报告中解释）。

### 阶段 6：E3 主实验
模板（`$TASK ∈ mmlu bbh tydiqa`，`$S ∈ 3 6 9`）：

```bash
EPS=$(python -c "import json;print(json.load(open('$OUT/E1/choice.json'))['epsilon'])")
SCH=$(python -c "import json;print(json.load(open('$OUT/E1/choice.json'))['scheme'])")
COMMON="--model-path $MODEL_PATH --data-dir $DATA_DIR --train-files $DATA_DIR/train/pool40k/pool.jsonl --task $TASK --seed $S"
L="torchrun --standalone --nproc_per_node=$NGPU -m less.train.online_proxy"
$L $COMMON --filter none   --output-dir $OUT/online/$TASK/none/s$S
$L $COMMON --filter cosine --output-dir $OUT/online/$TASK/cosine/s$S     # mmlu s3 已在阶段 4 完成
$L $COMMON --filter fd --fd-epsilon $EPS --fd-scheme $SCH --output-dir $OUT/online/$TASK/fd/s$S
$L $COMMON --filter random --random-match-log $OUT/online/$TASK/fd/s$S/batches.jsonl \
   --random-seed $((1000+S)) --output-dir $OUT/online/$TASK/random_matched/s$S    # 必须在 fd 之后
```

每个运行结束后评测其 `model/`：

```bash
torchrun --standalone --nproc_per_node=$NGPU -m evaluation.eval.qwen \
  --model_path $RUN/model --base_model_path $MODEL_PATH --task $TASK \
  --data_dir $DATA_DIR --output_dir $RUN/eval --seed 42 \
  --max_length 8192 --max_new_tokens 512 --batch_size 4 --device cuda:0 --dtype bfloat16
```
（参数与 `less/experiments.py` 的 `evaluate()` 一致；多卡下设备分配以 `evaluation/eval/qwen.py --help` 的实际行为为准。）

离线基线：

```bash
python -m less.experiments run --methods random transfer --tasks mmlu bbh tydiqa \
  --selection-seeds 3 6 9 --training-seeds 0 --model-path $MODEL_PATH --data-dir $DATA_DIR \
  --output-dir $OUT/offline --train-nproc $NGPU --eval-nproc $NGPU --execute
# 预算允许时追加 --methods less
```
子池上的 random-5%：用 `less/data_selection/random_subset.py --train_files $DATA_DIR/train/pool40k/pool.jsonl --percentage 0.05 --seed $S --output_dir ...` 选样后，用 `less.experiments single --only finetune --selected-file ...` 训练、`--only after --adapter-path ...` 评测。

验收：每个格子都有 `train_results.json` 和 `eval/metrics.json`（`completed: true`，`n` 与阶段 2 preflight 一致）；M2 每批保留数与匹配的 M4 完全一致（脚本核对）。

### 阶段 7：E4、E5
E4 直接从 E3 中 mmlu/s3 的 M1、M3、M4 日志的 `timing`、`peak_mem_gb` 汇总；路径 B 若实现则单独跑一次 mmlu/s3。E5 按第 5 节表格，输出目录 `$OUT/ablation/<name>/s$S`。

### 阶段 8：汇总
`python -m less.analysis.summarize_fd --root $OUT --output $OUT/report`，生成第 8 节全部表图，并写 `report/README.md`。

## 8. 指标、统计检验与报告

### 8.1 指标定义
- 下游：MMLU accuracy、BBH exact_match、TyDiQA F1（均为子任务宏平均，×100，取 `metrics.json` 的 `score`）。
- 训练过程：总保留率 = Σaccepted / Σcandidates；有效更新次数；跳过批次数；刷新次数及原因分布；参考集损失曲线（每次刷新时记录）。
- 保真度：符号一致率、κ、AUC、分桶一致率、保留集合 Jaccard（定义见 E1/E2）。
- 成本：每批墙钟、单次运行 GPU·h、峰值显存、相对 M3 的加速比 = T(M3)/T(M4)。

### 8.2 统计检验
- 主比较（预注册）：M4 vs M3（非劣）、M4 vs M2、M4 vs M1。配对单位 = (任务, seed)；对每个任务报告 3 个 seed 的均值 ± 标准差，以及配对差值均值。
- 三任务合并：9 个配对差值（3 任务 × 3 seed），报告 bootstrap 95% CI（10,000 次重采样，按配对单元重采样）。
- 题目级检验：同一任务同一 seed 下，两模型对同一批评测题的逐题得分用 paired bootstrap（按题重采样 10,000 次），MMLU/BBH 另报 McNemar 检验；三个比较做 Holm 校正。
- 只有 3 个 seed 时不报 seed 级 t 检验的 p 值作为主要证据；如实写“3 seed，统计功效有限”。
- 非劣判定：M4 − M3 合并差值的 95% CI 下界 > −1.0 分 → 非劣成立；否则如实报告不成立。

### 8.3 报告必须包含的表图
1. 表 1 主结果：方法 × 任务（均值 ± std），附成本列（GPU·h）。
2. 表 2 配对差值与 95% CI、Holm 校正后 p 值。
3. 表 3 成本：每批时间分解、峰值显存、加速比。
4. 表 4 E1 ε 网格（一致率 / 相关 / 相对误差）。
5. 图 1 E2 分桶一致率；图 2 一致率随训练步变化；图 3 cos 与 d_i 散点（对数轴）。
6. 图 4 各方法每批保留率曲线；图 5 参考集损失随更新步变化。
7. 表 5 消融。
8. 偏差与缺失说明：未跑格子、子池替代完整池、单 seed 消融等。

## 9. 产物清单

```
$OUT/
  env/          pip-freeze.txt, gpu.txt, commit.txt
  data/         pool_manifest.json, leak_check.json, preflight.json
  smoke/        cosine/, fd/
  E1/           grid.csv, choice.json
  E2/           fidelity.csv, buckets.csv
  online/<task>/<none|cosine|fd|random_matched>/s<seed>/
                run.json, batches.jsonl, train_results.json, model/, checkpoint-epoch-1/, eval/metrics.json, eval/predictions.jsonl
  offline/      less.experiments 原生输出
  ablation/<name>/s<seed>/
  report/       tables/*.csv, figures/*.png, README.md
```
代码交付：`fd-filter` 分支的 commit 列表与 diff；新增测试的运行输出。

## 10. 最终验收清单

- [ ] 第 6 节测试全部通过，默认 `--filter cosine` 行为不变
- [ ] E1 门槛通过并有 `choice.json`
- [ ] P0 格子（mmlu 上 M0–M4，3 seed）全部完成且评测 `completed: true`
- [ ] M2 与 M4 每批保留数一致的核对结果
- [ ] 所有报告数字可由 `summarize_fd` 从原始日志重算
- [ ] 报告写明 GPU 型号、卡数、子池大小、未完成格子与原因
- [ ] 不含任何测试集答案参与参考集或筛选的证据（leak_check）

## 11. 风险与处置

| 风险 | 征兆 | 处置 |
|---|---|---|
| bf16 舍入淹没 Δ | E1 一致率低、小 ε 下 Δ 大量为 0 | 增大 ε；central 格式；基座 FP32 前向（成本上升，计入 E4） |
| ε 过大，非线性主导 | 大 ε 一致率下降 | 按 E1 规则选最优，不调测试集 |
| 非确定性 kernel | 同 θ 两次前向不同 | 设 `torch.use_deterministic_algorithms(True)`（能用则用），或引入死区 δ（作为消融报告） |
| 精确版太慢 | 单次 > 24 GPU·h | 子池降到 20k；所有在线方法同步降 |
| 保留率过低导致欠训练 | 保留率 < 20% | 如实报告；M2 已控制数据量效应，不改阈值 |
| 显存溢出 | 批量前向 OOM | 减小 `--fd-batch-size`（只影响速度，不影响结果） |
