# bi-deriction-gradient-dc

沿目标任务参考梯度方向做有限差分，用逐样本损失变化筛掉梯度冲突数据。
基于已有 LESS/Qwen 工具实现在线训练；**不运行 LESS 的 warmup、梯度库或离线选数实验**。

- `cosine`：逐样本反传，作为精确判定和更新对照，复用已计算梯度。
- `fd`：单边/中心差分前向打分，仅对保留样本求更新梯度。
- `none`：不筛选，不计算不需要的参考梯度/熵开销。
- `random`：按指定 FD 日志逐批复现保留数量，随机选择样本。

当前运行目标为单张 A800；同一训练入口也支持 `torchrun` 多卡，候选 batch 始终按全局定义。
完整运行与产物说明见 [docs/fd_experiments.md](docs/fd_experiments.md)。

## 安装和测试

服务器使用 Python 3.10+，先安装与 CUDA 匹配的 PyTorch，再安装项目依赖。
在线 FD 不依赖 `traker`；该包仅用于旧 LESS 离线梯度特征流程。

```bash
python -m pip install -r requirements-qwen3_5.txt -r requirements-fd-analysis.txt
python -m pytest tests -q
```

## 单卡冒烟

仓库包含代码、测试与评测资源；训练候选池及历史选样数据不随 Git 提交。
请从 [LESS 数据仓库](https://huggingface.co/datasets/princeton-nlp/less_data) 获取完整训练数据，
或者由它构造固定子池。数据目录与校验说明见 [实验文档](docs/fd_experiments.md)。
训练入口会拒绝尚未下载正文的 Git LFS 指针文件。

```bash
export MODEL_PATH=/path/to/qwen3.5-4b
export TRAIN_FILE=/path/to/pool.jsonl
export CUDA_VISIBLE_DEVICES=0

# 打印命令，不加载模型、不运行 GPU。
bash less/scripts/online_proxy_train.sh --dry-run

# epsilon=1e-3 仅作为数值冒烟起点，不是标定后的正式主实验配置。
FILTER=cosine OUTPUT_DIR=out/smoke/cosine bash less/scripts/online_proxy_train.sh --max-batches 20
FILTER=fd OUTPUT_DIR=out/smoke/fd bash less/scripts/online_proxy_train.sh --max-batches 20
```

扩展到多卡时只改可见设备和进程数；`GLOBAL_BATCH_SIZE` 保持不变。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 GLOBAL_BATCH_SIZE=16 FD_BATCH_SIZE=2 \
  FILTER=fd OUTPUT_DIR=out/smoke/fd-2gpu \
  bash less/scripts/online_proxy_train.sh --max-batches 20
```

## 汇总结果

```bash
python -m less.analysis.summarize_fd --root out --output out/report
```

输出 CSV 表、JSON 汇总和保留率/参考损失图。未完成、未评测及冒烟运行会明确标记，
不会填入正式效果表，也不会生成未经测量的 LESS 对比分数。
