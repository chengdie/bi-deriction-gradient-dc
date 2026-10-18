# 本地验证记录：2026-10-18

- 分支：`codex/fd-filter`。
- 环境：macOS arm64、Python 3.9.6、PyTorch 2.8.0 CPU、pytest 8.4.2。服务器运行依赖仍要求 Python 3.10+ 及 Qwen 对应依赖。
- 完整套件：**29 passed**。原始输出见 `validation/pytest.txt`。
- Ruff F 类检查、Python 语法检查、Shell 语法检查、`git diff --check` 通过。
- Matplotlib/pyparsing 有 14 条第三方弃用警告，不影响测试结果。

## 已覆盖

单边/中心差分符号与方向导数、扰动与异常恢复、padding 与逐样本均值、不同答案长度的等权更新、精确版/FD 更新等价、全批冲突刷新重筛与跳步、随机匹配、初始化失败记录、adapter 路径解析、checkpoint/Adam/缓存恢复、逐题评测/重复题双分数、JSONL/CSV/PNG 汇总。

四种模式的两进程 CPU/Gloo 与单进程结果一致，包含局部没有参考或保留样本的 rank；另验证 rank 局部错误会传播到所有进程。

## 本地数据预检

报告保存在 `out/preflight/data_audit.json`。

- Flan、CoT、OpenAssistant 主文件为 Git LFS 指针，尚未取得正文；Dolly 可读取。
- LESS 兼容评测：MMLU 14,042 题/57 学科，BBH 1,080 题/27 任务，TyDiQA 1,713 题/9 语言。
- MMLU dev/test 有 5 条问题文本重复：原口径主分数与排除重复题的 clean_score 同时保存。
- 本地没有找到可复用的已完成训练/评测产物。

## 尚未验证

未运行 A800/CUDA、Qwen3.5-4B 实际加载/训练、GPU bf16 差分数值标定、真实多 GPU NCCL 或下游模型评测。CPU 的模拟模型集成测试不代替这些门槛。没有启动 LESS 实验，没有产生或填造实际模型效果分数。
