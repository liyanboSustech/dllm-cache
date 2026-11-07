# dllm-cache

K4_NEXT 机制实验的可迁移运行包。当前仅执行一个问题：**在相同 KV 预算下，最终输出分布的 JS 纠偏收益，能否解释 K4 比随机四层有效？**

此仓库包含冻结代码、32 道旧开发题、实验协议、历史评分标签和精简 V100 基线。**本轮 GPU 实验尚未完成；历史结果不是 g8 的结果。**

## 在 g8 上启动

```bash
git pull --ff-only
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python run_k4.py self-test
```

需要 NVIDIA GPU 和兼容 CUDA 12.4 的驱动；使用四张至少 24 GB、建议 32 GB 以上的卡。每个模型占两张卡，各卡处理不同题目，并非把一个模型张量并行切开。若已有完全一致的 Python 环境，可以直接使用；`check` 会核对关键版本。

模型权重不在 Git 中，也不会由启动器自动下载。默认路径为：

- `/dataset/money-shredder-group/tv/LLaDA-8B-Instruct`
- `/dataset/money-shredder-group/tv/Dream-v0-Instruct-7B`

若 g8 的模型在其他位置，设置：

```bash
export LLADA_MODEL=/实际路径/LLaDA-8B-Instruct
export DREAM_MODEL=/实际路径/Dream-v0-Instruct-7B
```

检查环境、模型配置、权重分片大小、全部提示词 token 和 GPU 占用；此命令不加载模型到 GPU：

```bash
python run_k4.py check --gpus 0,1,2,3
```

正式运行（目录必须尚不存在，避免覆盖旧结果）：

```bash
mkdir -p logs
nohup python -u run_k4.py run --gpus 0,1,2,3 --run-dir runs/k4_g8_01 --wait > logs/k4_g8_01.log 2>&1 &
tail -f logs/k4_g8_01.log
```

也可用 `--gpus 4,5,6,7` 选择另外四张卡。列表前两张运行 LLaDA，后两张运行 Dream。`--wait` 等所选四卡都空闲；省略它则遇到占用直接报错。启动器检查进程、显存和利用率，不会停止其他作业。请在 g8 的持久终端或作业系统中启动；本仓库不假设聊天工具中的后台进程会跨会话存活。

`--llada-model`、`--dream-model` 命令行参数也可代替环境变量。模型元数据不匹配时先核对模型版本，不要直接删除检查。模型来源指纹见 `references/model_reference.json`。当前校验分片名字和大小，没有重新计算所有权重文件的 SHA256。

## 固定的实验

- 同一批 32 道旧 GSM8K 开发题，两模型；不读取新保留集，不筛选成败题。
- LLaDA K4 为零起始层 `[4,5,6,21]`；Dream 沿用旧映射 `[3,4,5,18]`，不预先假设其有效。
- 三组随机四层已冻结，位于协议中，不重新抽取。
- 沿原生 S25 轨迹，在前七个块第一次稀疏读取之前取状态。各条件共享输入、原始 step1 缓存年龄和 RoPE。
- 比较 S25、各单层 NEXT、K4、K4 去掉各一层、三组随机四层。完整同年龄 KV 仅作为诊断参考，**不是无缓存 base model，也不是真实答案**。
- 每层保留 `floor(0.25 × 外部 KV 数)`。严格沿用原 QK、top-k、顺序及 FP16 inf 语义；JS 概率计算为 FP32。记录实际下一块 MASK 覆盖，尤其保留 Dream 数值情况。
- 评分为 `JS(P_full,P_S25) - JS(P_full,P_candidate)`。使用完整词表、当前位置均值、七块等权，不加 EOS 或阶段权重。七块包括已有 STOP 后的状态，主分析不事后切换为有利子集。
- 64 个题目×模型，共 448 个检查点、17,920 次诊断前向（另有 448 次原生空操作核对；热身单列）。
- 所有四卡评分先冻结，再运行 K4 和三组随机四层的完整生成。加上已捕获 S25，共 **320 个正式回答**。
- 主门槛为每模型：K4 正纠偏、K4 相比随机均值更大、去掉 K4 单层的平均贡献为正；六项比较统一 Holm 校正。
- 还要检验分数与最终正确率的联系，不能用分布接近或中断减少替代准确率。具体 AUC、有效题数和停止规则以冻结协议为准。
- **未通过或证据不足就停，不改指标，不自动启动选层或新样本实验。**

[完整冻结协议](data/k4_correction_20261107/protocol.json)

## 运行输出与评审

每次运行放在单独的 `runs/<名称>/` 中，默认不进 Git：

- `status.json`：等待、运行、失败、待评审或已分析状态。
- `execution.json`：模型路径、实际 GPU 映射及跨机器执行说明。
- `source_manifest.json`：本次代码、协议、样本、历史标签的 SHA256。
- `reports/*gpu*.log`：四个工作进程日志。
- `results/<模型>/gpu*/<题号>/probes.jsonl`：逐块、逐条件纠偏评分及实际预算。
- 同目录 `scores_frozen.json`、`outputs.jsonl`：先冻结评分，再保存完整回答。
- `reports/audit.json`：独立重建检查点、预算、提交序列和来源哈希的审计。

每张卡先核对同机官方 S25 解码、原生空操作和 K4 重放；同模型的两张卡热身 token 也必须一致。不同 GPU 架构可能与旧 V100 的 FP16 结果不同，历史 token 一致率只报告，不作为跨机器硬门槛；所有新比较都在 g8 同一批题和硬件上重算，不能与旧准确率直接拼接。

生成和审计成功后，启动器准备匿名评审队列，**不会把严格 Final-only 分数自动当成最终结论**：

```bash
python run_k4.py review --run-dir runs/k4_g8_01
```

待评审文件：`runs/k4_g8_01/reports/review_pending.json`。其中只包含题目、目标答案、回答和匿名 ID，没有方法或模型名。相同题目和回答的历史标签会沿用；明确的 `Final answer` 自动数字评分。剩余答案需按冻结口径评审，区分完整正确、完整错误、未完成。

新标签保存成 JSON 对象（请填入真实 review_id 和人工判定，示例不是标签）：

```json
{
  "实际的review_id": {
    "complete": true,
    "correct": false,
    "reason": "已给出目标数量，但数值计算错误。"
  }
}
```

导入并完成分析：

```bash
python run_k4.py label --run-dir runs/k4_g8_01 --labels /路径/labels.json
python run_k4.py audit --run-dir runs/k4_g8_01
python run_k4.py analyze --run-dir runs/k4_g8_01
```

分析会拒绝缺失标签、未完成的 320 个回答、失败审计或被改动的源文件。最终 `reports/final/summary.json` 包含两个模型的支持/停止判定；结果不会自动触发下一轮。

如果失败，保留原运行目录和日志。环境修复后使用新目录重跑，不能覆盖失败证据或通过换指标解决失败。需要换依赖或科学设置时，先记录修改，不能沿用原冻结结论。

## 文件与验证范围

- `scripts/`、`vendor/`：按原文件哈希复制的研究核心和历史依赖。**旧启动脚本依赖 g4，不要直接调用；统一使用 `run_k4.py`。**
- `portable/`：新机器的路径、显卡映射、运行管理和审计适配；不更改核心评分或条件。
- `data/`：本轮固定题目、提示词、协议，以及导入链需要的旧热身/评估器。
- `references/`：精简历史 token、模型元数据与导出清单。
- `reports/`：只读的历史结论、标签、冻结哈希；不是当前运行进度。

本次打包会检查冻结文件完整性、CPU 数值/原生选择测试、导入与端到端分析链路。**尚未在 g8 上完成 GPU 验证或得到本轮实验结论。**运行时 GPU 预检是正式生成前的必要门槛。

开发测试：

```bash
python -m unittest discover -s tests -v
python run_k4.py self-test
```

## 来源

模型实现来自 [OpenMOSS/Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM)，具体提交和各文件来源保存在两个 `vendor/*/UPSTREAM.json` 中；原文件头部署名和许可声明保持不变。
模型权重分别使用 LLaDA-8B-Instruct 与 Dream-v0-Instruct-7B，由使用者在本地准备并遵守其模型许可。
开发题来自 [GSM8K](https://github.com/openai/grade-school-math) 训练集；辅助 HotpotQA 评估器仅为历史导入依赖，本轮不跑 HotpotQA。
