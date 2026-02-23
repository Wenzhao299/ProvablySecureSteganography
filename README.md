# ProvablySecureSteganography

**语言 / Language**: [中文](README.md) | [English](README_en.md)

面向大语言模型文本隐写的实验代码仓库，支持多种隐写方法的嵌入与提取，并提供并行批处理、单条测试、PPL 与容量/速度评估脚本。

## 主要功能

- 多卡并行批量隐写与提取：`stego_parallel.py`
- 单条隐写与提取：`stego_single.py`
- 困惑度评估：`evaluate_ppl.py`
- 容量与速度评估：`evaluate_capacity_speed.py`
- 支持方法：`ac`、`meteor`、`adg`、`discop`、`imec`、`sparsamp`

## 集成方法与论文

- `ac`：[Neural Linguistic Steganography](https://aclanthology.org/D19-1115/)（2019，ACL）
- `meteor`：[METEOR: Neural Linguistic Steganography with Self-Adjusting Arithmetic Coding](https://dl.acm.org/doi/10.1145/3460120.3484550)（2021，ACM CCS）
- `adg`：[Provably Secure Generative Linguistic Steganography](https://aclanthology.org/2021.findings-acl.268/)（2021，ACL Findings）
- `discop`：[Discop: Provably Secure Steganography in Practice based on "Distribution Copies"](https://ieeexplore.ieee.org/document/10179287)（2023，IEEE S&P）
- `imec`：[Near-Optimal Practical and Secure Text Steganography](https://iclr.cc/virtual/2023/poster/11490)（2023，ICLR）
- `sparsamp`：[Provably Secure Text Steganography via Sparsification and Sampling](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-yaofei)（2025，USENIX Security）

## 项目结构

```text
.
├── config.py                     # 项目配置：模型路径、默认参数、输出路径工具
├── utils.py                      # 公共工具：模型加载、方法分发、采样/数学工具
├── stego_parallel.py             # 多卡并行批处理入口
├── stego_single.py               # 单条处理入口
├── evaluate_ppl.py               # PPL 评估
├── evaluate_capacity_speed.py    # 容量/速度评估
├── pss/                          # 各隐写算法实现
└── context_movie.csv             # 示例输入数据
```

## 环境准备

核心依赖来自脚本导入：

- `torch`
- `transformers`
- `numpy`
- `scipy`
- `tqdm`
- `bitarray`

## 模型与配置

模型路径和默认参数在 `config.py` 中集中配置：

- `MODEL_PATH_MAP`
- `SUPPORTED_MODEL_NAMES`
- `DEFAULT_*`（如 `DEFAULT_MAX_TOKENS`、`DEFAULT_TEMP` 等）

如需改本地模型目录，修改 `config.py` 中 `MODEL_PATH_MAP`。

## 输入数据格式

并行脚本默认读取 CSV，要求至少两列：

- 第 1 列：`id`
- 第 2 列：`context`

示例见 `context_movie.csv`。

## 用法

### 1) 多卡并行批量隐写与提取

```bash
conda run -n env_ac python stego_parallel.py \
  --method meteor \
  --model_name qwen2.5 \
  --input_csv context_movie.csv \
  --max_contexts 1000 \
  --max_tokens 200
```

说明：

- `message_bits` 默认随机生成（长度默认 `max_tokens * 16`）。
- 并行模式要求可用 CUDA（脚本会按 GPU 数启动多进程）。
- 支持断点续跑：会自动读取已有输出与临时分片中的 `id`，跳过已处理样本。

输出：

- 隐写结果：`results_parallel/{dataset}/{method}.jsonl`
  - 每行一条 JSON：`{\"id\": \"...\", \"text\": \"...\"}`
- 临时分片目录：`results_parallel/{dataset}/.tmp_{method}`（合并后会清理）

### 2) 单条隐写与提取

```bash
conda run -n env_ac python stego_single.py \
  --method meteor \
  --model_name qwen2.5 \
  --input_text "Tell a short story about a rabbit." \
  --message_bits 010101110011 \
  --max_tokens 200
```

输出：

- 隐写文本记录：`results_single/{method}.jsonl`（追加模式）
- 每行一条 JSON，字段包含：`model_name`、`input_text`、`message_bits`、`text`
- 终端会打印嵌入/提取 bit 串及匹配情况。

### 3) PPL 评估

```bash
conda run -n env_ac python evaluate_ppl.py \
  --model qwen2.5 \
  --dataset movie \
  --method meteor
```

默认读取：`results_parallel/{dataset}/{method}.jsonl`

输出目录：`results_parallel/{dataset}/ppl/`

- 逐条 PPL：`results_parallel/{dataset}/ppl/{method}.jsonl`
  - 每行一条 JSON：`{"id": "...", "text": "...", "ppl": 12.34}`
- 统计信息：`results_parallel/{dataset}/ppl/stats_{method}.jsonl`
  - 单行 JSON：`{"average_ppl": ..., "max_ppl": ..., ...}`

### 4) 容量与速度评估

```bash
conda run -n env_ac python evaluate_capacity_speed.py \
  --model_name qwen2.5 \
  --input_csv context_movie.csv \
  --dataset movie \
  --max_contexts 1000 \
  --max_tokens 200
```

输出：`results_parallel/{dataset}/capacity_speed_{model_name}.jsonl`

- 每行一条 JSON，字段包含：`model_name`、`dataset`、`input_csv`、`max_contexts`、`max_tokens`、`method`、`avg_capacity`、`avg_speed`、`processed_count`

## 常见参数

- `--method`：隐写方法（必选）
- `--model_name`：`gpt2` / `qwen2.5` / `llama3.2`
- `--model_path`：自定义模型路径（可覆盖 `config.py`）
- `--max_tokens`：生成 token 上限
- `--top_p` / `--top_k` / `--temp`：采样相关参数
- `--seed`：随机种子
- `--ac_precision`、`--meteor_reorder`、`--imec_block_size`、`--sparsamp_block_size`：方法特定参数

## 推荐许可证

使用 **MIT License**：

- 允许商用、修改、分发、私有使用
- 条款简洁，便于学术与工程复用

仓库已附带 `LICENSE` 文件。
