# 训练脚本使用说明

本目录包含用于法律判决预测任务的 Rejection Sampling + SFT 训练流程。

## 📁 文件结构

```
train/
├── rejection_sampling.py    # 拒绝采样脚本
├── sft_train.py             # QLoRA SFT训练脚本
└── README.md                # 本文件
```

## 🚀 训练流程

### 第一步：拒绝采样 (Rejection Sampling)

对训练集中的每个样本并行生成8条回复，筛选出罪名和刑期均正确的回复作为SFT训练数据。

#### 使用方法

```bash
python train/rejection_sampling.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --training_data LawShift/training_set.json \
    --articles LawShift/[某个目录]/articles_original.json \
    --output train/sampled_data.json \
    --num_samples 8 \
    --temperature 0.8 \
    --top_p 0.95
```

#### 参数说明

- `--model_path`: 基础模型路径（默认使用 Qwen3-0.6B）
- `--training_data`: 训练数据路径，默认为 `LawShift/training_set.json`
- `--articles`: 法条数据路径（需要指定具体的articles_original.json文件）
- `--output`: 采样结果输出路径，默认为 `train/sampled_data.json`
- `--num_samples`: 每个样本采样的回复数量，默认为 8
- `--temperature`: 采样温度，默认为 0.8
- `--top_p`: nucleus sampling 参数，默认为 0.95
- `--device`: 设备类型，默认为 "auto"
- `--max_new_tokens`: 最大生成 token 数，默认为 2048

#### 输出文件

1. **sampled_data.json**: 采样得到的训练数据，每个样本包含：
   - `fact`: 案件事实
   - `relevant_articles`: 相关法条ID列表
   - `charge`: 真实罪名
   - `prison_time`: 真实刑期
   - `response`: 模型生成的正确回复
   - `num_correct_for_this_sample`: 该问题的正确回复总数（用于计算权重）

2. **sampling_stats_*.json**: 采样统计信息，包含：
   - 总样本数、总生成数、总接受数
   - 接受率
   - 无正确回复的样本数
   - 平均每个样本的正确回复数

### 第二步：QLoRA SFT训练

使用拒绝采样得到的数据进行监督微调，采用QLoRA方法降低显存占用。

#### 特殊设计：梯度权重调整

为保证训练的无偏性，对于同一个问题有 n 个正确回复的情况，每个样本的梯度会乘以权重 `1/n`。这样可以确保：
- 每个问题对模型的影响力相同
- 避免某些问题因为有更多正确回复而主导训练过程

#### 使用方法

```bash
python train/sft_train.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --sampled_data train/sampled_data.json \
    --articles_path LawShift/[某个目录]/articles_original.json \
    --output_dir train/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4
```

#### 参数说明

**模型相关**
- `--model_path`: 基础模型路径
- `--output_dir`: 输出目录，默认为 `train/checkpoints`

**数据相关**
- `--sampled_data`: 拒绝采样得到的数据路径，默认为 `train/sampled_data.json`
- `--articles_path`: 法条数据路径
- `--max_length`: 最大序列长度，默认为 4096

**LoRA相关**
- `--lora_r`: LoRA rank，默认为 64
- `--lora_alpha`: LoRA alpha，默认为 16
- `--lora_dropout`: LoRA dropout，默认为 0.05

**训练相关**
- `--num_train_epochs`: 训练轮数，默认为 3
- `--per_device_train_batch_size`: 每个设备的批大小，默认为 2
- `--gradient_accumulation_steps`: 梯度累积步数，默认为 8（实际批大小 = 2 × 8 = 16）
- `--learning_rate`: 学习率，默认为 2e-4
- `--warmup_steps`: 预热步数，默认为 100
- `--logging_steps`: 日志记录步数，默认为 10
- `--save_steps`: 保存步数，默认为 100
- `--weight_decay`: 权重衰减，默认为 0.01

**量化相关**
- `--no_4bit`: 不使用4bit量化（默认使用4bit量化）

#### 输出文件

训练完成后，会在输出目录下生成：
- `final_model/`: 最终训练好的模型（LoRA权重）
- `checkpoint-*/`: 中间检查点
- `logs/`: TensorBoard日志文件

## 💡 完整训练示例

```bash
# 1. 拒绝采样（使用Qwen3-0.6B基础模型）
python train/rejection_sampling.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --training_data LawShift/training_set.json \
    --articles LawShift/term_up/articles_original.json \
    --output train/sampled_data.json \
    --num_samples 8

# 2. QLoRA SFT训练
python train/sft_train.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --sampled_data train/sampled_data.json \
    --articles_path LawShift/term_up/articles_original.json \
    --output_dir train/checkpoints \
    --num_train_epochs 3

# 3. 使用训练好的模型进行评估
python eval/evaluate.py \
    --model_path train/checkpoints/sft_20250115_120000/final_model \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --test_dir LawShift/term_up \
    --output_dir results/finetuned_model
```

## 📊 训练监控

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir train/checkpoints/sft_*/logs
```

## ⚙️ 显存需求

- **拒绝采样**: 需要约 8-12 GB 显存（取决于模型大小）
- **QLoRA训练**: 需要约 12-16 GB 显存（使用4bit量化）

如果显存不足，可以：
1. 减小 `--per_device_train_batch_size`
2. 增大 `--gradient_accumulation_steps`（保持实际批大小不变）
3. 减小 `--max_length`

## 🎯 关键特性

1. **拒绝采样**: 通过并行生成多条回复并筛选正确的，提高训练数据质量
2. **梯度权重调整**: 通过 1/n 权重确保训练无偏性
3. **QLoRA训练**: 使用4bit量化+LoRA，显著降低显存占用
4. **灵活配置**: 支持丰富的训练参数配置

## 📝 注意事项

1. **训练数据**: `training_set.json` 较大，拒绝采样可能需要较长时间
2. **法条数据**: 不同的数据集子目录有不同的法条文件，需要指定正确的路径
3. **模型路径**: 确保模型路径正确，首次运行会自动下载模型
4. **输出格式**: 训练版prompt要求模型输出 "{罪名} | {刑期}" 或 "不违规"
5. **死刑/无期徒刑**: 刑期为死刑或无期徒刑时，应输出 "XT" 而非数字

## 🔧 依赖安装

```bash
pip install torch transformers accelerate peft bitsandbytes datasets tqdm tensorboard
```

## 📞 问题反馈

如有问题，请检查：
1. 数据路径是否正确
2. 模型是否成功加载
3. 显存是否充足
4. Python环境是否正确安装所有依赖
