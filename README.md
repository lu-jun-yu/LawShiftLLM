# LawShift 法律判决预测评估工具

本项目提供了针对 LawShift 数据集的法律判决预测（Legal Judgment Prediction, LJP）评估工具。

## 项目结构

```
.
├── LawShift/                    # 数据集根目录
│   ├── action_explicit_extend/  # 各个测试场景文件夹
│   │   ├── articles_original.json
│   │   ├── articles_poisoned.json
│   │   ├── original.json
│   │   └── poisoned.json
│   └── ...                      # 其他场景文件夹
├── eval/                        # 评估结果输出目录
├── evaluate.py                  # 主评估脚本
├── prompt_template.py           # 提示词模板
└── README.md                    # 本文档
```

## 功能特性

1. **自动化评估**：遍历所有测试场景，对 original 和 poisoned 数据分别进行评估
2. **双任务评测**：同时评估罪名（charge）预测和刑期（prison_time）预测的准确率
3. **对比分析**：自动对比 original 和 poisoned 数据的性能差异
4. **结果保存**：生成详细的 JSON 结果文件和可读的文本报告

## 安装依赖

```bash
pip install torch transformers tqdm
```

## 使用方法

### 基本使用

使用默认参数运行评估（默认模型路径为 `../Qwen3-0.6B`）：

```bash
python evaluate.py
```

### 自定义参数

指定模型路径、数据集路径和输出目录：

```bash
python evaluate.py \
    --model_path /path/to/your/model \
    --dataset_root ./LawShift \
    --output_dir ./eval \
    --device cuda
```

### 参数说明

- `--model_path`: 模型路径（默认：`../Qwen3-0.6B`）
- `--dataset_root`: LawShift 数据集根目录（默认：`./LawShift`）
- `--output_dir`: 评估结果输出目录（默认：`./eval`）
- `--device`: 设备类型，可选 `auto`/`cuda`/`cpu`（默认：`auto`）

## 输出说明

评估完成后，会在输出目录生成两类文件：

### 1. 详细结果（JSON格式）

文件名：`detailed_results_{model_name}_{timestamp}.json`

包含每个测试样本的详细预测结果，便于后续分析。

### 2. 汇总报告（文本格式）

文件名：`summary_{model_name}_{timestamp}.txt`

包含：
- 各场景的评估结果
- 罪名准确率和刑期准确率
- Original vs Poisoned 的对比分析
- 总体统计信息

示例输出：
```
================================================================================
文件夹: action_explicit_extend
================================================================================

【Original数据】
  总数: 50
  罪名准确率: 85.00% (42/50)
  刑期准确率: 78.00% (39/50)

【Poisoned数据】
  总数: 50
  罪名准确率: 72.00% (36/50)
  刑期准确率: 65.00% (32/50)

【对比分析】
  罪名准确率变化: -13.00%
  刑期准确率变化: -13.00%
```

## 提示词模板

提示词模板定义在 `prompt_template.py` 中：

### 系统提示词

定义了模型的角色和输出格式要求：
- 模型扮演专业的法律判决助手
- 要求输出格式：`<think>推理过程</think><answer>罪名|刑期</answer>`

### 用户提示词

包含：
- 案件事实（fact）
- 相关法条（relevant_articles）
- 任务要求

可根据需要修改 `prompt_template.py` 中的提示词来优化模型表现。

## 评估流程

1. **加载模型**：使用 transformers 库加载指定的语言模型
2. **遍历数据集**：依次处理每个测试场景文件夹
3. **生成预测**：
   - 读取案件事实和相关法条
   - 构建提示词
   - 调用模型生成预测
   - 解析模型输出的罪名和刑期
4. **计算指标**：统计罪名准确率和刑期准确率
5. **保存结果**：生成 JSON 和文本格式的评估报告

## 数据格式

### 测试数据格式（original.json / poisoned.json）

```json
[
  {
    "fact": "案件事实描述...",
    "relevant_articles": ["354-0-0"],
    "charge": "容留他人吸毒",
    "prison_time": 7
  }
]
```

### 法条数据格式（articles_original.json / articles_poisoned.json）

```json
{
  "354-0-0": "【容留他人吸毒罪】容留他人吸食、注射毒品的，处三年以下有期徒刑..."
}
```

## 注意事项

1. **模型兼容性**：确保使用的模型支持 transformers 库加载
2. **GPU 内存**：大模型可能需要较大的 GPU 内存，建议使用至少 16GB 显存的 GPU
3. **评估时间**：完整评估所有场景可能需要较长时间，建议使用 GPU 加速
4. **提示词优化**：不同模型可能需要调整提示词格式以获得最佳性能

## 故障排除

### 内存不足

如果遇到 CUDA out of memory 错误，可以：
- 使用更小的模型
- 设置 `--device cpu` 使用 CPU
- 减少 `max_new_tokens` 参数

### 解析错误

如果模型输出格式不符合预期，检查：
- `prompt_template.py` 中的提示词是否清晰
- 模型是否支持指令跟随
- 考虑在 `parse_prediction` 函数中添加更多解析模式

## 许可证

本项目遵循相关开源协议。

## 联系方式

如有问题或建议，请联系项目维护者。