# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/Qwen2.5-7B-Instruct

**评估时间**: 2025-11-16 19:15:53

## 各文件夹评估结果

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| action_explicit_extend | V | 100/102 (98.04%) | 94/102 (92.16%) | -5.88% |
| action_explicit_reduce | NV | 97/102 (95.10%) | 0/102 (0.00%) | -95.10% |
| action_implicit_expand | V | 99/102 (97.06%) | 100/102 (98.04%) | +0.98% |
| action_implicit_reduce | NV | 100/102 (98.04%) | 12/102 (11.76%) | -86.27% |
| action_reallocated_expand | V | 98/102 (96.08%) | 97/102 (95.10%) | -0.98% |
| action_reallocated_reduce | NV | 98/102 (96.08%) | 0/102 (0.00%) | -96.08% |
| objCon_addition | NV | 387/409 (94.62%) | 2/409 (0.49%) | -94.13% |
| objCon_explicit_extend | NV | 147/150 (98.00%) | 1/150 (0.67%) | -97.33% |
| objCon_implicit_expand | NV | 146/150 (97.33%) | 3/150 (2.00%) | -95.33% |
| objCon_implicit_reduce | V | 148/150 (98.67%) | 146/150 (97.33%) | -1.33% |
| objCon_reallocated_expand | NV | 147/150 (98.00%) | 3/150 (2.00%) | -96.00% |
| objCon_reallocated_reduce | V | 149/150 (99.33%) | 145/150 (96.67%) | -2.67% |
| objCon_removal | V | 146/149 (97.99%) | 147/149 (98.66%) | +0.67% |
| object_explicit_extend | V | 453/459 (98.69%) | 439/459 (95.64%) | -3.05% |
| object_explicit_reduce | NV | 452/459 (98.47%) | 3/459 (0.65%) | -97.82% |
| object_implicit_expand | V | 448/459 (97.60%) | 453/459 (98.69%) | +1.09% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 3215/3297 (97.51%) | 1645/3297 (49.89%) | -47.62% |
