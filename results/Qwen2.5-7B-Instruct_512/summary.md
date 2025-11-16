# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/Qwen2.5-7B-Instruct

**评估时间**: 2025-11-16 16:39:37

## 各文件夹评估结果

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| action_explicit_extend | V | 98/102 (96.08%) | 97/102 (95.10%) | -0.98% |
| action_explicit_reduce | NV | 100/102 (98.04%) | 0/102 (0.00%) | -98.04% |
| action_implicit_expand | V | 96/102 (94.12%) | 98/102 (96.08%) | +1.96% |
| action_implicit_reduce | NV | 97/102 (95.10%) | 6/102 (5.88%) | -89.22% |
| action_reallocated_expand | V | 99/102 (97.06%) | 97/102 (95.10%) | -1.96% |
| action_reallocated_reduce | NV | 98/102 (96.08%) | 0/102 (0.00%) | -96.08% |
| objCon_addition | NV | 394/409 (96.33%) | 2/409 (0.49%) | -95.84% |
| objCon_explicit_extend | NV | 147/150 (98.00%) | 1/150 (0.67%) | -97.33% |
| objCon_implicit_expand | NV | 143/150 (95.33%) | 0/150 (0.00%) | -95.33% |
| objCon_implicit_reduce | V | 146/150 (97.33%) | 147/150 (98.00%) | +0.67% |
| objCon_reallocated_expand | NV | 147/150 (98.00%) | 3/150 (2.00%) | -96.00% |
| objCon_reallocated_reduce | V | 146/150 (97.33%) | 140/150 (93.33%) | -4.00% |
| objCon_removal | V | 144/149 (96.64%) | 149/149 (100.00%) | +3.36% |
| object_explicit_extend | V | 448/459 (97.60%) | 433/459 (94.34%) | -3.27% |
| object_explicit_reduce | NV | 453/459 (98.69%) | 2/459 (0.44%) | -98.26% |
| object_implicit_expand | V | 448/459 (97.60%) | 445/459 (96.95%) | -0.65% |
| object_implicit_reduce | NV | 445/459 (96.95%) | 2/459 (0.44%) | -96.51% |
| object_reallocated_expand | V | 447/459 (97.39%) | 448/459 (97.60%) | +0.22% |
| object_reallocated_reduce | NV | 449/459 (97.82%) | 4/459 (0.87%) | -96.95% |
| sbCon_addition | NV | 427/470 (90.85%) | 3/470 (0.64%) | -90.21% |
| sbCon_removal | V | 106/115 (92.17%) | 107/115 (93.04%) | +0.87% |
| subject_explicit_extend | V | 250/261 (95.79%) | 240/261 (91.95%) | -3.83% |
| subject_explicit_reduce | NV | 246/261 (94.25%) | 3/261 (1.15%) | -93.10% |
| subject_implicit_expand | V | 246/261 (94.25%) | 247/261 (94.64%) | +0.38% |
| subject_implicit_reduce | NV | 249/261 (95.40%) | 2/261 (0.77%) | -94.64% |
| subject_reallocated_expand | V | 246/261 (94.25%) | 243/261 (93.10%) | -1.15% |
| subject_reallocated_reduce | NV | 240/261 (91.95%) | 3/261 (1.15%) | -90.80% |
| term_down | TD | 20/215 (9.30%) | 46/215 (21.40%) | +12.09% |
| term_extremity_in | XT | 13/215 (6.05%) | 62/215 (28.84%) | +22.79% |
| term_extremity_out | NX | 85/99 (85.86%) | 47/99 (47.47%) | -38.38% |
| term_up | TU | 17/215 (7.91%) | 2/215 (0.93%) | -6.98% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 6690/7569 (88.39%) | 3079/7569 (40.68%) | -47.71% |
