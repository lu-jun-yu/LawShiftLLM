# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/Qwen3-0.6B

**评估时间**: 2025-11-16 18:15:00

## 各文件夹评估结果

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| action_explicit_extend | V | 80/102 (78.43%) | 85/102 (83.33%) | +4.90% |
| action_explicit_reduce | NV | 79/102 (77.45%) | 1/102 (0.98%) | -76.47% |
| action_implicit_expand | V | 75/102 (73.53%) | 80/102 (78.43%) | +4.90% |
| action_implicit_reduce | NV | 74/102 (72.55%) | 2/102 (1.96%) | -70.59% |
| action_reallocated_expand | V | 84/102 (82.35%) | 77/102 (75.49%) | -6.86% |
| action_reallocated_reduce | NV | 83/102 (81.37%) | 4/102 (3.92%) | -77.45% |
| objCon_addition | NV | 240/409 (58.68%) | 29/409 (7.09%) | -51.59% |
| objCon_explicit_extend | NV | 78/150 (52.00%) | 7/150 (4.67%) | -47.33% |
| objCon_implicit_expand | NV | 85/150 (56.67%) | 7/150 (4.67%) | -52.00% |
| objCon_implicit_reduce | V | 91/150 (60.67%) | 85/150 (56.67%) | -4.00% |
| objCon_reallocated_expand | NV | 85/150 (56.67%) | 4/150 (2.67%) | -54.00% |
| objCon_reallocated_reduce | V | 86/150 (57.33%) | 81/150 (54.00%) | -3.33% |
| objCon_removal | V | 88/149 (59.06%) | 86/149 (57.72%) | -1.34% |
| object_explicit_extend | V | 251/459 (54.68%) | 189/459 (41.18%) | -13.51% |
| object_explicit_reduce | NV | 234/459 (50.98%) | 30/459 (6.54%) | -44.44% |
| object_implicit_expand | V | 230/459 (50.11%) | 239/459 (52.07%) | +1.96% |
| object_implicit_reduce | NV | 257/459 (55.99%) | 16/459 (3.49%) | -52.51% |
| object_reallocated_expand | V | 238/459 (51.85%) | 187/459 (40.74%) | -11.11% |
| object_reallocated_reduce | NV | 219/459 (47.71%) | 36/459 (7.84%) | -39.87% |
| sbCon_addition | NV | 223/470 (47.45%) | 46/470 (9.79%) | -37.66% |
| sbCon_removal | V | 79/115 (68.70%) | 81/115 (70.43%) | +1.74% |
| subject_explicit_extend | V | 171/261 (65.52%) | 173/261 (66.28%) | +0.77% |
| subject_explicit_reduce | NV | 177/261 (67.82%) | 27/261 (10.34%) | -57.47% |
| subject_implicit_expand | V | 158/261 (60.54%) | 162/261 (62.07%) | +1.53% |
| subject_implicit_reduce | NV | 174/261 (66.67%) | 13/261 (4.98%) | -61.69% |
| subject_reallocated_expand | V | 159/261 (60.92%) | 166/261 (63.60%) | +2.68% |
| subject_reallocated_reduce | NV | 170/261 (65.13%) | 48/261 (18.39%) | -46.74% |
| term_down | TD | 0/215 (0.00%) | 35/215 (16.28%) | +16.28% |
| term_extremity_in | XT | 0/215 (0.00%) | 63/215 (29.30%) | +29.30% |
| term_extremity_out | NX | 63/99 (63.64%) | 0/99 (0.00%) | -63.64% |
| term_up | TU | 0/215 (0.00%) | 0/215 (0.00%) | +0.00% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 4031/7569 (53.26%) | 2059/7569 (27.20%) | -26.05% |
