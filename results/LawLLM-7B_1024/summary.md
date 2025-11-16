# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/LawLLM-7B

**评估时间**: 2025-11-16 22:41:40

## 各文件夹评估结果

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| action_explicit_extend | V | 98/102 (96.08%) | 99/102 (97.06%) | +0.98% |
| action_explicit_reduce | NV | 91/102 (89.22%) | 0/102 (0.00%) | -89.22% |
| action_implicit_expand | V | 100/102 (98.04%) | 95/102 (93.14%) | -4.90% |
| action_implicit_reduce | NV | 90/102 (88.24%) | 0/102 (0.00%) | -88.24% |
| action_reallocated_expand | V | 98/102 (96.08%) | 98/102 (96.08%) | +0.00% |
| action_reallocated_reduce | NV | 96/102 (94.12%) | 0/102 (0.00%) | -94.12% |
| objCon_addition | NV | 255/409 (62.35%) | 1/409 (0.24%) | -62.10% |
| objCon_explicit_extend | NV | 77/150 (51.33%) | 0/150 (0.00%) | -51.33% |
| objCon_implicit_expand | NV | 75/150 (50.00%) | 1/150 (0.67%) | -49.33% |
| objCon_implicit_reduce | V | 84/150 (56.00%) | 87/150 (58.00%) | +2.00% |
| objCon_reallocated_expand | NV | 80/150 (53.33%) | 0/150 (0.00%) | -53.33% |
| objCon_reallocated_reduce | V | 90/150 (60.00%) | 81/150 (54.00%) | -6.00% |
| objCon_removal | V | 74/149 (49.66%) | 96/149 (64.43%) | +14.77% |
| object_explicit_extend | V | 308/459 (67.10%) | 300/459 (65.36%) | -1.74% |
| object_explicit_reduce | NV | 318/459 (69.28%) | 3/459 (0.65%) | -68.63% |
| object_implicit_expand | V | 318/459 (69.28%) | 347/459 (75.60%) | +6.32% |
| object_implicit_reduce | NV | 317/459 (69.06%) | 7/459 (1.53%) | -67.54% |
| object_reallocated_expand | V | 331/459 (72.11%) | 315/459 (68.63%) | -3.49% |
| object_reallocated_reduce | NV | 303/459 (66.01%) | 23/459 (5.01%) | -61.00% |
| sbCon_addition | NV | 283/470 (60.21%) | 0/470 (0.00%) | -60.21% |
| sbCon_removal | V | 55/115 (47.83%) | 56/115 (48.70%) | +0.87% |
| subject_explicit_extend | V | 168/261 (64.37%) | 143/261 (54.79%) | -9.58% |
| subject_explicit_reduce | NV | 155/261 (59.39%) | 0/261 (0.00%) | -59.39% |
| subject_implicit_expand | V | 170/261 (65.13%) | 169/261 (64.75%) | -0.38% |
| subject_implicit_reduce | NV | 160/261 (61.30%) | 0/261 (0.00%) | -61.30% |
| subject_reallocated_expand | V | 142/261 (54.41%) | 167/261 (63.98%) | +9.58% |
| subject_reallocated_reduce | NV | 158/261 (60.54%) | 1/261 (0.38%) | -60.15% |
| term_down | TD | 13/215 (6.05%) | 49/215 (22.79%) | +16.74% |
| term_extremity_in | XT | 11/215 (5.12%) | 73/215 (33.95%) | +28.84% |
| term_extremity_out | NX | 63/99 (63.64%) | 22/99 (22.22%) | -41.41% |
| term_up | TU | 12/215 (5.58%) | 0/215 (0.00%) | -5.58% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 4593/7569 (60.68%) | 2233/7569 (29.50%) | -31.18% |
