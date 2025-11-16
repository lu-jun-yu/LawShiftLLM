# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/Qwen2.5-7B-Instruct

**评估时间**: 2025-11-16 20:37:22

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
| object_implicit_reduce | NV | 453/459 (98.69%) | 3/459 (0.65%) | -98.04% |
| object_reallocated_expand | V | 454/459 (98.91%) | 452/459 (98.47%) | -0.44% |
| object_reallocated_reduce | NV | 451/459 (98.26%) | 6/459 (1.31%) | -96.95% |
| sbCon_addition | NV | 441/470 (93.83%) | 5/470 (1.06%) | -92.77% |
| sbCon_removal | V | 108/115 (93.91%) | 109/115 (94.78%) | +0.87% |
| subject_explicit_extend | V | 239/261 (91.57%) | 248/261 (95.02%) | +3.45% |
| subject_explicit_reduce | NV | 247/261 (94.64%) | 2/261 (0.77%) | -93.87% |
| subject_implicit_expand | V | 244/261 (93.49%) | 251/261 (96.17%) | +2.68% |
| subject_implicit_reduce | NV | 246/261 (94.25%) | 1/261 (0.38%) | -93.87% |
| subject_reallocated_expand | V | 248/261 (95.02%) | 254/261 (97.32%) | +2.30% |
| subject_reallocated_reduce | NV | 249/261 (95.40%) | 6/261 (2.30%) | -93.10% |
| term_down | TD | 18/215 (8.37%) | 47/215 (21.86%) | +13.49% |
| term_extremity_in | XT | 16/215 (7.44%) | 76/215 (35.35%) | +27.91% |
| term_extremity_out | NX | 84/99 (84.85%) | 48/99 (48.48%) | -36.36% |
| term_up | TU | 21/215 (9.77%) | 2/215 (0.93%) | -8.84% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 6734/7569 (88.97%) | 3155/7569 (41.68%) | -47.28% |
