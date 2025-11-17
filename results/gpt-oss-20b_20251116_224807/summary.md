# LawShift 数据集评估报告 (vLLM版本)

**模型路径**: models/gpt-oss-20b

**评估时间**: 2025-11-17 00:15:32

## 各文件夹评估结果

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| action_explicit_extend | V | 86/102 (84.31%) | 82/102 (80.39%) | -3.92% |
| action_explicit_reduce | NV | 86/102 (84.31%) | 0/102 (0.00%) | -84.31% |
| action_implicit_expand | V | 94/102 (92.16%) | 72/102 (70.59%) | -21.57% |
| action_implicit_reduce | NV | 92/102 (90.20%) | 45/102 (44.12%) | -46.08% |
| action_reallocated_expand | V | 93/102 (91.18%) | 76/102 (74.51%) | -16.67% |
| action_reallocated_reduce | NV | 89/102 (87.25%) | 32/102 (31.37%) | -55.88% |
| objCon_addition | NV | 213/409 (52.08%) | 60/409 (14.67%) | -37.41% |
| objCon_explicit_extend | NV | 70/150 (46.67%) | 3/150 (2.00%) | -44.67% |
| objCon_implicit_expand | NV | 77/150 (51.33%) | 2/150 (1.33%) | -50.00% |
| objCon_implicit_reduce | V | 73/150 (48.67%) | 60/150 (40.00%) | -8.67% |
| objCon_reallocated_expand | NV | 72/150 (48.00%) | 3/150 (2.00%) | -46.00% |
| objCon_reallocated_reduce | V | 73/150 (48.67%) | 68/150 (45.33%) | -3.33% |
| objCon_removal | V | 62/149 (41.61%) | 45/149 (30.20%) | -11.41% |
| object_explicit_extend | V | 202/459 (44.01%) | 106/459 (23.09%) | -20.92% |
| object_explicit_reduce | NV | 201/459 (43.79%) | 207/459 (45.10%) | +1.31% |

## 总体统计

| 文件夹名称 | Label Type | Original | Poisoned | Comparison |
|-----------|-----------|----------|----------|------------|
| **总体** | | 1583/2838 (55.78%) | 861/2838 (30.34%) | -25.44% |
