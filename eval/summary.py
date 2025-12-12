"""
生成 LawShift 评估结果的 summary.md 文件
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def generate_summary(results_dir: str):
    """
    根据指定目录下的 *_results.json 文件生成 summary.md

    Args:
        results_dir: 结果文件所在目录
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"错误: 目录不存在: {results_dir}")
        return

    # 查找所有 *_results.json 文件
    result_files = list(results_path.glob("*_results.json"))

    if not result_files:
        print(f"警告: 在 {results_dir} 中未找到任何 *_results.json 文件")
        return

    # 加载所有结果
    all_results = []
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                all_results.append(result)
            print(f"已加载: {result_file.name}")
        except Exception as e:
            print(f"加载 {result_file.name} 时出错: {e}")

    if not all_results:
        print("错误: 没有成功加载任何结果文件")
        return

    print(f"\n共加载 {len(all_results)} 个评估结果")

    # 生成 summary.md
    summary_file = results_path / "summary.md"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# LawShift 数据集评估报告\n\n")
        f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total_orig_correct = 0
        total_orig_count = 0
        total_pois_correct = 0
        total_pois_count = 0

        f.write(f"## 各文件夹评估结果\n\n")
        f.write(f"| 文件夹名称 | Label Type | Original | Poisoned | Comparison |\n")
        f.write(f"|-----------|-----------|----------|----------|------------|\n")

        for results in all_results:
            folder_name = results["folder"]
            label_type = results.get("label_type", "")

            orig = results["original"]
            if orig['total'] > 0:
                orig_acc = orig.get('accuracy', 0) if orig.get('accuracy') is not None else 0
                orig_correct = orig['correct']
                # 处理 correct 可能是浮点数的情况（多次采样时）
                if isinstance(orig_correct, float) and not orig_correct.is_integer():
                    orig_text = f"{orig_correct:.2f}/{orig['total']} ({orig_acc:.2%})"
                else:
                    orig_text = f"{int(orig_correct)}/{orig['total']} ({orig_acc:.2%})"
            else:
                orig_text = "0/0 (N/A)"

            pois = results["poisoned"]
            if pois['total'] > 0:
                pois_acc = pois.get('accuracy', 0) if pois.get('accuracy') is not None else 0
                pois_correct = pois['correct']
                # 处理 correct 可能是浮点数的情况（多次采样时）
                if isinstance(pois_correct, float) and not pois_correct.is_integer():
                    pois_text = f"{pois_correct:.2f}/{pois['total']} ({pois_acc:.2%})"
                else:
                    pois_text = f"{int(pois_correct)}/{pois['total']} ({pois_acc:.2%})"
            else:
                pois_text = "0/0 (N/A)"

            comparison_text = ""
            if orig['total'] > 0 and pois['total'] > 0:
                orig_accuracy = orig.get('accuracy', 0) if orig.get('accuracy') is not None else 0
                pois_accuracy = pois.get('accuracy', 0) if pois.get('accuracy') is not None else 0
                accuracy_diff = pois_accuracy - orig_accuracy
                comparison_text = f"{accuracy_diff:+.2%}"

            f.write(f"| {folder_name} | {label_type} | {orig_text} | {pois_text} | {comparison_text} |\n")

            total_orig_correct += orig['correct']
            total_orig_count += orig['total']
            total_pois_correct += pois['correct']
            total_pois_count += pois['total']

        f.write(f"\n## 总体统计\n\n")
        f.write(f"| 文件夹名称 | Label Type | Original | Poisoned | Comparison |\n")
        f.write(f"|-----------|-----------|----------|----------|------------|\n")

        if total_orig_count > 0 or total_pois_count > 0:
            orig_text = "N/A"
            pois_text = "N/A"
            comparison_text = ""

            if total_orig_count > 0:
                orig_acc = total_orig_correct / total_orig_count
                # 处理 correct 可能是浮点数的情况（多次采样时）
                if isinstance(total_orig_correct, float) and not total_orig_correct.is_integer():
                    orig_text = f"{total_orig_correct:.2f}/{total_orig_count} ({orig_acc:.2%})"
                else:
                    orig_text = f"{int(total_orig_correct)}/{total_orig_count} ({orig_acc:.2%})"

            if total_pois_count > 0:
                pois_acc = total_pois_correct / total_pois_count
                # 处理 correct 可能是浮点数的情况（多次采样时）
                if isinstance(total_pois_correct, float) and not total_pois_correct.is_integer():
                    pois_text = f"{total_pois_correct:.2f}/{total_pois_count} ({pois_acc:.2%})"
                else:
                    pois_text = f"{int(total_pois_correct)}/{total_pois_count} ({pois_acc:.2%})"

            if total_orig_count > 0 and total_pois_count > 0:
                comparison_text = f"{(pois_acc - orig_acc):+.2%}"

            f.write(f"| **总体** | | {orig_text} | {pois_text} | {comparison_text} |\n")

    print(f"\n汇总结果已保存至: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成 LawShift 评估结果的 summary.md 文件")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="包含 *_results.json 文件的结果目录"
    )

    args = parser.parse_args()

    print("="*80)
    print("LawShift 评估结果汇总生成工具")
    print("="*80)
    print(f"结果目录: {args.results_dir}")
    print("="*80)

    generate_summary(args.results_dir)

    print("\n汇总完成！")


if __name__ == "__main__":
    main()
