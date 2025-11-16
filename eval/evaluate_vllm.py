"""
法律判决预测（LJP）测评脚本 - vLLM版本
用于评估模型在LawShift数据集上的性能，使用vLLM加速推理
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm

try:
    from ..prompt_template import SYSTEM_PROMPT, format_user_prompt, format_articles
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prompt_template import SYSTEM_PROMPT, format_user_prompt, format_articles


class LawShiftEvaluatorVLLM:

    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """
        初始化评估器（使用vLLM）

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行数（多GPU时使用）
            gpu_memory_utilization: GPU显存利用率 (0.0-1.0)
        """
        print(f"正在加载模型 (vLLM): {model_path}")
        self.model_path = model_path

        # 加载 label 映射
        self.label_mapping = {}
        label_path = Path(__file__).parent.parent / "label.json"

        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
                for item in labels:
                    self.label_mapping[item['name']] = item['label']
            print(f"已加载 {len(self.label_mapping)} 个标签映射")
        else:
            print(f"警告: 未找到 label.json 文件: {label_path}")

        # 使用 vLLM 加载模型
        print("加载模型中...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        # 获取 tokenizer（用于构建 prompt）
        self.tokenizer = self.llm.get_tokenizer()

        print("模型加载完成！")
        print(f"张量并行大小: {tensor_parallel_size}")
        print(f"GPU显存利用率: {gpu_memory_utilization}")

    def load_data(self, folder_path: str) -> Tuple[Dict, Dict, List[Dict], List[Dict]]:
        """
        加载指定文件夹的数据

        Args:
            folder_path: 文件夹路径

        Returns:
            (articles_original, articles_poisoned, data_original, data_poisoned)
        """
        folder = Path(folder_path)

        # 加载法条
        with open(folder / "articles_original.json", 'r', encoding='utf-8') as f:
            articles_original = json.load(f)

        with open(folder / "articles_poisoned.json", 'r', encoding='utf-8') as f:
            articles_poisoned = json.load(f)

        # 加载测试数据
        with open(folder / "original.json", 'r', encoding='utf-8') as f:
            data_original = json.load(f)

        with open(folder / "poisoned.json", 'r', encoding='utf-8') as f:
            data_poisoned = json.load(f)

        return articles_original, articles_poisoned, data_original, data_poisoned

    def generate_predictions_batch(self, prompts: List[str], temperature: float = 0.7,
                                   top_p: float = 0.9, max_tokens: int = 1024) -> List[str]:
        """
        批量生成预测（使用vLLM）

        Args:
            prompts: 提示词列表
            temperature: 温度参数
            top_p: top-p采样参数
            max_tokens: 最大生成token数

        Returns:
            生成的文本列表
        """
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # 批量生成
        outputs = self.llm.generate(prompts, sampling_params)

        # 提取生成的文本
        responses = [output.outputs[0].text for output in outputs]

        return responses

    def parse_prediction(self, response: str) -> Tuple[str, str]:
        """
        解析模型输出

        Args:
            response: 模型生成的文本

        Returns:
            (违规判断, 刑期)
            - 违规判断: "V" 或 "NV"
            - 刑期: 数字字符串、"XT"、None（如果只有V/NV）
        """
        # 模式1: <answer> NV </answer>
        if re.search(r'<answer>\s*NV\s*</answer>', response, re.IGNORECASE):
            return "NV", None

        # 模式2: <answer> V | XT </answer> (死刑/无期)
        xt_pattern = r'<answer>\s*V\s*\|\s*XT\s*</answer>'
        if re.search(xt_pattern, response, re.IGNORECASE):
            return "V", "XT"

        # 模式3: <answer> V | {数字} </answer> (标准违规+刑期)
        v_prison_pattern = r'<answer>\s*V\s*\|\s*(\d+)\s*</answer>'
        match = re.search(v_prison_pattern, response, re.IGNORECASE)
        if match:
            prison_time = match.group(1).strip()
            return "V", prison_time

        # 模式4: <answer> V </answer> (只有违规判断)
        if re.search(r'<answer>\s*V\s*</answer>', response, re.IGNORECASE):
            return "V", None

        # 如果都没匹配到，返回未识别
        return "未识别", None

    def check_prediction_success(self, pred_violation: str, pred_prison: str,
                                  label_type: str, split_name: str) -> bool:
        """
        根据不同的 label 类型判断预测是否成功

        Args:
            pred_violation: 预测的违规判断 ("V" 或 "NV")
            pred_prison: 预测的刑期 (数字字符串、"XT" 或 None)
            label_type: 标签类型 (V, NV, TU, TD, XT, NX)
            split_name: 分支名称 (Original, Poisoned)

        Returns:
            是否预测成功
        """
        if split_name == "Original":
            # 根据不同 label 类型判断成功标准
            if label_type == "V" or label_type == "NV":
                return pred_violation == "V"

            elif label_type == "TU" or label_type == "TD":
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return 36 < int(pred_prison) < 120
                return False

            elif label_type == "XT":
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return 36 < int(pred_prison) < 120
                return False

            elif label_type == "NX":
                return pred_violation == "V" and pred_prison == "XT"

        elif split_name == "Poisoned":
            # 根据不同 label 类型判断成功标准
            if label_type == "V":
                # label=V: 预测结果为 V（不需要考虑刑期）
                return pred_violation == "V"

            elif label_type == "NV":
                # label=NV: 预测结果为 NV
                return pred_violation == "NV"

            elif label_type == "TU":
                # label=TU: 预测结果为 'V | {刑期T}'，T > 120
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return int(pred_prison) > 120
                return False

            elif label_type == "TD":
                # label=TD: 预测结果为 'V | {刑期T}'，T < 36
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return int(pred_prison) < 36
                return False

            elif label_type == "XT":
                # label=XT: 预测结果为 'V | XT'
                return pred_violation == "V" and pred_prison == "XT"

            elif label_type == "NX":
                # label=NX: 预测结果为 'V | {刑期T}'，T > 36
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return int(pred_prison) > 36
                return False

        # 未知 label 类型
        return False

    def evaluate_dataset(self, folder_path: str, batch_size: int = 32) -> Dict[str, Any]:
        """
        评估指定文件夹的数据集（使用vLLM批量推理）

        Args:
            folder_path: 文件夹路径
            batch_size: 批量大小（vLLM会自动优化，这里主要用于内存管理）

        Returns:
            评估结果字典
        """
        folder_name = Path(folder_path).name
        print(f"\n{'='*60}")
        print(f"正在评估: {folder_name}")
        print(f"{'='*60}")

        # 从 label_mapping 中获取该文件夹的 label 类型
        label_type = self.label_mapping.get(folder_name, None)
        if label_type:
            print(f"标签类型: {label_type}")
        else:
            print(f"警告: 未找到文件夹 '{folder_name}' 的标签类型")

        # 加载数据
        articles_orig, articles_pois, data_orig, data_pois = self.load_data(folder_path)

        results = {
            "folder": folder_name,
            "label_type": label_type,
            "original": {"correct": 0, "total": 0, "predictions": []},
            "poisoned": {"correct": 0, "total": 0, "predictions": []}
        }

        # 评估original数据（批量推理）
        print(f"\n评估 original.json (共{len(data_orig)}条)")
        self._evaluate_split(data_orig, articles_orig, results["original"], batch_size, "Original", label_type)

        # 评估poisoned数据（批量推理）
        print(f"\n评估 poisoned.json (共{len(data_pois)}条)")
        self._evaluate_split(data_pois, articles_pois, results["poisoned"], batch_size, "Poisoned", label_type)

        # 计算准确率
        if results["original"]["total"] > 0:
            results["original"]["accuracy"] = results["original"]["correct"] / results["original"]["total"]

        if results["poisoned"]["total"] > 0:
            results["poisoned"]["accuracy"] = results["poisoned"]["correct"] / results["poisoned"]["total"]

        # 打印结果
        self.print_results(results)

        return results

    def _evaluate_split(self, data: List[Dict], articles: Dict, results: Dict, batch_size: int, split_name: str, label_type: str = None):
        """
        评估一个数据分支（使用vLLM批量推理）

        Args:
            data: 数据列表
            articles: 法条字典
            results: 结果字典
            batch_size: 批量大小
            split_name: 分割名称（用于进度条）
            label_type: 标签类型（V, NV, TU, TD, XT, NX）
        """
        # 批量处理
        for i in tqdm(range(0, len(data), batch_size), desc=split_name):
            batch = data[i:i + batch_size]

            # 准备批量提示词
            prompts = []
            for item in batch:
                fact = item["fact"]
                article_ids = item["relevant_articles"]
                articles_text = format_articles(articles, article_ids)
                user_prompt = format_user_prompt(fact, articles_text)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

                if hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"

                prompts.append(prompt)

            # 批量生成
            try:
                responses = self.generate_predictions_batch(prompts)

                # 处理每个结果
                for item, prompt, response in zip(batch, prompts, responses):
                    fact = item["fact"]

                    # 解析预测结果
                    pred_violation, pred_prison = self.parse_prediction(response)

                    # 使用新的评估逻辑判断是否成功
                    is_correct = self.check_prediction_success(
                        pred_violation, pred_prison, label_type, split_name
                    )

                    if is_correct:
                        results["correct"] += 1

                    # 保存评估结果（包含完整的prompt和response）
                    results["predictions"].append({
                        "sample_id": results["total"],
                        "fact": fact[:100] + "...",
                        "pred_violation": pred_violation,
                        "pred_prison": pred_prison,
                        "is_correct": is_correct,
                        "full_prompt": prompt,
                        "full_response": response
                    })

                    results["total"] += 1

            except Exception as e:
                print(f"\n批量预测出错: {e}")
                # 出错时记录错误
                for item, prompt in zip(batch, prompts):
                    results["predictions"].append({
                        "sample_id": results["total"],
                        "error": str(e),
                        "fact": item["fact"][:100] + "...",
                        "full_prompt": prompt
                    })
                    results["total"] += 1

    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"评估结果: {results['folder']}")
        if results.get('label_type'):
            print(f"标签类型: {results['label_type']}")
        print(f"{'='*60}")

        print("\n【Original数据】")
        orig = results["original"]
        print(f"  总数: {orig['total']}")
        print(f"  成功率: {orig.get('accuracy', 0):.2%} ({orig['correct']}/{orig['total']})")

        print("\n【Poisoned数据】")
        pois = results["poisoned"]
        print(f"  总数: {pois['total']}")
        print(f"  成功率: {pois.get('accuracy', 0):.2%} ({pois['correct']}/{pois['total']})")

        # 对比分析
        if orig['total'] > 0 and pois['total'] > 0:
            accuracy_diff = pois.get('accuracy', 0) - orig.get('accuracy', 0)

            print("\n【对比分析】")
            print(f"  成功率变化: {accuracy_diff:+.2%}")

            # 根据 label 类型显示评估标准
            label_type = results.get('label_type')
            if label_type:
                print(f"\n【评估标准 (基于 label={label_type})】")
                if label_type == "V":
                    print(f"  预测结果为 V → 成功")
                elif label_type == "NV":
                    print(f"  预测结果为 NV → 成功")
                elif label_type == "TU":
                    print(f"  预测结果为 'V | {{刑期T}}'，且 T > 120 → 成功")
                elif label_type == "TD":
                    print(f"  预测结果为 'V | {{刑期T}}'，且 T < 36 → 成功")
                elif label_type == "XT":
                    print(f"  预测结果为 'V | XT' → 成功")
                elif label_type == "NX":
                    print(f"  预测结果为 'V | {{刑期T}}'，且 T > 36 → 成功")

    def evaluate_all(self, dataset_root: str = "./LawShift", batch_size: int = 32, output_dir: str = "./results") -> Tuple[List[Dict[str, Any]], str]:
        """
        评估所有子文件夹（每完成一个folder就保存一次）

        Args:
            dataset_root: 数据集根目录
            batch_size: 批量大小
            output_dir: 输出目录

        Returns:
            (所有评估结果列表, 结果保存目录)
        """
        dataset_path = Path(dataset_root)
        all_results = []

        # 生成一次时间戳，所有保存都使用同一个时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.model_path).name

        # 创建以模型名称和时间戳命名的子目录
        results_dir = Path(output_dir) / f"{model_name}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n结果将保存至: {results_dir}")

        # 遍历所有子文件夹
        for folder in sorted(dataset_path.iterdir()):
            if folder.is_dir() and (folder / "original.json").exists():
                try:
                    results = self.evaluate_dataset(str(folder), batch_size=batch_size)
                    all_results.append(results)

                    # 每完成一个folder就保存一次（增量保存，覆盖同一个文件）
                    print(f"\n💾 保存当前结果 ({len(all_results)} 个文件夹已完成)...")
                    self.save_results(all_results, str(results_dir))

                except Exception as e:
                    print(f"\n评估 {folder.name} 时出错: {e}")
                    continue

        return all_results, str(results_dir)

    def save_results(self, all_results: List[Dict[str, Any]], output_dir: str = "./results"):
        """
        保存评估结果

        Args:
            all_results: 所有评估结果
            output_dir: 输出目录（已经是包含模型名称和时间戳的子目录）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存详细结果（JSON）- 包含完整的prompt和response
        detailed_results = []
        for result in all_results:
            detailed_result = {
                "folder": result["folder"],
                "label_type": result.get("label_type"),
                "original": {
                    "correct": result["original"]["correct"],
                    "total": result["original"]["total"],
                    "accuracy": result["original"].get("accuracy"),
                    "predictions": result["original"]["predictions"]
                },
                "poisoned": {
                    "correct": result["poisoned"]["correct"],
                    "total": result["poisoned"]["total"],
                    "accuracy": result["poisoned"].get("accuracy"),
                    "predictions": result["poisoned"]["predictions"]
                }
            }
            detailed_results.append(detailed_result)

        detailed_file = output_path / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至: {detailed_file}")

        # 保存汇总结果（Markdown）
        summary_file = output_path / "summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            # 标题和基本信息
            f.write(f"# LawShift 数据集评估报告 (vLLM版本)\n\n")
            f.write(f"**模型路径**: {self.model_path}\n\n")
            f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 统计总体结果
            total_orig_correct = 0
            total_orig_count = 0
            total_pois_correct = 0
            total_pois_count = 0

            # 各文件夹评估结果表格
            f.write(f"## 各文件夹评估结果\n\n")
            f.write(f"| 文件夹名称 | Label Type | Original | Poisoned | Comparison |\n")
            f.write(f"|-----------|-----------|----------|----------|------------|\n")

            for results in all_results:
                folder_name = results["folder"]
                label_type = results.get("label_type", "")

                # Original结果
                orig = results["original"]
                orig_text = f"{orig['correct']}/{orig['total']} ({orig.get('accuracy', 0):.2%})"

                # Poisoned结果
                pois = results["poisoned"]
                pois_text = f"{pois['correct']}/{pois['total']} ({pois.get('accuracy', 0):.2%})"

                # 对比分析
                comparison_text = ""
                if orig['total'] > 0 and pois['total'] > 0:
                    accuracy_diff = pois.get('accuracy', 0) - orig.get('accuracy', 0)
                    comparison_text = f"{accuracy_diff:+.2%}"

                # 写入表格行
                f.write(f"| {folder_name} | {label_type} | {orig_text} | {pois_text} | {comparison_text} |\n")

                # 累加统计
                total_orig_correct += orig['correct']
                total_orig_count += orig['total']
                total_pois_correct += pois['correct']
                total_pois_count += pois['total']

            # 总体统计表格
            f.write(f"\n## 总体统计\n\n")
            f.write(f"| 文件夹名称 | Label Type | Original | Poisoned | Comparison |\n")
            f.write(f"|-----------|-----------|----------|----------|------------|\n")

            if total_orig_count > 0 and total_pois_count > 0:
                orig_acc = total_orig_correct / total_orig_count
                pois_acc = total_pois_correct / total_pois_count
                orig_text = f"{total_orig_correct}/{total_orig_count} ({orig_acc:.2%})"
                pois_text = f"{total_pois_correct}/{total_pois_count} ({pois_acc:.2%})"
                comparison_text = f"{(pois_acc - orig_acc):+.2%}"
                f.write(f"| **总体** | | {orig_text} | {pois_text} | {comparison_text} |\n")

        print(f"汇总结果已保存至: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LawShift数据集评估脚本（vLLM加速版）")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-0.6B",
        help="模型路径"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./LawShift",
        help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批量推理的batch size（vLLM会自动优化）"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小（多GPU时使用）"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU显存利用率 (0.0-1.0)"
    )

    args = parser.parse_args()

    print("="*80)
    print("LawShift 法律判决预测评估 (vLLM加速版)")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"数据集路径: {args.dataset_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"批量大小: {args.batch_size}")
    print(f"张量并行大小: {args.tensor_parallel_size}")
    print(f"GPU显存利用率: {args.gpu_memory_utilization}")
    print("="*80)

    # 创建评估器
    evaluator = LawShiftEvaluatorVLLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # 评估所有数据
    all_results, results_dir = evaluator.evaluate_all(
        args.dataset_root,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # 最终再保存一次（确保完整）
    evaluator.save_results(all_results, results_dir)

    print("\n评估完成！")


if __name__ == "__main__":
    main()
