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
import yaml

try:
    from ..prompt_template import SYSTEM_PROMPT, format_user_prompt, format_user_prompt_poisoned, format_articles
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prompt_template import SYSTEM_PROMPT, format_user_prompt, format_user_prompt_poisoned, format_articles


class LawShiftEvaluatorVLLM:

    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
                 temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 1024,
                 num_samples: int = 1):
        """
        初始化评估器（使用vLLM）

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行数（多GPU时使用）
            gpu_memory_utilization: GPU显存利用率 (0.0-1.0)
            temperature: 温度参数
            top_p: top-p采样参数
            max_tokens: 最大生成token数
            num_samples: 采样次数（>1时对同一prompt多次采样，指标取平均值）
        """
        print(f"正在加载模型 (vLLM): {model_path}")
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_samples = num_samples

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

    def generate_predictions_batch(self, prompts: List[str]) -> List[str]:
        """
        批量生成预测（使用vLLM）

        Args:
            prompts: 提示词列表

        Returns:
            生成的文本列表
        """
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
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
                    return 36 <= int(pred_prison) <= 120
                return False

            elif label_type == "XT":
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return 36 <= int(pred_prison) <= 120
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
                    return int(pred_prison) >= 120
                return False

            elif label_type == "TD":
                # label=TD: 预测结果为 'V | {刑期T}'，T < 36
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return int(pred_prison) <= 36
                return False

            elif label_type == "XT":
                # label=XT: 预测结果为 'V | XT'
                return pred_violation == "V" and pred_prison == "XT"

            elif label_type == "NX":
                # label=NX: 预测结果为 'V | {刑期T}'，T > 36
                if pred_violation == "V" and pred_prison and pred_prison.isdigit():
                    return int(pred_prison) >= 36
                return False

        # 未知 label 类型
        return False

    def evaluate_dataset(self, folder_path: str, batch_size: int = 32, evaluate_type: str = "all") -> Dict[str, Any]:
        """
        评估指定文件夹的数据集（使用vLLM批量推理）

        Args:
            folder_path: 文件夹路径
            batch_size: 批量大小
            evaluate_type: 评估类型 (original/poisoned/all)

        Returns:
            评估结果字典
        """
        folder_name = Path(folder_path).name
        print(f"\n{'='*60}")
        print(f"正在评估: {folder_name}")
        print(f"{'='*60}")

        label_type = self.label_mapping.get(folder_name, None)
        if label_type:
            print(f"标签类型: {label_type}")
        else:
            print(f"警告: 未找到文件夹 '{folder_name}' 的标签类型")

        articles_orig, articles_pois, data_orig, data_pois = self.load_data(folder_path)

        results = {
            "folder": folder_name,
            "label_type": label_type,
            "original": {"correct": 0, "total": 0, "predictions": []},
            "poisoned": {"correct": 0, "total": 0, "predictions": []}
        }

        if evaluate_type in ["original", "all"]:
            print(f"\n评估 original.json (共{len(data_orig)}条)")
            self._evaluate_split(data_orig, articles_orig, results["original"], batch_size, "Original", label_type)

        if evaluate_type in ["poisoned", "all"]:
            print(f"\n评估 poisoned.json (共{len(data_pois)}条)")
            self._evaluate_split(data_pois, articles_pois, results["poisoned"], batch_size, "Poisoned", label_type, articles_orig)

        # 计算准确率
        if results["original"]["total"] > 0:
            results["original"]["accuracy"] = results["original"]["correct"] / results["original"]["total"]

        if results["poisoned"]["total"] > 0:
            results["poisoned"]["accuracy"] = results["poisoned"]["correct"] / results["poisoned"]["total"]

        # 打印结果
        self.print_results(results)

        return results

    def _evaluate_split(self, data: List[Dict], articles: Dict, results: Dict, batch_size: int, split_name: str, label_type: str = None, articles_original: Dict = None):
        """
        评估一个数据分支（使用vLLM批量推理，支持多次采样）

        Args:
            data: 数据列表
            articles: 法条字典
            results: 结果字典
            batch_size: 批量大小
            split_name: 分割名称
            label_type: 标签类型
            articles_original: 原始法条字典（仅在 split_name="Poisoned" 时需要）
        """
        num_samples = self.num_samples

        # 如果多次采样，初始化采样结果存储
        if num_samples > 1:
            print(f"  使用 {num_samples} 次采样")
            results["num_samples"] = num_samples

        for i in tqdm(range(0, len(data), batch_size), desc=split_name):
            batch = data[i:i + batch_size]

            prompts = []
            for item in batch:
                fact = item["fact"]
                article_ids = item["relevant_articles"]

                # 根据 split_name 选择不同的 prompt 构建方式
                if split_name == "Poisoned" and articles_original is not None:
                    # 对于 Poisoned 数据，使用旧法+新法的模板
                    articles_text_original = format_articles(articles_original, article_ids)
                    articles_text_poisoned = format_articles(articles, article_ids)
                    user_prompt = format_user_prompt_poisoned(fact, articles_text_original, articles_text_poisoned)
                else:
                    # 对于 Original 数据，使用标准模板
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
                    ) + " <think>\n"
                else:
                    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant: <think>\n"

                prompts.append(prompt)

            try:
                # 多次采样
                all_sample_results = []  # [sample_idx][item_idx] = (pred_violation, pred_prison, response)

                for sample_idx in range(num_samples):
                    responses = self.generate_predictions_batch(prompts)
                    sample_results = []
                    for response in responses:
                        pred_violation, pred_prison = self.parse_prediction(response)
                        sample_results.append((pred_violation, pred_prison, response))
                    all_sample_results.append(sample_results)

                # 处理每个样本的多次采样结果
                for item_idx, (item, prompt) in enumerate(zip(batch, prompts)):
                    fact = item["fact"]
                    article_ids = item["relevant_articles"]
                    relevant_articles_texts = [articles.get(str(aid), f"Article {aid} not found") for aid in article_ids]

                    # 收集该样本在所有采样中的结果
                    pred_violations = []
                    pred_prisons = []
                    full_responses = []
                    correct_count = 0

                    for sample_idx in range(num_samples):
                        pred_violation, pred_prison, response = all_sample_results[sample_idx][item_idx]
                        pred_violations.append(pred_violation)
                        pred_prisons.append(pred_prison)  # 不犯罪时为None
                        full_responses.append(response)

                        is_correct = self.check_prediction_success(
                            pred_violation, pred_prison, label_type, split_name
                        )
                        if is_correct:
                            correct_count += 1

                    # 计算该样本的平均准确率
                    avg_correct = correct_count / num_samples
                    results["correct"] += avg_correct

                    prediction_record = {
                        "sample_id": results["total"],
                        "pred_violation": pred_violations if num_samples > 1 else pred_violations[0],
                        "pred_prison": pred_prisons if num_samples > 1 else pred_prisons[0],
                        "is_correct": correct_count == num_samples if num_samples == 1 else None,
                        "fact": fact,
                        "relevant_articles": relevant_articles_texts,
                        "full_prompt": prompt,
                        "full_response": full_responses if num_samples > 1 else full_responses[0]
                    }

                    # 如果多次采样，添加额外信息
                    if num_samples > 1:
                        prediction_record["num_samples"] = num_samples
                        prediction_record["correct_count"] = correct_count
                        prediction_record["avg_correct"] = avg_correct

                    results["predictions"].append(prediction_record)
                    results["total"] += 1

            except Exception as e:
                print(f"\n批量预测出错: {e}")
                for item, prompt in zip(batch, prompts):
                    article_ids = item["relevant_articles"]
                    relevant_articles_texts = [articles.get(str(aid), f"Article {aid} not found") for aid in article_ids]

                    results["predictions"].append({
                        "sample_id": results["total"],
                        "error": str(e),
                        "fact": item["fact"],
                        "relevant_articles": relevant_articles_texts,
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

    def evaluate_all(self, dataset_root: str = "./LawShift", batch_size: int = 32, output_dir: str = "./results", evaluate_type: str = "all", resume: bool = False) -> Tuple[List[Dict[str, Any]], str]:
        """
        评估所有子文件夹

        Args:
            dataset_root: 数据集根目录
            batch_size: 批量大小
            output_dir: 输出目录
            evaluate_type: 评估类型 (original/poisoned/all)
            resume: 是否从已有结果恢复

        Returns:
            (所有评估结果列表, 结果保存目录)
        """
        dataset_path = Path(dataset_root)
        all_results = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.model_path).name

        # 如果是resume模式，使用已有的output_dir；否则创建新的带时间戳的目录
        if resume:
            results_dir = Path(output_dir)
            if not results_dir.exists():
                print(f"警告: 指定的output_dir不存在: {results_dir}")
                print("将创建新的评估目录...")
                results_dir = Path(output_dir) / f"{model_name}_{timestamp}"
                results_dir.mkdir(parents=True, exist_ok=True)
                resume = False
            else:
                print(f"\n从已有目录恢复评估: {results_dir}")
                # 加载已有的评估结果
                for result_file in results_dir.glob("*_results.json"):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            all_results.append(result)
                        print(f"已加载: {result_file.name}")
                    except Exception as e:
                        print(f"加载 {result_file.name} 时出错: {e}")
                print(f"已加载 {len(all_results)} 个已完成的评估结果")
        else:
            results_dir = Path(output_dir) / f"{model_name}_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n结果将保存至: {results_dir}")

        # 获取已完成的文件夹名称
        completed_folders = {result["folder"] for result in all_results}

        for folder in sorted(dataset_path.iterdir()):
            if folder.is_dir() and (folder / "original.json").exists():
                folder_name = folder.name

                # 如果是resume模式且该文件夹已完成，则跳过
                if resume and folder_name in completed_folders:
                    print(f"\n跳过已完成的文件夹: {folder_name}")
                    continue

                try:
                    results = self.evaluate_dataset(str(folder), batch_size=batch_size, evaluate_type=evaluate_type)
                    all_results.append(results)

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
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for result in all_results:
            folder_name = result["folder"]
            folder_result = {
                "folder": folder_name,
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

            result_file = output_path / f"{folder_name}_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(folder_result, f, ensure_ascii=False, indent=2)
            print(f"已保存: {result_file}")


def load_config(config_path: str) -> dict:
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LawShift数据集评估脚本（vLLM加速版）")
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluate_vllm.yaml",
        help="配置文件路径（YAML格式）"
    )
    args = parser.parse_args()

    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在 ({config_path})")
        return

    print(f"从配置文件加载参数: {config_path}")
    config = load_config(str(config_path))

    # 从配置文件提取参数
    model_path = config["model"]["model_path"]
    tensor_parallel_size = config["model"]["tensor_parallel_size"]
    gpu_memory_utilization = config["model"]["gpu_memory_utilization"]

    dataset_root = config["data"]["dataset_root"]
    output_dir = config["data"]["output_dir"]

    batch_size = config["inference"]["batch_size"]
    num_samples = config["inference"]["num_samples"]
    temperature = config["inference"]["temperature"]
    top_p = config["inference"]["top_p"]
    max_tokens = config["inference"]["max_tokens"]

    evaluate_type = config["evaluation"]["evaluate_type"]
    resume = config["evaluation"]["resume"]

    print("="*80)
    print("LawShift 法律判决预测评估 (vLLM加速版)")
    print("="*80)
    print(f"配置文件: {config_path}")
    print(f"模型路径: {model_path}")
    print(f"数据集路径: {dataset_root}")
    print(f"输出目录: {output_dir}")
    print(f"批量大小: {batch_size}")
    print(f"张量并行大小: {tensor_parallel_size}")
    print(f"GPU显存利用率: {gpu_memory_utilization}")
    print(f"采样次数: {num_samples}")
    print(f"温度: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"最大token数: {max_tokens}")
    print(f"评估类型: {evaluate_type}")
    print(f"恢复模式: {resume}")
    print("="*80)

    evaluator = LawShiftEvaluatorVLLM(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_samples=num_samples
    )

    all_results, results_dir = evaluator.evaluate_all(
        dataset_root,
        batch_size=batch_size,
        output_dir=output_dir,
        evaluate_type=evaluate_type,
        resume=resume
    )

    print("\n评估完成！")


if __name__ == "__main__":
    main()
