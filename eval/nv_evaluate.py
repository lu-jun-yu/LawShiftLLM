"""
NV（无罪）类型数据集测评脚本
用于评估模型在 NV_construct 数据集上的性能
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import yaml

try:
    from ..prompt_template import SYSTEM_PROMPT, format_user_prompt, format_articles
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prompt_template import SYSTEM_PROMPT, format_user_prompt, format_articles


class NVEvaluator:

    def __init__(self, model_path: str, device: str = "auto", use_flash_attn: bool = False,
                 temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 1024,
                 num_samples: int = 1, max_prompt_length: int = 0):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            device: 设备类型 (auto/cuda/cpu)
            use_flash_attn: 是否使用 Flash Attention 2
            temperature: 温度参数
            top_p: top-p采样参数
            max_tokens: 最大生成token数
            num_samples: 采样次数（>1时对同一prompt多次采样，指标取平均值）
            max_prompt_length: 最大prompt长度（token数），0表示不限制
        """
        print(f"正在加载模型: {model_path}")
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.max_prompt_length = max_prompt_length

        # 加载法条
        self.articles = {}
        articles_path = Path(__file__).parent.parent / "articles.json"
        if articles_path.exists():
            with open(articles_path, 'r', encoding='utf-8') as f:
                self.articles = json.load(f)
            print(f"已加载 {len(self.articles)} 条法条")
        else:
            print(f"警告: 未找到 articles.json 文件: {articles_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        print("加载模型中...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device,
        }

        # 自动选择最佳精度
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_kwargs["dtype"] = torch.bfloat16
                print("使用 BF16 精度")
            else:
                model_kwargs["dtype"] = torch.float16
                print("使用 FP16 精度")
        else:
            model_kwargs["dtype"] = torch.float32

        # 尝试使用 Flash Attention 2
        if use_flash_attn:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("使用 Flash Attention 2 加速")
            except Exception:
                print("Flash Attention 2 不可用，使用默认注意力")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        self.model.eval()

        print("模型加载完成！")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def load_nv_data(self, file_path: str) -> List[Dict]:
        """
        加载 NV 数据文件

        Args:
            file_path: JSON 文件路径

        Returns:
            数据列表，每个元素包含 fact, relevant_articles, label
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def generate_predictions_batch(self, prompts: List[str]) -> List[str]:
        """
        批量生成预测

        Args:
            prompts: 提示词列表

        Returns:
            生成的文本列表
        """
        # 批量编码，使用padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 批量解码
        responses = []
        input_lengths = inputs['input_ids'].shape[1]
        for output in outputs:
            # 只取新生成的部分
            response = self.tokenizer.decode(
                output[input_lengths:],
                skip_special_tokens=True
            )
            responses.append(response)

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

    def check_prediction_success(self, pred_violation: str, label: str) -> bool:
        """
        判断预测是否成功

        Args:
            pred_violation: 预测的违规判断 ("V" 或 "NV")
            label: 标签（对于 NV 数据集，所有标签都是 "NV"）

        Returns:
            是否预测成功
        """
        # NV 数据集：预测结果应该是 NV
        return pred_violation == label

    def evaluate_file(self, file_path: str, batch_size: int = 8) -> Dict[str, Any]:
        """
        评估单个 NV 数据文件

        Args:
            file_path: 文件路径
            batch_size: 批量大小

        Returns:
            评估结果字典
        """
        file_name = Path(file_path).stem
        print(f"\n{'='*60}")
        print(f"正在评估: {file_name}")
        print(f"{'='*60}")

        data = self.load_nv_data(file_path)
        print(f"原始数据条数: {len(data)}")

        # 预处理：构建所有 prompt 并过滤超长的
        processed_data = []
        skipped_count = 0

        for item in data:
            fact = item["fact"]
            article_ids = item["relevant_articles"]

            # 获取法条内容
            articles_text = format_articles(self.articles, article_ids)
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

            # 检查 prompt 长度
            if self.max_prompt_length > 0:
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_tokens) > self.max_prompt_length:
                    skipped_count += 1
                    continue

            processed_data.append({
                "item": item,
                "prompt": prompt
            })

        if self.max_prompt_length > 0:
            print(f"过滤超长prompt后数据条数: {len(processed_data)} (跳过 {skipped_count} 条，超过 {self.max_prompt_length} tokens)")

        results = {
            "file": file_name,
            "correct": 0,
            "total": 0,
            "skipped": skipped_count,
            "predictions": []
        }

        num_samples = self.num_samples
        if num_samples > 1:
            print(f"  使用 {num_samples} 次采样")
            results["num_samples"] = num_samples

        for i in tqdm(range(0, len(processed_data), batch_size), desc=file_name):
            batch_data = processed_data[i:i + batch_size]
            batch = [d["item"] for d in batch_data]
            prompts = [d["prompt"] for d in batch_data]

            try:
                # 多次采样
                all_sample_results = []

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
                    label = item["label"]
                    relevant_articles_texts = [self.articles.get(str(aid), f"Article {aid} not found") for aid in article_ids]

                    # 收集该样本在所有采样中的结果
                    pred_violations = []
                    pred_prisons = []
                    full_responses = []
                    correct_count = 0

                    for sample_idx in range(num_samples):
                        pred_violation, pred_prison, response = all_sample_results[sample_idx][item_idx]
                        pred_violations.append(pred_violation)
                        pred_prisons.append(pred_prison)
                        full_responses.append(response)

                        is_correct = self.check_prediction_success(pred_violation, label)
                        if is_correct:
                            correct_count += 1

                    # 计算该样本的平均准确率
                    avg_correct = correct_count / num_samples
                    results["correct"] += avg_correct

                    prediction_record = {
                        "sample_id": results["total"],
                        "pred_violation": pred_violations if num_samples > 1 else pred_violations[0],
                        "pred_prison": pred_prisons if num_samples > 1 else pred_prisons[0],
                        "label": label,
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
                    relevant_articles_texts = [self.articles.get(str(aid), f"Article {aid} not found") for aid in article_ids]

                    results["predictions"].append({
                        "sample_id": results["total"],
                        "error": str(e),
                        "fact": item["fact"],
                        "label": item["label"],
                        "relevant_articles": relevant_articles_texts,
                        "full_prompt": prompt
                    })
                    results["total"] += 1

        # 计算准确率
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]

        # 打印结果
        self.print_results(results)

        return results

    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"评估结果: {results['file']}")
        print(f"{'='*60}")

        print(f"  总数: {results['total']}")
        print(f"  正确数: {results['correct']}")
        print(f"  准确率: {results.get('accuracy', 0):.2%}")

        print("\n【评估标准】")
        print(f"  预测结果为 NV → 成功")

    def evaluate_all(self, nv_data_root: str = "./NV_construct/output", batch_size: int = 8,
                     output_dir: str = "./results", resume: bool = False) -> Tuple[List[Dict[str, Any]], str]:
        """
        评估所有 NV 数据文件

        Args:
            nv_data_root: NV 数据集根目录
            batch_size: 批量大小
            output_dir: 输出目录
            resume: 是否从已有结果恢复

        Returns:
            (所有评估结果列表, 结果保存目录)
        """
        nv_data_path = Path(nv_data_root)
        all_results = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.model_path).name

        # 如果是resume模式，使用已有的output_dir；否则创建新的带时间戳的目录
        if resume:
            results_dir = Path(output_dir)
            if not results_dir.exists():
                print(f"警告: 指定的output_dir不存在: {results_dir}")
                print("将创建新的评估目录...")
                results_dir = Path(output_dir) / f"nv_{model_name}_{timestamp}"
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
            results_dir = Path(output_dir) / f"nv_{model_name}_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n结果将保存至: {results_dir}")

        # 获取已完成的文件名称
        completed_files = {result["file"] for result in all_results}

        # 遍历所有 JSON 文件
        for json_file in sorted(nv_data_path.glob("*.json")):
            file_name = json_file.stem

            # 如果是resume模式且该文件已完成，则跳过
            if resume and file_name in completed_files:
                print(f"\n跳过已完成的文件: {file_name}")
                continue

            try:
                results = self.evaluate_file(str(json_file), batch_size=batch_size)
                all_results.append(results)

                print(f"\n💾 保存当前结果 ({len(all_results)} 个文件已完成)...")
                self.save_results(all_results, str(results_dir))

            except Exception as e:
                print(f"\n评估 {json_file.name} 时出错: {e}")
                continue

        # 打印总体结果
        self.print_summary(all_results)

        return all_results, str(results_dir)

    def print_summary(self, all_results: List[Dict[str, Any]]):
        """打印总体评估结果摘要"""
        print(f"\n{'='*60}")
        print("总体评估结果摘要")
        print(f"{'='*60}")

        total_correct = 0
        total_samples = 0

        for result in all_results:
            file_name = result["file"]
            accuracy = result.get("accuracy", 0)
            correct = result["correct"]
            total = result["total"]

            print(f"  {file_name}: {accuracy:.2%} ({correct}/{total})")

            total_correct += correct
            total_samples += total

        if total_samples > 0:
            overall_accuracy = total_correct / total_samples
            print(f"\n  总体准确率: {overall_accuracy:.2%} ({total_correct}/{total_samples})")

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
            file_name = result["file"]
            file_result = {
                "file": file_name,
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result.get("accuracy"),
                "predictions": result["predictions"]
            }

            result_file = output_path / f"{file_name}_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(file_result, f, ensure_ascii=False, indent=2)
            print(f"已保存: {result_file}")

        # 保存摘要
        summary = {
            "total_files": len(all_results),
            "results": []
        }

        total_correct = 0
        total_samples = 0

        for result in all_results:
            summary["results"].append({
                "file": result["file"],
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result.get("accuracy")
            })
            total_correct += result["correct"]
            total_samples += result["total"]

        if total_samples > 0:
            summary["overall_accuracy"] = total_correct / total_samples
            summary["overall_correct"] = total_correct
            summary["overall_total"] = total_samples

        summary_file = output_path / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"已保存摘要: {summary_file}")


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
    parser = argparse.ArgumentParser(description="NV数据集评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="config/nv_evaluate.yaml",
        help="配置文件路径（YAML格式）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--nv_data_root",
        type=str,
        default=None,
        help="NV数据集路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批量大小（覆盖配置文件）"
    )
    args = parser.parse_args()

    # 加载配置文件
    config_path = Path(args.config)
    if config_path.exists():
        print(f"从配置文件加载参数: {config_path}")
        config = load_config(str(config_path))
    else:
        print(f"配置文件不存在 ({config_path})，使用默认参数")
        config = {
            "model": {
                "model_path": "./models/Qwen2.5-7B-Instruct",
                "device": "auto",
                "use_flash_attn": False
            },
            "data": {
                "nv_data_root": "./NV_construct/output",
                "output_dir": "./results"
            },
            "inference": {
                "batch_size": 8,
                "num_samples": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "max_prompt_length": 0
            },
            "evaluation": {
                "resume": False
            }
        }

    # 从配置文件提取参数（命令行参数优先）
    model_path = args.model_path or config["model"]["model_path"]
    device = config["model"]["device"]
    use_flash_attn = config["model"]["use_flash_attn"]

    nv_data_root = args.nv_data_root or config["data"]["nv_data_root"]
    output_dir = args.output_dir or config["data"]["output_dir"]

    batch_size = args.batch_size or config["inference"]["batch_size"]
    num_samples = config["inference"]["num_samples"]
    temperature = config["inference"]["temperature"]
    top_p = config["inference"]["top_p"]
    max_tokens = config["inference"]["max_tokens"]
    max_prompt_length = config["inference"].get("max_prompt_length", 0)

    resume = config["evaluation"]["resume"]

    print("="*80)
    print("NV 数据集评估")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"数据集路径: {nv_data_root}")
    print(f"输出目录: {output_dir}")
    print(f"批量大小: {batch_size}")
    print(f"设备: {device}")
    print(f"Flash Attention: {use_flash_attn}")
    print(f"采样次数: {num_samples}")
    print(f"温度: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"最大token数: {max_tokens}")
    print(f"最大prompt长度: {max_prompt_length if max_prompt_length > 0 else '不限制'}")
    print(f"恢复模式: {resume}")
    print("="*80)

    evaluator = NVEvaluator(
        model_path,
        device=device,
        use_flash_attn=use_flash_attn,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_samples=num_samples,
        max_prompt_length=max_prompt_length
    )

    all_results, results_dir = evaluator.evaluate_all(
        nv_data_root,
        batch_size=batch_size,
        output_dir=output_dir,
        resume=resume
    )

    print("\n评估完成！")


if __name__ == "__main__":
    main()
