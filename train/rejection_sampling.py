"""
拒绝采样脚本 - vLLM版本
对training_set.json中的每个样本并行采样8条回复，筛选出罪名和刑期均正确的回复作为SFT训练样本
使用vLLM加速推理
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_template import SYSTEM_PROMPT_TRAIN, format_user_prompt_train, format_articles


class RejectionSampler:
    """拒绝采样器：为每个样本生成多个候选回复，筛选正确的回复"""

    def __init__(
        self,
        model_path: str,
        num_samples: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        """
        初始化拒绝采样器（使用vLLM）

        Args:
            model_path: 模型路径
            num_samples: 每个样本采样的回复数量
            temperature: 采样温度
            top_p: nucleus sampling参数
            max_new_tokens: 最大生成token数
            tensor_parallel_size: 张量并行大小（多GPU时使用）
            gpu_memory_utilization: GPU显存利用率 (0.0-1.0)
        """
        print(f"正在加载模型 (vLLM): {model_path}")
        self.model_path = model_path
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

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

    def parse_answer(self, response: str) -> Optional[Tuple[str, str]]:
        """
        解析模型回复，提取罪名和刑期

        Args:
            response: 模型的完整回复

        Returns:
            (charge, prison_time) 或 None（解析失败）
            - charge: 罪名，不违规时为"不违规"
            - prison_time: 刑期，数字（月）或"XT"（死刑/无期），不违规时为None
        """
        # 提取<answer>标签中的内容
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if not answer_match:
            return None

        answer = answer_match.group(1).strip()

        # 判断是否违规
        if answer == "不违规" or answer.lower() == "nv":
            return ("不违规", None)

        # 解析违规案件：{罪名} | {刑期}
        pattern = r'^(.+?)\s*\|\s*(.+)$'
        match = re.match(pattern, answer)
        if not match:
            return None

        charge = match.group(1).strip()
        prison_time = match.group(2).strip()

        return (charge, prison_time)

    def normalize_prison_time(self, prison_time: Any) -> str:
        """
        标准化刑期格式

        Args:
            prison_time: 刑期（可能是数字或"XT"或字符串）

        Returns:
            标准化后的刑期字符串
        """
        if prison_time is None:
            return "None"

        prison_time_str = str(prison_time).strip()

        # 处理XT（死刑/无期徒刑）
        if prison_time_str.upper() == "XT":
            return "XT"

        # 提取数字
        match = re.search(r'\d+', prison_time_str)
        if match:
            return match.group(0)

        return prison_time_str

    def check_correctness(
        self,
        pred_charge: str,
        pred_prison_time: str,
        gt_charge: str,
        gt_prison_time: Any
    ) -> bool:
        """
        检查预测是否正确

        Args:
            pred_charge: 预测的罪名
            pred_prison_time: 预测的刑期
            gt_charge: 真实罪名
            gt_prison_time: 真实刑期

        Returns:
            是否正确
        """
        # 标准化真实刑期
        gt_prison_time_norm = self.normalize_prison_time(gt_prison_time)

        # 不违规案件
        if gt_charge == "不违规":
            return pred_charge == "不违规"

        # 违规案件：罪名和刑期都要正确
        charge_correct = pred_charge == gt_charge

        # 刑期比较
        if pred_prison_time is None:
            return False

        pred_prison_time_norm = self.normalize_prison_time(pred_prison_time)
        prison_time_correct = pred_prison_time_norm == gt_prison_time_norm

        return charge_correct and prison_time_correct

    def generate_responses(
        self,
        fact: str,
        article_dict: Dict[str, str],
        article_ids: List[str]
    ) -> List[str]:
        """
        为单个样本生成多个回复（使用vLLM）

        Args:
            fact: 案件事实
            article_dict: 法条字典
            article_ids: 相关法条ID列表

        Returns:
            生成的回复列表
        """
        # 构造prompt
        formatted_articles = format_articles(article_dict, article_ids)
        user_prompt = format_user_prompt_train(fact, formatted_articles)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TRAIN},
            {"role": "user", "content": user_prompt}
        ]

        # 应用chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            n=self.num_samples,  # 生成多个候选回复
        )

        # 使用 vLLM 生成
        outputs = self.llm.generate([prompt], sampling_params)

        # 提取生成的文本
        responses = [output.text for output in outputs[0].outputs]

        return responses

    def sample_dataset(
        self,
        training_data: List[Dict[str, Any]],
        article_dict: Dict[str, str],
        output_path: str
    ) -> Dict[str, Any]:
        """
        对整个数据集进行拒绝采样

        Args:
            training_data: 训练数据列表
            article_dict: 法条字典
            output_path: 输出文件路径

        Returns:
            采样统计信息
        """
        accepted_samples = []
        stats = {
            "total_samples": len(training_data),
            "total_generated": 0,
            "total_accepted": 0,
            "acceptance_rate": 0.0,
            "samples_with_no_correct": 0,
            "avg_correct_per_sample": 0.0
        }

        print(f"\n开始拒绝采样，共 {len(training_data)} 个样本...")

        for idx, sample in enumerate(tqdm(training_data, desc="拒绝采样")):
            fact = sample["fact"]
            article_ids = sample["relevant_articles"]
            gt_charge = sample.get("charge", "不违规")
            gt_prison_time = sample.get("prison_time", None)

            # 生成多个回复
            responses = self.generate_responses(fact, article_dict, article_ids)
            stats["total_generated"] += len(responses)

            # 筛选正确的回复
            correct_responses = []
            for response in responses:
                parsed = self.parse_answer(response)
                if parsed is None:
                    continue

                pred_charge, pred_prison_time = parsed

                # 检查是否正确
                if self.check_correctness(pred_charge, pred_prison_time, gt_charge, gt_prison_time):
                    correct_responses.append(response)

            # 保存所有正确的回复
            if len(correct_responses) > 0:
                for response in correct_responses:
                    accepted_samples.append({
                        "fact": fact,
                        "relevant_articles": article_ids,
                        "charge": gt_charge,
                        "prison_time": gt_prison_time,
                        "response": response,
                        "num_correct_for_this_sample": len(correct_responses)
                    })
                stats["total_accepted"] += len(correct_responses)
            else:
                stats["samples_with_no_correct"] += 1

            # 定期保存
            if (idx + 1) % 100 == 0:
                self._save_samples(accepted_samples, output_path)
                print(f"\n进度: {idx + 1}/{len(training_data)}, "
                      f"已接受: {stats['total_accepted']}, "
                      f"接受率: {stats['total_accepted'] / stats['total_generated'] * 100:.2f}%")

        # 最终保存
        self._save_samples(accepted_samples, output_path)

        # 计算统计信息
        stats["acceptance_rate"] = stats["total_accepted"] / stats["total_generated"] if stats["total_generated"] > 0 else 0
        stats["avg_correct_per_sample"] = stats["total_accepted"] / stats["total_samples"]

        return stats

    def _save_samples(self, samples: List[Dict[str, Any]], output_path: str):
        """保存采样结果"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)


def load_articles(articles_path: str) -> Dict[str, str]:
    """加载法条数据"""
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # 转换为字典格式
    article_dict = {}
    for article in articles:
        article_dict[article['id']] = article['content']

    return article_dict


def main():
    parser = argparse.ArgumentParser(description="拒绝采样脚本 (vLLM版本)")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--training_data", type=str, default="LawShift/training_set.json", help="训练数据路径")
    parser.add_argument("--articles", type=str, required=True, help="法条数据路径（articles_original.json）")
    parser.add_argument("--output", type=str, default="train/sampled_data.json", help="输出文件路径")
    parser.add_argument("--num_samples", type=int, default=8, help="每个样本采样的回复数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="nucleus sampling参数")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成token数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小（多GPU时使用）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU显存利用率 (0.0-1.0)")

    args = parser.parse_args()

    # 加载数据
    print(f"加载训练数据: {args.training_data}")
    with open(args.training_data, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    print(f"训练数据加载完成，共 {len(training_data)} 个样本")

    print(f"\n加载法条数据: {args.articles}")
    article_dict = load_articles(args.articles)
    print(f"法条数据加载完成，共 {len(article_dict)} 条法条")

    # 初始化采样器
    sampler = RejectionSampler(
        model_path=args.model_path,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # 执行采样
    stats = sampler.sample_dataset(
        training_data=training_data,
        article_dict=article_dict,
        output_path=args.output
    )

    # 输出统计信息
    print("\n" + "="*50)
    print("拒绝采样完成！")
    print(f"总样本数: {stats['total_samples']}")
    print(f"总生成数: {stats['total_generated']}")
    print(f"总接受数: {stats['total_accepted']}")
    print(f"接受率: {stats['acceptance_rate'] * 100:.2f}%")
    print(f"无正确回复的样本数: {stats['samples_with_no_correct']}")
    print(f"平均每个样本的正确回复数: {stats['avg_correct_per_sample']:.2f}")
    print(f"输出文件: {args.output}")
    print("="*50)

    # 保存统计信息
    stats_path = Path(args.output).parent / f"sampling_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计信息已保存至: {stats_path}")


if __name__ == "__main__":
    main()
