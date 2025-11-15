"""
QLoRA SFT训练脚本
使用拒绝采样得到的数据进行监督微调，支持梯度权重调整（1/n）
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import BitsAndBytesConfig
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_template import SYSTEM_PROMPT_TRAIN, format_user_prompt_train, format_articles


@dataclass
class SFTConfig:
    """SFT训练配置"""
    # 模型相关
    model_path: str = field(metadata={"help": "基础模型路径"})
    output_dir: str = field(default="train/checkpoints", metadata={"help": "输出目录"})

    # 数据相关
    sampled_data: str = field(default="train/sampled_data.json", metadata={"help": "拒绝采样得到的数据路径"})
    articles_path: str = field(metadata={"help": "法条数据路径"})
    max_length: int = field(default=4096, metadata={"help": "最大序列长度"})

    # LoRA相关
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA目标模块"}
    )

    # 训练相关
    num_train_epochs: int = field(default=3, metadata={"help": "训练轮数"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "每个设备的批大小"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "梯度累积步数"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    warmup_steps: int = field(default=100, metadata={"help": "预热步数"})
    logging_steps: int = field(default=10, metadata={"help": "日志记录步数"})
    save_steps: int = field(default=100, metadata={"help": "保存步数"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减"})

    # 量化相关
    use_4bit: bool = field(default=True, metadata={"help": "是否使用4bit量化"})
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"help": "4bit计算类型"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "4bit量化类型"})


class LegalSFTDataset(Dataset):
    """法律判决SFT数据集，支持样本权重"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        article_dict: Dict[str, str],
        tokenizer: AutoTokenizer,
        max_length: int = 4096
    ):
        """
        初始化数据集

        Args:
            data: 拒绝采样得到的数据
            article_dict: 法条字典
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.article_dict = article_dict
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 处理数据并计算权重
        self.processed_data = self._process_data()

    def _process_data(self) -> List[Dict[str, Any]]:
        """处理数据，为每个样本添加权重"""
        processed = []

        for item in self.data:
            fact = item["fact"]
            article_ids = item["relevant_articles"]
            response = item["response"]
            num_correct = item["num_correct_for_this_sample"]  # 同一问题的正确回复数

            # 计算权重：1/n（保证无偏性）
            weight = 1.0 / num_correct

            # 构造prompt
            formatted_articles = format_articles(self.article_dict, article_ids)
            user_prompt = format_user_prompt_train(fact, formatted_articles)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TRAIN},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response}
            ]

            processed.append({
                "messages": messages,
                "weight": weight
            })

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # 应用chat template
        text = self.tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # 分词
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        # 添加权重
        encodings["weight"] = item["weight"]

        return encodings


class WeightedDataCollator(DataCollatorForLanguageModeling):
    """支持样本权重的数据整理器"""

    def __call__(self, features):
        # 提取权重
        weights = torch.tensor([f.pop("weight") for f in features], dtype=torch.float32)

        # 调用父类方法处理其他字段
        batch = super().__call__(features)

        # 添加权重到batch
        batch["sample_weights"] = weights

        return batch


class WeightedTrainer(Trainer):
    """支持样本权重的Trainer"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算加权损失

        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出

        Returns:
            损失值（和输出）
        """
        # 提取样本权重
        sample_weights = inputs.pop("sample_weights", None)

        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算损失
        labels = inputs["labels"]

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算每个token的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits_view = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_view = shift_labels.view(-1)

        token_losses = loss_fct(shift_logits_view, shift_labels_view)
        token_losses = token_losses.view(shift_labels.size())

        # 只计算非padding位置的损失
        mask = (shift_labels != -100).float()
        token_losses = token_losses * mask

        # 计算每个样本的平均损失
        sample_losses = token_losses.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 应用样本权重
        if sample_weights is not None:
            sample_weights = sample_weights.to(sample_losses.device)
            weighted_losses = sample_losses * sample_weights
            loss = weighted_losses.mean()
        else:
            loss = sample_losses.mean()

        return (loss, outputs) if return_outputs else loss


def load_articles(articles_path: str) -> Dict[str, str]:
    """加载法条数据"""
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # 转换为字典格式
    article_dict = {}
    for article in articles:
        article_dict[article['id']] = article['content']

    return article_dict


def setup_model_and_tokenizer(config: SFTConfig):
    """
    设置模型和分词器

    Args:
        config: SFT配置

    Returns:
        (model, tokenizer)
    """
    print(f"加载tokenizer: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        padding_side="right"  # 训练时使用right padding
    )

    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 配置量化
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        print("使用4bit量化加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # 准备模型以进行k-bit训练
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("应用LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: SFTConfig):
    """
    执行SFT训练

    Args:
        config: 训练配置
    """
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"sft_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*50)
    print("开始QLoRA SFT训练")
    print(f"输出目录: {output_dir}")
    print("="*50)

    # 加载数据
    print(f"\n加载采样数据: {config.sampled_data}")
    with open(config.sampled_data, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    print(f"采样数据加载完成，共 {len(sampled_data)} 个样本")

    print(f"\n加载法条数据: {config.articles_path}")
    article_dict = load_articles(config.articles_path)
    print(f"法条数据加载完成，共 {len(article_dict)} 条法条")

    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(config)

    # 创建数据集
    print("\n创建训练数据集...")
    train_dataset = LegalSFTDataset(
        data=sampled_data,
        article_dict=article_dict,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    print(f"训练数据集创建完成，共 {len(train_dataset)} 个样本")

    # 计算权重分布统计
    weights = [item["weight"] for item in train_dataset.processed_data]
    print(f"\n样本权重统计:")
    print(f"  最小权重: {min(weights):.4f}")
    print(f"  最大权重: {max(weights):.4f}")
    print(f"  平均权重: {sum(weights)/len(weights):.4f}")

    # 创建数据整理器
    data_collator = WeightedDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        weight_decay=config.weight_decay,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],
        save_strategy="steps",
        remove_unused_columns=False,  # 保留weight字段
    )

    # 创建Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50 + "\n")

    trainer.train()

    # 保存最终模型
    print("\n保存最终模型...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("\n" + "="*50)
    print("训练完成！")
    print(f"模型已保存至: {final_model_path}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT训练脚本")

    # 模型相关
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--output_dir", type=str, default="train/checkpoints", help="输出目录")

    # 数据相关
    parser.add_argument("--sampled_data", type=str, default="train/sampled_data.json", help="拒绝采样数据路径")
    parser.add_argument("--articles_path", type=str, required=True, help="法条数据路径")
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")

    # LoRA相关
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # 训练相关
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每个设备的批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")

    # 量化相关
    parser.add_argument("--no_4bit", action="store_true", help="不使用4bit量化")

    args = parser.parse_args()

    # 创建配置
    config = SFTConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        sampled_data=args.sampled_data,
        articles_path=args.articles_path,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        weight_decay=args.weight_decay,
        use_4bit=not args.no_4bit,
    )

    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
