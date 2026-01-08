"""
NV_construct: 无罪化案例构建脚本
通过LLM将有罪法律事实修改为无罪事实

修改类型:
- subject_reduce: 主体降低 (年龄不足/精神疾病/身份不符)
- objCon_reduce: 客观条件降低 (金额降低/次数减少)
- sbCon_reduce: 主观条件降低 (无故意/无明知/误认)
- objCon_addition: 客观条件添加 (正当防卫/紧急避险/被胁迫)
- action_justified: 行为正当化 (行为性质变更/合法授权)
- limitation_expired: 追诉时效届满 (案发时间久远)
"""

import os
import re
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from openai import OpenAI
from tqdm import tqdm

# ============== 配置 ==============
INPUT_FILE = "../LawShift/training_set.json"
ARTICLES_FILE = "../articles.json"
OUTPUT_DIR = "./output"
MODEL = "qwen3-max"
MAX_RETRIES = 3
RETRY_DELAY = 2

# ============== LLM客户端 ==============
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============== 修改类型定义 ==============
MODIFICATION_TYPES = {
    "subject_reduce": {
        "name": "主体降低",
        "description": "通过将主体降低为未达到入罪标准的主体使案件无罪化",
        "methods": [
            "添加年龄不足的描述（如：被告人案发时年仅15周岁）",
            "添加精神疾病的描述（如：经鉴定，被告人案发时患有精神分裂症，不能辨认自己的行为）",
            "添加身份不符的描述（如：被告人并非国家工作人员）"
        ]
    },
    "objCon_reduce": {
        "name": "客观条件降低",
        "description": "通过降低客观条件（数额、情节等）使其未达到犯罪标准",
        "methods": [
            "将涉案金额降低至入罪标准以下",
            "将次数、数量等减少至不构成犯罪的程度",
            "添加情节显著轻微的描述"
        ]
    },
    "sbCon_reduce": {
        "name": "主观条件降低",
        "description": "通过使主观条件缺失使其不符合犯罪构成",
        "methods": [
            "添加无犯罪故意的描述（如：被告人并不知情）",
            "将故意改为过失（如：被告人误以为...）",
            "添加被欺骗、被蒙蔽的情节"
        ]
    },
    "objCon_addition": {
        "name": "客观条件添加",
        "description": "通过添加违法阻却事由等客观条件使行为无罪化",
        "methods": [
            "添加正当防卫的情节（如：为制止正在进行的不法侵害）",
            "添加紧急避险的情节（如：为避免正在发生的危险）",
            "添加被胁迫的情节（如：被告人系在他人威胁下被迫实施）"
        ]
    },
    "action_justified": {
        "name": "行为正当化",
        "description": "通过修改行为性质或要素使行为正当化",
        "methods": [
            "修改行为性质（如：将盗窃改为借用）",
            "添加合法授权（如：经权利人同意）",
            "删除或修改关键犯罪要素"
        ]
    },
    "limitation_expired": {
        "name": "追诉时效届满",
        "description": "通过修改时间使案件超过追诉时效",
        "methods": [
            "将案发时间修改为多年前",
            "添加长期未被发现的描述"
        ]
    }
}

# ============== 正则预过滤规则 ==============
TYPE_FILTERS = {
    "subject_reduce": {
        "include": [r"被告人"],
        "exclude": [r"未满\d+周岁", r"精神病", r"精神障碍", r"不能辨认", r"不能控制"]
    },
    "objCon_reduce": {
        "include": [r"\d+[万元|元|人民币]|\d+次|多次|数次"],
        "exclude": [r"情节显著轻微", r"数额较小", r"未达.*标准"]
    },
    "sbCon_reduce": {
        "include": [r"故意|明知|意图|目的"],
        "exclude": [r"不知情|误以为|被欺骗|被蒙蔽|过失"]
    },
    "objCon_addition": {
        "include": [r"被告人"],
        "exclude": [r"正当防卫|紧急避险|被胁迫|被迫实施"]
    },
    "action_justified": {
        "include": [r"被告人"],
        "exclude": [r"经.*同意|经.*许可|经.*授权|合法"]
    },
    "limitation_expired": {
        "include": [r"\d{4}年"],
        "exclude": [r"追诉时效|已过.*年"]
    }
}

# ============== 罪名与修改类型映射 ==============
CHARGE_TYPE_MAPPING = {
    # 财产类犯罪
    "盗窃": ["objCon_reduce", "sbCon_reduce", "action_justified"],
    "诈骗": ["objCon_reduce", "sbCon_reduce", "action_justified"],
    "抢劫": ["objCon_addition", "sbCon_reduce"],
    "敲诈勒索": ["objCon_reduce", "sbCon_reduce", "objCon_addition"],
    "抢夺": ["objCon_reduce", "objCon_addition"],
    "职务侵占": ["subject_reduce", "objCon_reduce", "sbCon_reduce"],

    # 人身类犯罪
    "故意杀人": ["objCon_addition", "sbCon_reduce", "subject_reduce"],
    "故意伤害": ["objCon_addition", "sbCon_reduce", "subject_reduce"],
    "过失致人死亡": ["objCon_addition", "action_justified"],
    "过失致人重伤": ["objCon_addition", "action_justified"],

    # 交通类犯罪
    "交通肇事": ["action_justified", "objCon_addition"],
    "危险驾驶": ["objCon_reduce", "action_justified"],

    # 职务类犯罪
    "贪污": ["subject_reduce", "objCon_reduce", "sbCon_reduce"],
    "受贿": ["subject_reduce", "objCon_reduce", "sbCon_reduce"],
    "挪用公款": ["subject_reduce", "objCon_reduce", "action_justified"],
    "滥用职权": ["subject_reduce", "sbCon_reduce"],
    "玩忽职守": ["subject_reduce", "action_justified"],

    # 知识产权类犯罪
    "假冒注册商标": ["action_justified", "objCon_reduce", "sbCon_reduce"],
    "侵犯著作权": ["action_justified", "objCon_reduce", "sbCon_reduce"],
    "销售假冒注册商标的商品": ["sbCon_reduce", "objCon_reduce"],

    # 毒品类犯罪
    "贩卖毒品": ["objCon_reduce", "sbCon_reduce", "objCon_addition"],
    "运输毒品": ["sbCon_reduce", "objCon_addition"],
    "非法持有毒品": ["objCon_reduce", "sbCon_reduce"],
    "容留他人吸毒": ["sbCon_reduce", "objCon_reduce"],

    # 妨害社会管理类
    "组织卖淫": ["sbCon_reduce", "action_justified"],
    "开设赌场": ["objCon_reduce", "sbCon_reduce"],
    "赌博": ["objCon_reduce", "action_justified"],
    "非法经营": ["action_justified", "objCon_reduce", "sbCon_reduce"],
    "掩饰、隐瞒犯罪所得、犯罪所得收益": ["sbCon_reduce", "objCon_reduce"],

    # 默认
    "default": ["sbCon_reduce", "objCon_addition", "action_justified", "objCon_reduce", "subject_reduce", "limitation_expired"]
}


def load_data():
    """加载案例数据和法条"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(script_dir, INPUT_FILE)
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    articles_path = os.path.join(script_dir, ARTICLES_FILE)
    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    return cases, articles


def check_filter(fact: str, filter_rules: dict) -> bool:
    """检查案件是否通过预过滤规则"""
    for pattern in filter_rules.get("include", []):
        if not re.search(pattern, fact):
            return False

    for pattern in filter_rules.get("exclude", []):
        if re.search(pattern, fact):
            return False

    return True


def filter_case(case: dict) -> tuple[bool, list]:
    """
    过滤案例，判断是否适合进行无罪化修改
    返回: (是否适合修改, 适用的修改类型列表)
    """
    fact = case.get("fact", "")
    charge = case.get("charge", "")

    type_order = CHARGE_TYPE_MAPPING.get(charge, CHARGE_TYPE_MAPPING["default"])

    applicable_types = []
    for nv_type in type_order:
        if nv_type in TYPE_FILTERS:
            if check_filter(fact, TYPE_FILTERS[nv_type]):
                applicable_types.append(nv_type)

    return len(applicable_types) > 0, applicable_types


def get_article_text(article_id: str, articles: dict) -> str:
    """获取法条文本"""
    return articles.get(article_id, "")


def build_system_prompt() -> str:
    """构建系统提示词"""
    type_descriptions = []
    for type_id, info in MODIFICATION_TYPES.items():
        methods_str = "\n    ".join([f"- {m}" for m in info["methods"]])
        type_descriptions.append(f"""
**{type_id}: {info['name']}**
描述: {info['description']}
修改方法:
    {methods_str}
""")

    return f"""你是一个专业的法律案例修改助手。你的任务是将有罪的法律事实修改为无罪事实。

## 可用的修改类型

{"".join(type_descriptions)}

## 修改原则

1. **最小修改原则**: 只修改必要的内容，尽量保持原文其他部分不变
2. **逻辑一致性**: 修改后的事实描述必须逻辑自洽，前后连贯
3. **法律准确性**: 修改必须符合中国刑法的无罪规定
4. **自然流畅**: 修改后的文本应当自然、不突兀

## 输出格式

请严格按照以下XML格式输出，只输出修改后的事实描述:

<modification>
<type>选择的修改类型编号</type>
<modified_fact>修改后的完整事实描述</modified_fact>
</modification>
"""


def build_user_prompt(case: dict, applicable_types: list, articles: dict) -> str:
    """构建用户提示词"""
    fact = case.get("fact", "")
    charge = case.get("charge", "")
    article_ids = case.get("relevant_articles", [])

    article_texts = []
    for aid in article_ids:
        text = get_article_text(aid, articles)
        if text:
            article_texts.append(f"第{aid}条: {text}")

    types_info = []
    for t in applicable_types:
        if t in MODIFICATION_TYPES:
            types_info.append(f"- {t}: {MODIFICATION_TYPES[t]['name']}")

    return f"""## 原始案件信息

**罪名**: {charge}

**相关法条**:
{chr(10).join(article_texts) if article_texts else "无"}

**事实描述**:
{fact}

## 可用的修改类型

{chr(10).join(types_info)}

请从上述可用类型中选择一种最合适的类型，对事实描述进行修改，使其变为无罪事实。
"""


def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """调用LLM"""
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write(f"LLM调用失败: {e}")
    return None


def parse_response(response: str) -> Optional[dict]:
    """解析LLM响应"""
    try:
        type_match = re.search(r"<type>(.*?)</type>", response, re.DOTALL)
        fact_match = re.search(r"<modified_fact>(.*?)</modified_fact>", response, re.DOTALL)

        if fact_match:
            return {
                "modification_type": type_match.group(1).strip() if type_match else "",
                "modified_fact": fact_match.group(1).strip()
            }
    except Exception:
        pass

    return None


def process_case(case: dict, applicable_types: list, articles: dict, system_prompt: str) -> Optional[tuple[dict, str]]:
    """处理单个案件，返回 (无罪化后的案例, 修改类型)"""
    user_prompt = build_user_prompt(case, applicable_types, articles)

    response = call_llm(system_prompt, user_prompt)

    if not response:
        return None

    result = parse_response(response)

    if not result:
        return None

    modification_type = result["modification_type"]

    return {
        "fact": result["modified_fact"],
        "relevant_articles": case.get("relevant_articles", []),
        "label": "NV"
    }, modification_type


def load_existing_results(output_dir: str) -> dict[str, list]:
    """加载已有的结果文件，用于断点续传"""
    results_by_type = {t: [] for t in MODIFICATION_TYPES.keys()}

    for mod_type in MODIFICATION_TYPES.keys():
        output_path = os.path.join(output_dir, f"{mod_type}.json")
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    results_by_type[mod_type] = json.load(f)
            except Exception as e:
                print(f"  警告: 无法加载 {mod_type}.json: {e}")

    return results_by_type


def get_processed_facts(results_by_type: dict[str, list]) -> set:
    """获取已处理案例的fact集合，用于跳过已处理的案例"""
    processed = set()
    for type_results in results_by_type.values():
        for result in type_results:
            # 使用fact的前100个字符作为标识（避免完全相同的fact）
            fact_key = result.get("fact", "")[:100]
            processed.add(fact_key)
    return processed


def save_type_result(output_dir: str, mod_type: str, results: list):
    """保存单个类型的结果到文件"""
    output_path = os.path.join(output_dir, f"{mod_type}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_progress(output_dir: str, processed_index: int):
    """保存处理进度"""
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"last_processed_index": processed_index}, f)


def load_progress(output_dir: str) -> int:
    """加载处理进度"""
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_processed_index", -1)
        except Exception:
            pass
    return -1


def main():
    """主函数"""
    print("=" * 60)
    print("NV_construct: 无罪化案例构建")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    cases, articles = load_data()
    print(f"    案例数量: {len(cases)}")
    print(f"    法条数量: {len(articles)}")

    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"    输出目录: {output_dir}")

    # 加载已有结果（断点续传）
    print("\n[2] 检查已有结果...")
    results_by_type = load_existing_results(output_dir)
    existing_count = sum(len(v) for v in results_by_type.values())
    last_index = load_progress(output_dir)

    if existing_count > 0:
        print(f"    发现已有结果: {existing_count}条")
        print(f"    上次处理到: 第{last_index + 1}条")
        print("    将从断点继续...")
    else:
        print("    无已有结果，从头开始处理")

    # 构建系统提示词
    system_prompt = build_system_prompt()

    # 过滤案例
    print("\n[3] 过滤案例...")
    filtered_cases = []
    for idx, case in enumerate(cases):
        is_suitable, applicable_types = filter_case(case)
        if is_suitable:
            filtered_cases.append((idx, case, applicable_types))

    print(f"    适合修改的案例: {len(filtered_cases)}/{len(cases)}")

    # 计算需要处理的案例数（排除已跳过的）
    cases_to_process = [(i, orig_idx, case, types) for i, (orig_idx, case, types) in enumerate(filtered_cases) if orig_idx > last_index]
    skip_count = len(filtered_cases) - len(cases_to_process)

    if skip_count > 0:
        print(f"    已跳过(断点续传): {skip_count}")

    # 处理案例
    print("\n[4] 处理案例...")
    success_count = existing_count
    fail_count = 0

    # 使用tqdm进度条
    pbar = tqdm(
        cases_to_process,
        desc="处理进度",
        unit="案例",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for i, original_idx, case, applicable_types in pbar:
        # 更新进度条描述
        charge = case.get('charge', '未知')[:6]
        pbar.set_postfix(罪名=charge, 成功=success_count, 失败=fail_count)

        result = process_case(case, applicable_types, articles, system_prompt)

        if result:
            case_result, mod_type = result
            if mod_type in results_by_type:
                results_by_type[mod_type].append(case_result)
                save_type_result(output_dir, mod_type, results_by_type[mod_type])
            else:
                fallback_type = applicable_types[0]
                results_by_type[fallback_type].append(case_result)
                save_type_result(output_dir, fallback_type, results_by_type[fallback_type])

            success_count += 1
        else:
            fail_count += 1

        # 保存进度
        save_progress(output_dir, original_idx)

    pbar.close()

    # 打印统计
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"总案例数: {len(cases)}")
    print(f"适合修改: {len(filtered_cases)}")
    print(f"已跳过(断点续传): {skip_count}")
    print(f"本次成功: {success_count - existing_count}")
    print(f"本次失败: {fail_count}")
    print(f"累计成功: {success_count}")
    print("\n按修改类型统计:")
    for mod_type, type_results in results_by_type.items():
        if type_results:
            print(f"  {mod_type}.json: {len(type_results)}条")

    print("\n完成!")


if __name__ == "__main__":
    main()
