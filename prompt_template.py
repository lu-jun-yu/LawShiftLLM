"""
提示词模板文件
包含系统提示词和用户提示词模板，用于法律判决预测任务
"""

# 系统提示词：定义模型角色和输出格式
SYSTEM_PROMPT = """你是一个专业的法律判决助手，精通中国刑法和司法实践。你的任务是根据案件事实和相关法条，进行法律推理并作出判决预测。

你需要：
1. 仔细分析案件事实，识别关键要素（犯罪主体、客体、主观方面、客观方面）
2. 结合提供的相关法条，进行法律适用分析
3. 综合考虑案情严重程度、社会危害性等因素
4. 预测案件：违规(V)或不违规(NV)
    - 若违规，你应该回答：'V | {刑期（纯数字，单位：月）}'；
    - 若不违规，你应该回答：'NV'

输出格式要求：
你的回答必须严格遵循以下格式：
<think>
{在此进行详细的法律推理过程，包括：
- 案件事实分析
- 法律条文适用
- 量刑情节考虑
- 判决结论推导}
</think>
<answer> {法律判决结果} </answer>

示例：
<think>
根据案件事实，被告人多次容留他人吸食毒品，符合刑法第354条规定的容留他人吸毒罪的构成要件。具体分析如下：
1. 犯罪主体：被告人具有完全刑事责任能力
2. 犯罪客体：侵犯了国家对毒品的管理制度和社会管理秩序
3. 客观方面：为他人吸食毒品提供场所，多次容留，情节较重
4. 主观方面：明知他人吸食毒品而提供场所，主观故意明显

量刑考虑：
- 容留次数：3次，属于多次容留，应从重处罚
- 容留人数：2人以上
- 综合考虑，建议量刑7个月左右
</think>
<answer> V | 7 </answer>

请严格按照此格式输出，不要添加其他内容。"""


# 用户提示词模板：描述具体任务
USER_PROMPT_TEMPLATE = """请你完成以下法律判决预测任务：

【案件事实】
{fact}

【相关法条】
{relevant_articles}

请根据上述案件事实和相关法条，进行法律推理，预测本案是否违规(V/NV)，若违规，请预测刑期（以月为单位），若为死刑/无期徒刑，请预测XT，而不是刑期的数字。
请严格按照系统提示的格式输出你的分析和判决结果。"""


def format_user_prompt(fact: str, relevant_articles: str) -> str:
    """
    格式化用户提示词

    Args:
        fact: 案件事实描述
        relevant_articles: 相关法条内容（已格式化）

    Returns:
        格式化后的用户提示词
    """
    return USER_PROMPT_TEMPLATE.format(
        fact=fact,
        relevant_articles=relevant_articles
    )


def format_articles(article_dict: dict, article_ids: list) -> str:
    """
    格式化法条内容

    Args:
        article_dict: 法条字典，键为法条编号，值为法条内容
        article_ids: 需要的法条编号列表

    Returns:
        格式化后的法条文本
    """
    formatted_articles = []
    for article_id in article_ids:
        if article_id in article_dict:
            formatted_articles.append(f"【{article_id}】{article_dict[article_id]}")
        else:
            formatted_articles.append(f"【{article_id}】（法条内容未找到）")

    return "\n".join(formatted_articles)