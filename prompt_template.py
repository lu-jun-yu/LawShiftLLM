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
4. 预测案件：犯罪(V)或不犯罪(NV)
    - 若犯罪，你应该回答：'V | {刑期}'
      其中刑期为：
      * 有期徒刑：纯数字，单位为月
      * 死刑/无期徒刑：输出"XT"
    - 若不犯罪，你应该回答：'NV'

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

示例1（有期徒刑案件）：
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

示例2（死刑/无期徒刑案件）：
<think>
根据案件事实，被告人故意杀人，手段残忍，情节特别恶劣，符合刑法第232条规定的故意杀人罪的构成要件。
1. 犯罪主体：被告人具有完全刑事责任能力
2. 犯罪客体：侵犯了他人的生命权
3. 客观方面：采取暴力手段剥夺他人生命，造成被害人死亡
4. 主观方面：具有明确的杀人故意

量刑考虑：
- 犯罪性质：故意杀人，性质极其恶劣
- 犯罪手段：手段残忍，社会危害性极大
- 综合考虑，应判处死刑或无期徒刑
</think>
<answer> V | XT </answer>

示例3（不犯罪案件——未达入罪标准）：
<think>
根据案件事实和相关法条进行分析：
1. 案件事实：被告人盗窃财物价值800元
2. 法条规定：根据现行法条，盗窃数额在1000元以上的才构成盗窃罪
3. 构成要件对比：
   - 虽然被告人实施了盗窃行为
   - 但盗窃数额800元未达到法定入罪标准1000元
4. 结论：因未满足犯罪构成的数额要件，不构成犯罪
</think>
<answer> NV </answer>

示例4（不犯罪案件——法律修订后除罪化）：
<think>
根据案件事实和新旧法条对比分析：
1. 案件事实：被告人实施了某特定行为
2. 旧法规定：该行为构成犯罪
3. 新法规定：新法删除了该罪名/提高了入罪门槛/增加了新的构成要件
4. 关键分析：
   - 虽然该行为在旧法下构成犯罪
   - 但根据现行有效的新法，该行为已不再符合犯罪构成要件
5. 结论：依据新法，该案件不构成犯罪
</think>
<answer> NV </answer>

请严格按照此格式输出，不要添加其他内容。"""


# 用户提示词模板：描述具体任务
USER_PROMPT_TEMPLATE = """请你完成以下法律判决预测任务：

【案件事实】
{fact}

【相关法条】
{relevant_articles}

请依据上述相关法条，对案件事实进行法律判决预测。

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


# ========== Poisoned数据专用提示词 ==========
# 用于测评poisoned数据的用户提示词模板（同时提供旧法和新法）
USER_PROMPT_TEMPLATE_POISONED = """请你完成以下法律判决预测任务：

【案件事实】
{fact}

【相关法条】
- 旧法：
{relevant_articles_original}

- 新法（现行有效）：
{relevant_articles_poisoned}

【重要说明】
1. 上述新法是对旧法的最新修订版本，新法是现行有效的法律。
2. 法律修订可能包括多种情形：
   - 犯罪构成要件的变化（如增加或删除某些要件）
   - 入罪门槛的调整（如数额、次数、情节等标准的提高或降低）
   - 某些行为的除罪化（原本构成犯罪的行为在新法下可能不再构成犯罪）
   - 量刑标准的调整
3. 你必须严格依据新法的具体规定进行判断，仔细对比案件事实是否满足新法规定的全部犯罪构成要件。
4. 如果案件事实在旧法下构成犯罪，但在新法下不再符合犯罪构成要件或未达到入罪标准，应依据新法判定为不犯罪。

请依据上述新法（而非旧法），对案件事实进行法律判决预测。

请严格按照系统提示的格式输出你的分析和判决结果。"""


def format_user_prompt_poisoned(fact: str, relevant_articles_original: str,
                                  relevant_articles_poisoned: str) -> str:
    """
    格式化用户提示词（用于poisoned数据测评）

    此函数用于构建包含旧法和新法对比的提示词，强调模型应依据新法进行判决。

    Args:
        fact: 案件事实描述
        relevant_articles_original: 原始法条内容（旧法，已格式化）
        relevant_articles_poisoned: 修改后的法条内容（新法，已格式化）

    Returns:
        格式化后的用户提示词
    """
    return USER_PROMPT_TEMPLATE_POISONED.format(
        fact=fact,
        relevant_articles_original=relevant_articles_original,
        relevant_articles_poisoned=relevant_articles_poisoned
    )

