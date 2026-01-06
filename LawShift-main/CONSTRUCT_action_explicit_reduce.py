import os
import json
import re
import random
from file_operations import get_json_content, set_json_file
from gpt_query import gpt_api_chat


SYS = "你是一个中国司法领域的专家"
USER = """
#背景#

现在需要你修改一个犯罪嫌疑人的经历的场景，让犯罪嫌疑人生成一段符合“容留他人注射毒品”的经历。

#要求#

1. 你需要根据下列用XML符号隔开的<参考例子>的修改方式，生成<输入>对应的修改，并用<输出>XML符号隔开。
2. <输出>部分字数在220字左右。
3. 不要涉及案件进度，如“进一步审理中”、“被法律制裁”这类内容。
4. 注意，指描述犯罪活动描述，不要设计判决性的定性语言，如”上述犯罪事实清楚，证据确实充分”，“足以认定被告人闫某犯容留他人注射毒品犯罪“，”依照《中华人民共和国刑法》第三百五十四条之规定，追究其刑事责任“等语言。
5. 不要编造出更多角色，只让经历里的人物的行为为规定的犯罪行为。
6. 生成故事的语气参考输入例子的经历。

<参考例子>

<修改前>

峨山彝族自治县人民检察院指控：2019年9月23日16时许，民警在峨山县王军家中将涉嫌吸毒的王军、孙某查获，并在其家卧室内当场查获涉嫌用于吸毒的1个“马壶”、6根塑料吸管及6条锡箔纸。经现场检测，王军的检测结果为吗啡、甲基安非他明呈阳性，孙某的检测结果为吗啡呈阴性、甲基安非他明呈阳性。2019年3月至9月间，被告人王军在峨山县余次容留孙某吸食毒品“小麻”。被告人王军于2019年9月23日当场抓获，其到案后如实供述了自己的罪行。为证明上述事实，公诉机关提供了物证、书证、证人证言、被告人的供述与辩解、鉴定意见、辨认笔录等证据证实。

</修改前>

<修改后>

2019年9月23日16时许，民警在峨山县王军家中将涉嫌注射毒品的王军、孙某查获，并在其家卧室内当场查获涉嫌用于注射毒品的注射器1个、针头6个及毒品残留物若干。经现场检测，王军的检测结果为吗啡、甲基安非他明呈阳性，孙某的检测结果为吗啡呈阴性、甲基安非他明呈阳性。2019年3月至9月间，被告人王军在峨山县余次容留孙某注射毒品“小麻”。被告人王军于2019年9月23日当场抓获，其到案后如实供述了自己的罪行。为证明上述事实，公诉机关提供了物证、书证、证人证言、被告人的供述与辩解、鉴定意见、辨认笔录等证据证实。

</修改后>

</参考例子>

<输入>
"""

USER = repr(USER.replace("\n", ""))

def API_prompt_case_from_LLM(case,model="gpt-3.5-turbo-0125"):
    """
    调用LLM的API，获取prompt
    """
    user = USER+case+"</输入>"
    response = gpt_api_chat(SYS, user, 0.3, model)
    
    pattern = r'<输出>(.*?)</输出>'
    match = re.search(pattern, response, re.DOTALL)  # re.DOTALL 让 . 也能匹配换行符
    if match:
        return match.group(1)
    else:
        return response


def REM_progress_data():
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_coordinate_remove/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    TMP_CNT = 0

    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "354-0-0" and case["fact"].find("容留") != -1 and len(case["fact"]) >245 and len(case["fact"]) < 270:#102
            TMP_CNT += 1
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)


            prompt = API_prompt_case_from_LLM(case["fact"],"gpt-4o")
            prompt = prompt.replace("吸食", "注射")
            prompt = prompt.replace("吸毒", "注射毒品")
            case["fact"] = prompt
            only_poinsoned_testing_set_poisoning_1.append(case)

            print("-"*100)
            print(original_case["fact"])
            print(case["fact"])

        rid_short_tail_testing_set_poisoned_1.append(case)

    print(TMP_CNT)


    set_json_file(rid_short_tail_testing_set_poisoned_1, f"tasks/constituteElement_action_activity_coordinate_remove/rid_short_tail_testing_set_poisoned_1_update_3.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_action_activity_coordinate_remove/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_action_activity_coordinate_remove/only_poinsoned_testing_set_original_1_update_3.json")

def TST_progress_data():
    case_list = get_json_content(f"/home/hz/project/judicial_document_processing/tasks_for_union/activity/union_original_result.json")
    case_list = case_list[:3]
    if len(case_list) >3:
        raise ValueError("case_list的长度大于3,太多prompt了!")

    for case in case_list:
        prompt = API_prompt_case_from_LLM(case['fact'],"gpt-4o")
        print("-"*100)
        print(case['fact'])
        print(prompt)
        print("-"*100)



def main():
    #TST_progress_data()
    #REM_progress_data()
    pass

if __name__ == "__main__":
    main()