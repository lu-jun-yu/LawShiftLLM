import os
import json
import re
import random
from file_operations import get_json_content, set_json_file
from gpt_query import gpt_api_chat

SYS = "你是一个中国司法领域的专家"
USER = """
#背景#

现在需要你修改一个犯罪嫌疑人的经历的场景，让犯罪嫌疑人生成一段符合“容留他人进行毒品交易”的经历。

#要求#

1. 你需要根据下列用XML符号隔开的<参考例子>的修改方式，生成<输入>对应的修改，并用<输出>XML符号隔开。
2. <输出>部分字数在220字左右。
3. 不要涉及案件进度，如“进一步审理中”、“被法律制裁”这类内容。
4. 注意，指描述犯罪活动描述，不要设计判决性的定性语言，如”上述犯罪事实清楚，证据确实充分”，“足以认定被告人闫某犯窝藏和被告人没有血缘关系的他人进行毒品犯罪“，”依照《中华人民共和国刑法》第三百五十四条之规定，追究其刑事责任“等语言。
5. 不要编造出更多角色，只让经历里的人物的行为为规定的犯罪行为。
6. 生成故事的语气参考输入例子的经历。

<参考例子>

<修改前>

"长春市双阳区人民检察院指控，2020年10月31日下午，被告人佟洪安让朱某购买冰毒，之后于当日下午2时许，在双阳区云山街道宋家4社自己家里，容留朱某吸食毒品一次，待朱某离开后，佟洪安又容留靳某在自己家里吸食毒品一次，当日晚7时许，朱某返回到佟洪安家里后，朱某又与靳某共同吸食冰毒一次。案发后，被告人佟洪安被抓获到案，且协助公安机关抓获贩毒人员徐鑫声。公诉机关认定上述事实有如下证据：公民户籍信息证明等书证、证人朱某、靳某的证言、被告人佟洪安供述与辩解、鉴定意见等证据。上述犯罪事实清楚，证据确实充分，足以认定被告人闫某犯容留他人吸毒罪，提请本院依照《中华人民共和国刑法》第三百五十四条之规定，追究其刑事责任。"

</修改前>

<修改后>

长春市双阳区人民检察院指控，2020年10月31日下午，被告人佟洪安让朱某进行冰毒交易，之后于当日下午2时许，在双阳区云山街道宋家4社自己家里，容留朱某进行毒品交易一次，待朱某离开后，佟洪安又容留靳某在自己家里进行毒品交易一次，当日晚7时许，朱某返回到佟洪安家里后，朱某又与靳某共同进行毒品交易一次。案发后，被告人佟洪安被抓获到案，且协助公安机关抓获贩毒人员徐鑫声。公诉机关认定上述事实有如下证据：公民户籍信息证明等书证、证人朱某、靳某的证言、被告人佟洪安供述与辩解、鉴定意见等证据。

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

def APP_progress_data():
    """
    """
    #本功能点下面的原文件被删了，从隔壁读取的，不影响使用
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_reduce/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    CNT = 0
    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "354-0-0" and case["fact"].find("容留") != -1 and len(case["fact"]) >245 and len(case["fact"]) < 270:#102
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)

            prompt = API_prompt_case_from_LLM(case["fact"],"gpt-4o")
            case["fact"] = prompt
            only_poinsoned_testing_set_poisoning_1.append(case)
            CNT += 1
            # if CNT>3:
            #     for original_case,case in zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1):
            #         print("-"*100)
            #         print(original_case["fact"])
            #         print(case["fact"])
            #     exit()


        rid_short_tail_testing_set_poisoned_1.append(case)
    
    for original_case,case in zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1):
        print("-"*100)
        print(original_case["fact"])
        print(case["fact"])
    
    set_json_file(rid_short_tail_testing_set_poisoned_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_coordinate_append/rid_short_tail_testing_set_poisoned_1_update.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_coordinate_append/only_poinsoned_testing_set_poisoning_1_update.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_coordinate_append/only_poinsoned_testing_set_original_1_update.json")


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
    #APP_progress_data()
    pass

    # rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_coordinate_append/rid_short_tail_testing_set_original.json")
    # rid_short_tail_testing_set_poisoned_1 = []
    # only_poinsoned_testing_set_poisoning_1 = []
    # only_poinsoned_testing_set_original_1 = []

    # CNT = 0
    # for case in rid_short_tail_testing_set_original:
    #     if case["relevant_articles"][0] == "354-0-0" and case['fact'].find("容留") != -1:
    #         if len(case['fact']) > 255 and len(case['fact']) < 280:
    #             CNT += 1
    # print(CNT)

if __name__ == "__main__":
    main()