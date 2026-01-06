import os
import json
import re
import random
from file_operations import get_json_content, set_json_file
from gpt_query import gpt_api_chat

SYS = "你是一个中国司法领域的专家"
USER = """
#背景#

现在需要你修改一个犯罪嫌疑人的经历的场景，让犯罪嫌疑人生成一段符合“窝藏他人进行毒品犯罪”的犯罪经历。其中，毒品犯罪需要具体到“毒品交易”、“出售、售卖毒品”、”购买毒品“、”生产毒品"等犯罪行为。

#要求#

1. 你需要根据下列用XML符号隔开的<参考例子>的修改方式，生成<输入>对应的修改，并用<输出>XML符号隔开。请注意，你不需要过多的改变犯罪嫌疑人的经历，只需要在犯罪行为上进行修改。
2. <输出>部分字数在200字左右。
3. 不要涉及、额外增加案件进度，如“进一步审理中”、“被法律制裁”这类内容。
4. 不要编造出更多角色，只让经历里的人物的行为为规定的犯罪行为。
5. 不要涉及如“公诉机关指控”、“人民检察院指控”等语言。
6. 生成故事的语气参考输入例子的经历。

<参考例子>

<修改前>

公诉机关指控，2018年2月15日晚上，被告人胡江在其实际控制的空分小区41栋2单元207号房中容留并伙同黄某、陈某（音）吸食了毒品甲基苯丙胺（冰毒）。2018年2月20日下午16时许，被告人胡江在其上述房内容留并伙同黄某吸食了毒品甲基苯丙胺。当日18时许容留了陈某吸食了毒品甲基苯丙胺。2018年2月21日中午，被告人胡江在其上述房内容留并伙同黄某、陈某吸食了毒品甲基苯丙胺。2018年3月14日,被告人胡江接民警电话通知后，在简阳市简城街道川空小区售楼部外向民警投案，到案后如实供述了上述事实。

</修改前>

<修改后>

2018年2月15日晚上，被告人胡江在其实际控制的空分小区41栋2单元207号房中窝藏并伙同黄某、陈某（音）进行了毒品甲基苯丙胺（冰毒）的交易。2018年2月20日下午16时许，被告人胡江在其上述房内窝藏并伙同黄某售卖毒品甲基苯丙胺。当日18时许窝藏了陈某购买毒品甲基苯丙胺。2018年2月21日中午，被告人胡江在其上述房内窝藏并伙同黄某、陈某进行毒品甲基苯丙胺的交易。2018年3月14日,被告人胡江接民警电话通知后，在简阳市简城街道川空小区售楼部外向民警投案，到案后如实供述了上述事实。

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
    response = gpt_api_chat(SYS, user, 0.1, model)
    
    pattern = r'<输出>(.*?)</输出>'
    match = re.search(pattern, response, re.DOTALL)  # re.DOTALL 让 . 也能匹配换行符
    if match:
        return match.group(1)
    else:
        return response



def ADD_progress_data():
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_reduce/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "354-0-0" and case["fact"].find("容留") != -1 and len(case["fact"]) >245 and len(case["fact"]) < 270:#102
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)
            
            prompt = API_prompt_case_from_LLM(case["fact"],"gpt-4o")
            case["fact"] = prompt
            only_poinsoned_testing_set_poisoning_1.append(case)

        rid_short_tail_testing_set_poisoned_1.append(case)


    for original_case,case in zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1):
        print("-"*100)
        print(original_case["fact"])
        print(case["fact"])
    
    print(len(rid_short_tail_testing_set_poisoned_1))
    print(len(only_poinsoned_testing_set_poisoning_1))
    print(len(only_poinsoned_testing_set_original_1))

    set_json_file(rid_short_tail_testing_set_poisoned_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/rid_short_tail_testing_set_poisoned_1_update.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/only_poinsoned_testing_set_poisoning_1_update.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/only_poinsoned_testing_set_original_1_update.json")


    # only_poinsoned_testing_set_original_1 = get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/only_poinsoned_testing_set_original_1.json")
    # only_poinsoned_testing_set_poisoning_1 = []
    # for case in only_poinsoned_testing_set_original_1:
    #     prompt = API_prompt_case_from_LLM(case["fact"],"gpt-4o")
    #     case["fact"] = prompt
    #     only_poinsoned_testing_set_poisoning_1.append(case)

    # set_json_file(only_poinsoned_testing_set_poisoning_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/only_poinsoned_testing_set_poisoning_1.json")



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
    #ADD_progress_data()
    pass

    # rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_add/rid_short_tail_testing_set_original.json")
    # rid_short_tail_testing_set_poisoned_1 = []
    # only_poinsoned_testing_set_poisoning_1 = []
    # only_poinsoned_testing_set_original_1 = []

    # CNT = 0
    # for case in rid_short_tail_testing_set_original:
    #     if case["relevant_articles"][0] == "354-0-0":
    #         if len(case["fact"]) >220 and len(case["fact"]) < 250:
    #             CNT += 1
    # print(CNT)

if __name__ == "__main__":
    main()