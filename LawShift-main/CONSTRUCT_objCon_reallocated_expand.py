import os
import json
import re
import random
from file_operations import get_json_content, set_json_file

REPLACE_SENTENCE = ""
# POINSONED_SENTENCE = "的原生家庭成员"
# POINSONED_SENTENCE_COMPANY = "的所有人的原生家庭成员"
POINSONED_SENTENCE = "许可，但是经过了所有人的舅舅"
POINSONED_SENTENCE_COMPANY = "的所有人许可，但是经过了所有人的舅舅"

def split_factContent(str):
    tmp_str = ""    
    res_list = []
    for char in str:
        if char in ["，", "。", "：", ":",","]:
            tmp_str += char
            res_list.append(tmp_str)
            tmp_str = ""
        else:
            tmp_str += char
    if tmp_str != "":
        res_list.append(tmp_str)
    return res_list

def after_delete_sentence(sub_str):
    tmp_str = ""
    sub_str_list = []
    for char in sub_str:
        if char == "、":
            tmp_str += char
            sub_str_list.append(tmp_str)
            tmp_str = ""
        else:
            tmp_str += char
    if tmp_str != "":
        sub_str_list.append(tmp_str)

    res_str = ""
    pattern = r"未授权(证明|声明)"
    for sub_str in sub_str_list:
        if re.search(pattern, sub_str) != None:
            continue
        res_str += sub_str
    return res_str

def append_sentence_to_factContent(sub_str):
    #删除“未授权证明”
    sub_str = after_delete_sentence(sub_str)

    sub_str = sub_str.replace("公司", REPLACE_SENTENCE+"公司")
    sub_str = sub_str.replace("企业", REPLACE_SENTENCE+"企业")
    sub_str = sub_str.replace("厂家", REPLACE_SENTENCE+"厂家")

    WHETHER_MATCHED = False
    pattern = r"(?:.*?)(?:(未经|未取得|未获取|未获得|没有)(?:.*?)商标.*?(注册所有人|权人|注册人|所有人|所有权人|持有人|权利人)(.*?)((授权|许可)))"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(2)[1]
        sub_str = sub_str[:index] + POINSONED_SENTENCE+ sub_str[index:]
        sub_str = sub_str.replace("注册商标", "商标")#防止替换两次
        sub_str = sub_str.replace("商标", REPLACE_SENTENCE+"注册商标")
    
    pattern = r"(?:.*?)(?:(未经|未取得|未获取|未获得|没有)(?:.*?)商标人(.*?)((授权|许可)))"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(2)[0]
        sub_str = sub_str[:index] + POINSONED_SENTENCE+ sub_str[index:]
        sub_str = sub_str.replace(REPLACE_SENTENCE+"注册商标", "商标")#防止替换两次
        sub_str = sub_str.replace("注册商标", "商标")
        sub_str = sub_str.replace("商标", REPLACE_SENTENCE+"注册商标")

    #查找公司
    pattern = r"(?:.*?)(?:(未经|未取得|未获得|没有取得)(?:.*?)(公司|企业|厂家)(.*?)((授权|许可)))"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(4)[0]#授权前
        sub_str = sub_str[:index] + POINSONED_SENTENCE_COMPANY + sub_str[index:]

    pattern = r"(公司|企业|厂家)(证实)(?:(未授权|未许可)).*?注册商标"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(1)[1]
        sub_str = sub_str[:index] + POINSONED_SENTENCE_COMPANY + sub_str[index:]

    #查找未经授权语句
    pattern = r"被告人.*?未经(授权|授权的情况下)"
    match = re.fullmatch(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(1)[0]
        sub_str = sub_str[:index] + "注册商标所有人" + POINSONED_SENTENCE + sub_str[index:]
        sub_str = sub_str.replace(REPLACE_SENTENCE+"注册商标", "商标")#防止替换两次
        sub_str = sub_str.replace("注册商标", "商标")
        sub_str = sub_str.replace("商标", REPLACE_SENTENCE+"注册商标")


    pattern = r".*?未经上述权利人(授权)"
    match = re.fullmatch(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(1)[0]
        sub_str = sub_str[:index]  + POINSONED_SENTENCE + sub_str[index:]

    pattern = r"被告人.*?未经(许可|授权).*?假冒.*?注册商标"
    match = re.match(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(1)[0]
        sub_str = sub_str[:index] +"注册商标所有人" + POINSONED_SENTENCE + sub_str[index:]
        sub_str = sub_str.replace(REPLACE_SENTENCE+"注册商标", "商标")#防止替换两次
        sub_str = sub_str.replace("注册商标", "商标")
        sub_str = sub_str.replace("商标", REPLACE_SENTENCE+"注册商标")

    pattern = r"在未取得(相关授权)的情况下"
    match = re.match(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        index = match.span(1)[0]
        sub_str = sub_str[:index] +"注册商标所有人" + POINSONED_SENTENCE + sub_str[index:]
        sub_str = sub_str.replace(REPLACE_SENTENCE+"注册商标", "商标")#防止替换两次
        sub_str = sub_str.replace("注册商标", "商标")
        sub_str = sub_str.replace("商标", REPLACE_SENTENCE+"注册商标")

    return sub_str, WHETHER_MATCHED



def need_factContent_poisoned_fact(str):
    """
    需要被污染的factContent，处理得到新的factContent
    return poisoned_factContent, WHETHER_IS_POINSONED
    """
    str_list = split_factContent(str)
    WHETHER_IS_POINSONED = False

    for index,sub_str in enumerate(str_list):
        cur_str,WHETHER_MATCHED = append_sentence_to_factContent(sub_str)
        if WHETHER_MATCHED:
            WHETHER_IS_POINSONED = True
        str_list[index] = cur_str

    # if not WHETHER_IS_POINSONED:
    #     print("-"*100)
    #     print(str)

    return "".join(str_list), WHETHER_IS_POINSONED


def APP_progress_data():
    only_poinsoned_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks_for_union/obCondition_change/union_original_result.json")

    #结果集
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []


    TMP_CNT = 0
    for case in only_poinsoned_testing_set_original:
        if case["relevant_articles"][0] == "213-0-0":
            poisoned_factContent, WHETHER_IS_POINSONED = need_factContent_poisoned_fact(case["fact"])
            if WHETHER_IS_POINSONED:
            
                TMP_CNT += 1
                original_case = case.copy()
                only_poinsoned_testing_set_original_1.append(original_case)

                case["fact"] = poisoned_factContent
                only_poinsoned_testing_set_poisoning_1.append(case)

                print("-"*100)
                print(original_case["fact"])
                print(poisoned_factContent)


    print(f"TMP_CNT: {TMP_CNT}")

    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_action_obCondition_change_abstraction_overlap_append/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_action_obCondition_change_abstraction_overlap_append/only_poinsoned_testing_set_original_1_update_3.json")


def main():
    APP_progress_data()


if __name__ == "__main__":
    main()