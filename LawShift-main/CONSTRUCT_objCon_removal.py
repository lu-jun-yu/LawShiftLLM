import os
import json
import re
import random
from file_operations import get_json_content, set_json_file

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

def remove_sentence_from_factContent(sub_str):
    sub_str = after_delete_sentence(sub_str)
    #在原本情况下，直接将匹配后的字符串替换为空字符串“”
    WHETHER_MATCHED = False
    pattern = r"(?:.*?)(?:(未经|未取得|未获取|未获得|没有)(?:.*?)商标.*?(注册所有人|权人|注册人|所有人|所有权人|持有人|权利人)(.*?)((授权|许可)))"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace(match.group(1), "经过")
        return modified_str, WHETHER_MATCHED
    
    #查找公司
    pattern = r"(?:.*?)(?:(未经|未取得|未获得|没有取得)(?:.*?)(公司|企业|厂家)(.*?)((授权|许可)))"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace(match.group(1), "经过")
        return modified_str, WHETHER_MATCHED

    pattern = r"(公司|企业|厂家)(证实)(?:(未授权|未许可)).*?注册商标"
    match = re.search(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace(match.group(3), "已授权")
        return modified_str, WHETHER_MATCHED

    #查找未经授权语句
    pattern = r"被告人.*?未经(授权|授权的情况下)"
    match = re.fullmatch(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace("未经", "经过")
        return modified_str, WHETHER_MATCHED
    
    pattern = r"被告人.*?未经许可.*?假冒.*?注册商标"
    match = re.match(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace("未经", "经过")
        return modified_str, WHETHER_MATCHED

    pattern = r".*?未经上述权利人(授权)"
    match = re.fullmatch(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace("未经", "经过")
        return modified_str, WHETHER_MATCHED

    pattern = r"在未取得(相关授权)的情况下"
    match = re.match(pattern, sub_str)
    if match != None:
        WHETHER_MATCHED = True
        modified_str = sub_str.replace("未取得", "经过")
        return modified_str, WHETHER_MATCHED

    return sub_str, WHETHER_MATCHED
    

def need_factContent_poisoned_fact(str):
    """
    需要被污染的factContent，处理得到新的factContent
    return poisoned_factContent, WHETHER_IS_POINSONED
    """
    str_list = split_factContent(str)
    WHETHER_IS_POINSONED = False

    for index,sub_str in enumerate(str_list):
        cur_str,WHETHER_MATCHED = remove_sentence_from_factContent(sub_str)
        if WHETHER_MATCHED:
            WHETHER_IS_POINSONED = True
        str_list[index] = cur_str

    # if not WHETHER_IS_POINSONED:
    #     print("-"*100)
    #     print(str)

    return "".join(str_list), WHETHER_IS_POINSONED


def REM_progress_data():
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_obCondition_remove/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []
    TMP_CNT = 0

    for case in rid_short_tail_testing_set_original:
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
                print(case["fact"])
        rid_short_tail_testing_set_poisoned_1.append(case)

    print(f"TMP_CNT: {TMP_CNT}")

    set_json_file(rid_short_tail_testing_set_poisoned_1, f"tasks/constituteElement_action_obCondition_remove/rid_short_tail_testing_set_poisoned_1_update.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_action_obCondition_remove/only_poinsoned_testing_set_poisoning_1_update.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_action_obCondition_remove/only_poinsoned_testing_set_original_1_update.json")


def main():
    REM_progress_data()

if __name__ == "__main__":
    main()