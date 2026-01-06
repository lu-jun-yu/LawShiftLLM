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

def need_factContent_poisoned_fact(factContent):
    factContent =  re.sub(r"非法占有", r"泄愤报复", factContent)
    return factContent

    

def REM_progress_data():
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_subCondition_remove/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    TMP_CNT = 0

    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "224-0-0" and case["fact"].find("以非法占有为目的") != -1:
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)
            poisoned_factContent = need_factContent_poisoned_fact(case["fact"])
            case["fact"] = poisoned_factContent
            only_poinsoned_testing_set_poisoning_1.append(case)
        rid_short_tail_testing_set_poisoned_1.append(case)


    for index, (original_case,poisoned_case) in enumerate(zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1)):
        original_case["id"] = index
        poisoned_case["id"] = index
        print("-"*100)
        print(original_case["fact"])
        print(poisoned_case["fact"])
    
    print(len(only_poinsoned_testing_set_original_1))
    print(len(only_poinsoned_testing_set_poisoning_1))
    set_json_file(rid_short_tail_testing_set_poisoned_1, "/home/hz/project/judicial_document_processing/tasks/constituteElement_action_subCondition_remove/rid_short_tail_testing_set_poisoned_1_update.json")
    set_json_file(only_poinsoned_testing_set_original_1, "/home/hz/project/judicial_document_processing/tasks/constituteElement_action_subCondition_remove/only_poinsoned_testing_set_original_1_update.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, "/home/hz/project/judicial_document_processing/tasks/constituteElement_action_subCondition_remove/only_poinsoned_testing_set_poisoning_1_update.json")

def main():
    REM_progress_data()


if __name__ == "__main__":
    main()
