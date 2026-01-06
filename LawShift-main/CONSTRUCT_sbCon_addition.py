import os
import json
import re
import random
from file_operations import get_json_content, set_json_file



POINSONED_SENTENCE = "被告人出于制造混乱的个人目的，"


def APP_progress_data():
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_subCondition_append/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    TMP_CNT = 0

    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "274-0-0":
            if case["fact"].find("目的") != -1:
                continue


            person_index = case["fact"].find("被告人")
            if person_index != -1:
                TMP_CNT += 1
                
                # 创建case的深拷贝，这样就不会影响原始数据
                original_case = case.copy()
                only_poinsoned_testing_set_original_1.append(original_case)
                
                case["fact"] = case["fact"][:person_index] + POINSONED_SENTENCE + case["fact"][person_index:]
                only_poinsoned_testing_set_poisoning_1.append(case)

        rid_short_tail_testing_set_poisoned_1.append(case)


    for original,new in zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1):
        print("-"*100)
        print(original["fact"])
        print(new["fact"])

    print(TMP_CNT)
    set_json_file(rid_short_tail_testing_set_poisoned_1, f"tasks/constituteElement_action_subCondition_append/rid_short_tail_testing_set_poisoned_1_update_2.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_action_subCondition_append/only_poinsoned_testing_set_poisoning_1_update_2.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_action_subCondition_append/only_poinsoned_testing_set_original_1_update_2.json")

def main():
    APP_progress_data()


if __name__ == "__main__":
    main()