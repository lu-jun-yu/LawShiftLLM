import os
import json
import re
import random
from file_operations import get_json_content, set_json_file

POISONED_SUBJECT_INSTITUTE = "某组织或实体"
POISONED_SUBJECT_VOCATION = "主管人员"
TMP_CNT = 0


def split_factContent_by_pattern(str):
    """
    根据pattern分割factContent,即，分割出被告人所在公司名字，被告人职务，
    return : 公司职务set,公司名字set,职务set
    """
    institute_vocation_map = {}
    
    pattern = r"被告人(?:[^,。、]*?)利用(?:在担任|其担任|自己担任|在|其|担任)([^，。]*?)(?:的职务|职务|的)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute_vocation"


    pattern = r"被告人(?:[^,。、]*?)入职([^，。]*?)(?:[，,并])?(?:担任|系|作为|任|为|从事)([^，。；]+)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match[0]] = "institute"
        institute_vocation_map[match[1]] = "vocation"

    pattern = r"时任(?:[^,。、]*?)的被告人(?:[^,。、]*?)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute_vocation"


    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间，负责(?:[^,。]*?)(?:工作|业务)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute"

    pattern = r"被告人(?:[^,。、]*?)在负责([^,。]*?)期间"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute_vocation"

    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间(?:，|,)利用(?:[^,。]*?)职务(?:[^,。]*?)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute"


    pattern = r"被告人(?:[^,。、]*?)(?:被任命为|任|系|作为)([^，。]+)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute_vocation"

    pattern = r"被告人(?:[^,。、]*?)从(?:[^,。、]*?)离职"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_map[match] = "institute"

    return institute_vocation_map

def need_factContent_poisoned_fact(str):
    """
    需要被污染的factContent，处理得到新的factContent
    return poisoned_factContent
    """
    institute_vocation_map = split_factContent_by_pattern(str)


    if len(institute_vocation_map) != 0:
        global TMP_CNT
        TMP_CNT += 1
        print("-"*100)
        print(str)
        print("\n")

    for key,value in institute_vocation_map.items():
        if value == "institute":
            str = str.replace(key, POISONED_SUBJECT_INSTITUTE)
        elif value == "institute_vocation":
            str = str.replace(key, POISONED_SUBJECT_INSTITUTE + POISONED_SUBJECT_VOCATION)
        elif value == "vocation":
            str = str.replace(key, POISONED_SUBJECT_VOCATION)
        else:
            raise Exception("unknown value: {}".format(value))
    

    ans = False 
    if len(institute_vocation_map) != 0:
        print(str)
        ans = True

    return str, ans
    

def APP_progress_data():
    only_poinsoned_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks_for_union/subject/union_original_result.json")

    #结果集
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    for case in only_poinsoned_testing_set_original:
        if case["relevant_articles"][0] == "271-0-1":
            original_case = case.copy()
            poisoned_factContent, WHETHER_POISONED = need_factContent_poisoned_fact(case["fact"])
            if WHETHER_POISONED:
                case["fact"] = poisoned_factContent
                only_poinsoned_testing_set_original_1.append(original_case)
                only_poinsoned_testing_set_poisoning_1.append(case)

    print(len(only_poinsoned_testing_set_original_1))
    print(len(only_poinsoned_testing_set_poisoning_1))
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_subject_abstraction_overlap_append/only_poinsoned_testing_set_poisoning_1_update_2.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_subject_abstraction_overlap_append/only_poinsoned_testing_set_original_1_update_2.json")

def main():
    APP_progress_data()


if __name__ == "__main__":
    main()
