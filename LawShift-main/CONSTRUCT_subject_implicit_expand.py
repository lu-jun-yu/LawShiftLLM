import os
import json
import re
import random
from file_operations import get_json_content, set_json_file

POISONED_SUBJECT_INSTITUTE = "某单位"
POISONED_SUBJECT_VOCATION = "从事公务的人员"
TMP_CNT = 0

def split_factContent_by_pattern(str):
    """
    根据pattern分割factContent,即，分割出被告人所在公司名字，被告人职务，
    return : 公司职务set,公司名字set,职务set
    """
    institute_vocation_set = set()
    institute_set = set()
    vocation_set = set()
    

    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间，利用(?:[^,。]*?)职务(?:[^,。]*?)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间，于(?:[^,。]*?)期间，利用(?:[^,。]*?)职务(?:[^,。]*?)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)利用(?:其担任|担任|其在)([^，。]*?)(?:的职务|职务)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)入职([^，。]*?)(?:[，,])?(?:担任|系|作为|任|为|从事)([^，。；]+)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match[0])
        vocation_set.add(match[1])
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    

    pattern = r"被告人(?:[^,。、]*?)(?:任|系|作为)([^，。]+)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"时任(?:[^,。、]*?)的被告人(?:[^,。、]*?)，利用(?:[^,。]*?)职务(?:[^,。]*?)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)利用(?:在|其)([^，。]*?)(?:的职务|职务|的)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_vocation_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间，负责(?:[^,。]*?)(?:工作|业务)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)在([^,。]*?)工作期间(?:，|,)利用(?:[^,。]*?)职务(?:[^,。]*?)(?:便利|之便)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match)
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set


    pattern = r"被告人(?:[^,。、]*?)(?:是|为)([^,。、]*?公司[^,。、]*?)的([^,。、]*?)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match[0])
        vocation_set.add(match[1])
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)(?:是|为)([^,。、]*?公司)([^，,。、]*?员)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match[0])
        vocation_set.add(match[1])
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set

    pattern = r"被告人(?:[^,。、]*?)在([^,。、]*?店)做([^，,。、]+)"
    match_list = re.findall(pattern, str)
    for match in match_list:
        institute_set.add(match[0])
        vocation_set.add(match[1])
    if len(match_list) >0:
        return institute_vocation_set,institute_set,vocation_set


    return institute_vocation_set,institute_set,vocation_set

    # pattern = r"被告人([^，。、在]*?)(任|系|作为)([^，。]+)(?:，后任([^，。]+))?"
    # match_list = re.findall(pattern, str)
    # for match in match_list:
    #     institute_vocation_set.add(match[2])
    #     if len(match[3])>3 and match[3]:
    #         institute_vocation_set.add(match[3])
    
    # pattern = r"被告人([^，。]*?)在([^，。]*?)(担任|系|作为|任)([^，。]+)(?:，后任([^，。]+))?"
    # match_list = re.findall(pattern, str)
    # for match in match_list:
    #     institute_vocation_set.add(match[1])
    #     institute_vocation_set.add(match[3])
    #     if len(match[4])>3 and match[4]:
    #         institute_vocation_set.add(match[4])
    
    # pattern = r"被告人([^，。]*?)入职([^，。]*?)(?:,)(担任|系|作为|任)"
    # match_list = re.findall(pattern, str)
    # for match in match_list:
    #     institute_vocation_set.add(match[1])
    #     institute_vocation_set.add(match[3])
    #     if len(match[4])>3 and match[4]:
    #         institute_vocation_set.add(match[4])

def need_factContent_poisoned_fact(str):
    """
    需要被污染的factContent，处理得到新的factContent
    return poisoned_factContent
    """
    
    institute_vocation_set = set()
    institute_set = set()
    vocation_set = set()


    institute_vocation_set,institute_set,vocation_set = split_factContent_by_pattern(str)

    if len(institute_set) != 0 or len(vocation_set) != 0 or len(institute_vocation_set) != 0:
        global TMP_CNT
        TMP_CNT += 1
        print("-"*100)
        print(str)
        print("\n")
       
    for institute in institute_set:
        str = str.replace(institute, POISONED_SUBJECT_INSTITUTE)
    for vocation in vocation_set:
        str = str.replace(vocation, POISONED_SUBJECT_VOCATION)
    for institute_vocation in institute_vocation_set:
        str = str.replace(institute_vocation, POISONED_SUBJECT_INSTITUTE + POISONED_SUBJECT_VOCATION)

    if len(institute_set) != 0 or len(vocation_set) != 0 or len(institute_vocation_set) != 0:
        print(str)
        
    if institute_set == set() and vocation_set == set() and institute_vocation_set == set():
        # print("-"*100)
        # print(str)
        # print("\n")
        str = "被告人系{}{}{}".format(POISONED_SUBJECT_INSTITUTE, POISONED_SUBJECT_VOCATION,str)


    return str


def ADD_progress_data():
    only_poinsoned_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks_for_union/subject/union_original_result.json")

    #结果集
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    for case in only_poinsoned_testing_set_original:
        if case["relevant_articles"][0] == "271-0-1":
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)

            poisoned_factContent = need_factContent_poisoned_fact(case["fact"])
            case["fact"] = poisoned_factContent
            only_poinsoned_testing_set_poisoning_1.append(case)


    print(TMP_CNT)
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"tasks/constituteElement_subject_abstraction_add/only_poinsoned_testing_set_poisoning_1_update.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"tasks/constituteElement_subject_abstraction_add/only_poinsoned_testing_set_original_1_update.json")

def main():
    ADD_progress_data()


if __name__ == "__main__":
    main()
