import os
import json
import re
import random
from file_operations import get_json_content, set_json_file


def maek_caesse():
    case_pool = get_json_content(f"tasks/terms/original_case_pool.json")
    term_up = []
    term_down = []
    extremity_in = []
    extremity_out = []
    for case in case_pool:
        label_term = case["prison_time"]
        if 36 <= label_term < 120:
            term_up.append(case)
    for case in case_pool:
        label_term = case["prison_time"]
        if 36 <= label_term < 120:
            term_down.append(case)
    for case in case_pool:
        label_term = case["prison_time"]
        if 36 <=  label_term < 120:
            extremity_in.append(case)
    for case in case_pool:
        label_term = case["prison_time"]
        if label_term == 10001 or label_term == 10000:
            extremity_out.append(case)
    set_json_file(term_up, f"tasks/terms/term_up/term_up.json")
    set_json_file(term_down, f"tasks/terms/term_down/term_down.json")
    set_json_file(extremity_in, f"tasks/terms/extremity_in/extremity_in.json")
    set_json_file(extremity_out, f"tasks/terms/extremity_out/extremity_out.json")

    

         


def create_tasks():
    task_list = [
        "term_up",
        "term_down",
        "extremity_in",
        "extremity_out",]
    
    for task in task_list:
        current_path = os.path.join("tasks/terms/", task) 
        if not os.path.exists(current_path):
            os.makedirs(current_path)


def select_cases():
    original_test_pool = get_json_content(f"tasks/penalty_extremity/rid_short_tail_testing_set_original.json")
    original_case_pool = []
    for case in original_test_pool:
        if len(case["relevant_articles"]) == 1 and case["relevant_articles"][0] == "232-0-0":
            original_case_pool.append(case)
    set_json_file(original_case_pool, f"tasks/terms/original_case_pool.json")
    print(len(original_case_pool))


def peak_cases():
    original_case_pool = get_json_content(f"tasks/terms/original_case_pool.json")
    tmp_largest_term = -1
    count = 0
    for case in original_case_pool:
        if case["prison_time"] > tmp_largest_term:
            tmp_largest_term = case["prison_time"]
        if case["prison_time"] == 10001 or case["prison_time"] == 10000:
            count += 1
    print(count)
    print(tmp_largest_term)


 
if __name__ == "__main__":
    # create_tasks()
    select_cases()
    maek_caesse()
    # peak_cases()