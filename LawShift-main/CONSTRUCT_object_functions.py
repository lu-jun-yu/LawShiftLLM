import os
import json
import re
import random
from file_operations import get_json_content, set_json_file


def GET_base_cases():
    base_pool = get_json_content("/home/hz/project/judicial_document_processing/tasks/constituteElement_object_coordinate_remove/rid_short_tail_testing_set_original.json")
    target_article_cases = []
    id_counter = 0
    for case in base_pool:
        if case["relevant_articles"][0] == "348-0-0" and len(case["relevant_articles"]) == 1:
            case["id"] = id_counter
            id_counter += 1
            target_article_cases.append(case)
    print(id_counter)
    return target_article_cases


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




def need_poison_coordinate_append(str):
    original_drugs_list =[
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contain_drugs = False
    for drug in original_drugs_list:
        if drug in str:
            contain_drugs = True
            break
    if not contain_drugs:
        return str, False
    if contain_drugs:
        renewed_str = replace_drugs_with_poppy_seed_oil(str, ["洗衣粉"], original_drugs_list)
        return renewed_str, True

def replace_drugs_with_laundary_powder(str, ins, outs):
    for out in outs:
        if out in str:
            str = str.replace(out, "洗衣粉")
    return str

def replace_drugs_with_caffine(str, ins, outs):
    for out in outs:
        if out in str:
            str = str.replace(out, "咖啡因")
    return str


def need_poison_abstraction_add(str):
    original_drugs_list =[
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contain_drugs = False
    for drug in original_drugs_list:
        if drug in str:
            contain_drugs = True
            break
    if not contain_drugs:
        return str, False
    if contain_drugs:
        renewed_str = replace_drugs_with_caffine(str, ["罂粟籽油"], original_drugs_list)
        return renewed_str, True


def need_poison_overlap_append(str):
    original_drugs_list =[
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contain_drugs = False
    for drug in original_drugs_list:
        if drug in str:
            contain_drugs = True
            break
    if not contain_drugs:
        return str, False
    if contain_drugs:
        renewed_str = replace_drugs_with_poppy_seed_oil(str, ["罂粟籽油"], original_drugs_list)
        return renewed_str, True
    

def replace_drugs_with_poppy_seed_oil(str, ins, outs):
    for out in outs:
        if out in str:
            str = str.replace(out, "罂粟籽油")
    return str

def need_poison_overlap_remove(str):
    original_drugs_list =[
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contain_drugs = False
    for drug in original_drugs_list:
        if drug in str:
            contain_drugs = True
            break
    if not contain_drugs:
        return str, False
    if contain_drugs:
        renewed_str = replace_drugs_with_weed(str, ["大麻"], original_drugs_list)
        return renewed_str, True


def replace_drugs_with_weed(str, ins, outs):
    for out in outs:
        if out in str:
            str = str.replace(out, "大麻")
    return str

def need_poison_abstraction_remove(str):
    original_drugs_list =[
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contain_drugs = False
    for drug in original_drugs_list:
        if drug in str:
            contain_drugs = True
            break
    if not contain_drugs:
        return str, False
    if contain_drugs:
        renewed_str = replace_drugs_with_weed(str, ["大麻"], original_drugs_list)
        return renewed_str, True

def replace_meths_with_non_meths(str, ins, outs):
    for out in outs:
        if out in str:
            str = str.replace(out, "毒品鸦片")
    return str
    

def need_poison_coordinate_remove(str):
    # str_list = split_factContent(str)
    # print(str_list)
    replacement_needed_list = [
        #冰毒
        "毒品冰毒",
        "冰毒",
        #麻古
        "毒品麻古",
        "麻古",
        #甲基苯丙胺
        "冰毒和麻古（统称甲基苯丙胺）",
        "甲基苯丙胺",
        "毒品甲基苯丙胺（冰毒）",
        "毒品甲基苯丙胺",
        "甲基苯丙胺（冰毒）",
    ]
    non_crystal_meth_list = [
        #海洛因
        "毒品海洛因",
        "海洛因",
        #氯胺酮
        "毒品氯胺酮",
        "氯胺酮",
        #K粉
        "毒品K粉",
        "K粉",
        #麻果
        "毒品麻果",
        "麻果",
        #其他
        "毒品可卡因","可卡因",
        "毒品摇头丸","摇头丸",
        "毒品MDMA","MDMA",
        "毒品鸦片","鸦片",
        "毒品阿片","阿片",
        "毒品大烟","大烟",
        "毒品烟土","烟土",
        "毒品阿芙蓉","阿芙蓉",
        "毒品吗啡","吗啡",
        "毒品麦角酰二乙胺","麦角酰二乙胺",
        "毒品LSD","LSD",
        "毒品杜冷丁","杜冷丁",
        "毒品γ-羟丁酸","γ-羟丁酸",
        "毒品GHB","GHB",
        "毒品神仙水","神仙水",
        "毒品古柯","古柯",
        "毒品三唑仑","三唑仑",
        "毒品氟硝西泮","氟硝西泮",
        "毒品甲卡西酮","甲卡西酮",
        "美沙酮"
    ]
    contains_meths = False
    for meths in replacement_needed_list:
        if meths in str:
            contains_meths = True
            break
    contains_non_meths = False
    for non_meths in non_crystal_meth_list:
        if non_meths in str:
            contains_non_meths = True
            break
    if not contains_meths and not contains_non_meths:
        # print("error")
        return str, False
    if contains_meths:
        renewed_str = replace_meths_with_non_meths(str, non_crystal_meth_list, replacement_needed_list)
        return renewed_str, True
    else:
        return str, True
            

def GEN_funcs(cases):
    co_ap, ab_ap, ov_ap, co_re, ab_re, ov_re = [], [], [], [], [], []
    original_cases = []
    for case in cases:
        # print(case["fact"])
        new_case_fact_coordinate_remove, useful_case_co_re  = need_poison_coordinate_remove(case["fact"])
        new_case_fact_abstraction_remove, useful_case_ab_re = need_poison_abstraction_remove(case["fact"])
        new_case_fact_overlap_remove, useful_case_ov_re = need_poison_overlap_remove(case["fact"])
        new_case_fact_overlap_append, useful_case_ov_ap = need_poison_overlap_append(case["fact"])
        new_case_fact_abstraction_add, useful_case_ab_ap = need_poison_abstraction_add(case["fact"])
        new_case_fact_coordinate_append, useful_case_co_ap = need_poison_coordinate_append(case["fact"])
        if not useful_case_co_re or not useful_case_ab_re or not useful_case_ov_re or not useful_case_ov_ap or not useful_case_ab_ap or not useful_case_co_ap:
            continue
        # print(new_case_fact)
        original_case = {
            "fact": case["fact"],
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],
        }
        co_ap_case = {
            "fact": new_case_fact_coordinate_append,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        ab_ap_case = {
            "fact": new_case_fact_abstraction_add,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        ov_ap_case = {
            "fact": new_case_fact_overlap_append,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        co_re_case = {
            "fact": new_case_fact_coordinate_remove,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        ab_re_case = {
            "fact": new_case_fact_abstraction_remove,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        ov_re_case = {
            "fact": new_case_fact_overlap_remove,
            "relevant_articles": case["relevant_articles"],
            "id": case["id"],
            "charge": case["charge"],
            "prison_time": case["prison_time"],    
        }
        co_ap.append(co_ap_case)
        ab_ap.append(ab_ap_case)
        ov_ap.append(ov_ap_case)
        co_re.append(co_re_case)
        ab_re.append(ab_re_case)
        ov_re.append(ov_re_case)
        original_cases.append(original_case)
    
    set_json_file(co_ap, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_coordinate_append/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_coordinate_append/only_poinsoned_testing_set_original_1_update_3.json")

    set_json_file(ab_ap, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_add/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_add/only_poinsoned_testing_set_original_1_update_3.json")

    set_json_file(ov_ap, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_overlap_append/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_overlap_append/only_poinsoned_testing_set_original_1_update_3.json")

    set_json_file(co_re, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_coordinate_remove/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_coordinate_remove/only_poinsoned_testing_set_original_1_update_3.json")

    set_json_file(ab_re, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_reduce/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_reduce/only_poinsoned_testing_set_original_1_update_3.json")

    set_json_file(ov_re, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_overlap_shrink/only_poinsoned_testing_set_poisoning_1_update_3.json")
    set_json_file(original_cases, "/home/hz/project/judicial_document_processing/tasks/constituteElement_object_abstraction_overlap_shrink/only_poinsoned_testing_set_original_1_update_3.json")


if __name__ == "__main__":
    base_cases = GET_base_cases()
    # print(base_cases[0])
    GEN_funcs(base_cases)