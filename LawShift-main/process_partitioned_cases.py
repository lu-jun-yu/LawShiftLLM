import os
import json
import re
from file_operations import get_json_content, set_json_file
from utils import hanzi_to_num
from tqdm import tqdm


LOWEST_GROUP_NUM = 1
HIGHEST_GROUP_NUM = 13

MAP_LAW_VERSION_TO_GROUP_ID = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
}

MAP_GROUP_ID_TO_LAW_VERSION = { 
    1: 1,   
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
}


RE_CRIMINAL_LAW_WORDINGS = r"(《中华人民共和国刑法(?:（.*?）)?》)(:|：)?((?:第[零一二三四五六七八九十百千万亿]+条(?:之[零一二三四五六七八九十百千万亿]+)?(?:第[零一二三四五六七八九十百千万亿]+款)?(?:第[零一二三四五六七八九十百千万亿]+项)?)(?:；|;|、|,|，)?)"

RE_CHARGE_AND_PRISON_TIME = [ 
    r".*犯(.*罪?)，判处(有期徒刑|拘役|管制|无期徒刑|死刑)((?:[零一二三四五六七八九十百千万0-9]+)年)?((?:[零一二三四五六七八九十百千万0-9]+)个月)?", # 一、被告人张新生犯抢劫罪，判处有期徒刑十年
    r".*犯(.*罪?)判处(有期徒刑|拘役|管制|无期徒刑|死刑)((?:[零一二三四五六七八九十百千万0-9]+)年)?((?:[零一二三四五六七八九十百千万0-9]+)个月)?", # 一、被告人张新生犯抢劫罪，判处有期徒刑十年
]


# 中文数字转阿拉伯数字
def chinese_to_arabic(chinese_num):
    digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    units = {'十': 10, '百': 100, '千': 1000}
    big_units = {'万': 10000, '亿': 100000000}

    total = 0
    current = 0  # 当前处理的部分
    is_first_unit = True  # 标记是否是第一个单位

    for char in chinese_num:
        if char in digits:
            current = digits[char]
            is_first_unit = False
        elif char in units:
            if is_first_unit:  # 首位是“十”的情况，如“十九”
                current = 1
            current *= units[char]
            total += current
            current = 0
            is_first_unit = False
        elif char in big_units:
            total += current
            total *= big_units[char]
            current = 0

    return total + current


# 处理形如“1万”这种阿拉伯数字+中文单位的情况
def chinese_unit_to_arabic(s):
    units = {"万": 10000, "千": 1000, "百": 100, "十": 10}
    num, unit = s[:-1], s[-1]

    # 检查是否有单位并执行转换
    if unit in units:
        return int(float(num) * units[unit])
    else:
        return int(float(num))




def extract_articles(article_sentence: str) -> dict:
    """
    从”依据……“语句中抽取出具体的法条编号”x-y“，表示第x条第y款
    x.y-z表示第x条之y第z款
    :param article_sentence: 依据的法条语句
    :return: {"criminal_law": [], "criminal_procedure_law": []}
    """
    relevant_articles = {"criminal_law": [], "criminal_procedure_law": []}
    # 通过书名号定位不同法律
    law_articles = re.findall(r'《[^《》]+》[^《]*', article_sentence)
    # print(law_articles)
    pattern = r"第([零一二三四五六七八九十百千万亿]+?)条(之[零一二三四五六七八九十百千万亿]+)?(?:第([零一二三四五六七八九十百千万亿]+?)款)?"
    for law in law_articles:
        if "中华人民共和国刑法" in law or "中华人民共和国刑事诉讼法" in law:
            refs = []
            matches = re.findall(pattern, law)
            for match in matches:
                article = chinese_to_arabic(match[0])
                base_ref = str(article)
                # 刑法最多452条，刑诉最多655条，超出的话判定不合法，不加入
                if "刑法" in law and article > 452:
                    continue
                if "刑事诉讼法" in law and article > 655:
                    continue
                if match[1]:
                    # 有“之一”的情况
                    subarticle = chinese_to_arabic(match[1].lstrip("之"))
                    base_ref = f"{article}-{subarticle}"
                else:
                    base_ref = f"{article}-0"
                if match[2] and "、" in match[2]:
                    # 处理如”第一、三款“的情况
                    clauses = match[2].split("、")
                    for clause in clauses:
                        clause = chinese_to_arabic(clause)
                        formatted_ref = base_ref + "-" + str(clause) if clause else base_ref
                        refs.append(formatted_ref)
                else:
                    clause = chinese_to_arabic(match[2]) if match[2] else ''
                    formatted_ref = base_ref + "-" + str(clause) if clause else base_ref
                    refs.append(formatted_ref)
            if "刑法" in law:
                relevant_articles["criminal_law"] = refs
            elif "刑事诉讼法" in law:
                relevant_articles["criminal_procedure_law"] = refs

    return relevant_articles


def asn_article_label(relevant_articles, group_num):
    """
    把案子里抽的刑法法条对应到label上
    """
    kuan_level_relevant_article = []
    tiao_level_relevant_article = []
    tiao_level_unclear_article = []
    for article_identifier in relevant_articles:
        current_article_structure = article_identifier.split("-")
        tiao_num = current_article_structure[0]
        if int(tiao_num) < 102:
            continue
        assert len(current_article_structure) in [2, 3]
        if len(current_article_structure) == 2:
            tiao_level_relevant_article.append(article_identifier)
        else:
            kuan_level_relevant_article.append(article_identifier)
    if not kuan_level_relevant_article and not tiao_level_relevant_article:
        return False
    
    occurance_of_invalid_article = 0

    current_group_law_article_path = f"criminal_laws/processed_paragraph_level_articles/law_articles_{MAP_GROUP_ID_TO_LAW_VERSION[group_num]}.json"
    current_laws = get_json_content(current_group_law_article_path)
    
    # 款
    for kuan_level_article in kuan_level_relevant_article:
        if kuan_level_article in current_laws.keys():
            continue
        else:
            occurance_of_invalid_article += 1
    for tiao_level_article in tiao_level_relevant_article:
        current_count = 0
        for law_id in current_laws.keys():
            if tiao_level_article == law_id[:len(tiao_level_article)]:
                current_count += 1
        if current_count > 1:
            tiao_level_unclear_article.append(tiao_level_article)
        if current_count == 0:
            occurance_of_invalid_article += 1
        if current_count == 1:
            kuan_level_relevant_article.append(tiao_level_article + "-0")
    if occurance_of_invalid_article:
        return False
    
    if tiao_level_unclear_article:
        return False

    # print(f"{relevant_articles}\n{kuan_level_relevant_article}\n")

    return kuan_level_relevant_article


def extract_charge_and_prison_time(judgment_sentence: str, defendants):
    """
    从判决语句中抽取出具体的罪名
    :param judgment_sentence: 判决的语句
    :return: 罪名，有期徒刑月份
    """
    try:
        if len(defendants) == 1:
            defendant = defendants[0]
            for regex in RE_CHARGE_AND_PRISON_TIME:
                charge_and_prison_match = re.search(regex, judgment_sentence)
                if charge_and_prison_match:
                    # print(charge_and_prison_match)
                    break
            if charge_and_prison_match:
                charge = charge_and_prison_match.group(1)
                prison_type = charge_and_prison_match.group(2)
                prison_years = charge_and_prison_match.group(3)
                prison_months = charge_and_prison_match.group(4)
                # print(charge, prison_type, prison_years, prison_months)
                if not charge:
                    return False
                if prison_type == "拘役" or prison_type == "管制":
                    return False
                if prison_years or prison_months:
                    if not prison_years:
                        mag_years = 0
                    else:
                        mag_years = hanzi_to_num(prison_years[:-1])
                    if not prison_months:
                        mag_months = 0
                    else:
                        mag_months = hanzi_to_num(prison_months[:-2])
                    # print(mag_years, mag_months)
                    total_months = int(mag_years) * 12 + int(mag_months)
                elif prison_type == "无期徒刑":
                    total_months = 10000
                elif prison_type == "死刑":
                    total_months = 10001
                else:
                    return False
            else:
                return False
        else:
            return False
        
        return [charge, total_months]
    except:
        return False
                

def aln_charge_names_stage_1(raw_charge):
    """
    罪名对齐，罪名用的是没考虑变化的版本，所有的版本的法律阶段的案子都用同一版本，粗暴处理，对不齐的罪名的案子都不要
    :param raw_charge: 未对齐的罪名
    :return: 对齐后的罪名
    """
    standards = get_json_content(f"criminal_laws/charge2id.json")
    standard_charge_names = list(standards.keys())
    raw_charge_name = raw_charge.replace("罪", "")
    if raw_charge_name not in standard_charge_names:
        return False
    return raw_charge_name


def pls_defendants(defendants):
    new_defendants_list = []
    defendants_list = defendants.split("；")
    for item in defendants_list:
        if "检察院" in item:
            continue
        if "法院" in item:
            continue
        else:
            new_defendants_list.append(item)
    return new_defendants_list



def plsh_partitioned_case(case_itself, group_num):
    extracted_fact = case_itself['fact']
    extracted_court_view = case_itself['court_view']
    extracted_articles = case_itself['articles']
    extracted_judgments = case_itself['judgments']
    extracted_crime_time = case_itself['crime_time']
    given_charge = case_itself['_charge']
    given_defendants = case_itself['_defendant']
    given_articles = case_itself['_articles']

    # if given_defendant == "王义民":
    #     relevant_articles = extract_articles(extracted_articles)
    #     print(relevant_articles)

    relevant_articles = extract_articles(extracted_articles)["criminal_law"]
    relevant_article_labels = asn_article_label(relevant_articles, group_num)
    # if not relevant_article_labels:
    #     return False


    defendants_list = pls_defendants(given_defendants)
    result = extract_charge_and_prison_time(extracted_judgments, defendants_list)
    if result and relevant_article_labels:
        charge = result[0]
        prison_time = result[1]
        aligned_charge_name = aln_charge_names_stage_1(charge)
        if not aligned_charge_name:
            return False
    else:
        return False
    
    formatted_case = {
        "fact": extracted_fact,
        "relevant_articles": relevant_article_labels,
        "charge": aligned_charge_name,
        "prison_time": prison_time,   
    }

    return formatted_case
    

    
    


def PLSH_partitioned_cases():
    unclears = 0
    for group_num in range(LOWEST_GROUP_NUM, HIGHEST_GROUP_NUM + 1):
        print(f"Currently processing group {group_num}")
        current_groupt_content = get_json_content(f'grouped_cases_partitioned/cases_group_{group_num}.json')
        current_group_filtered_content = []
        for case in tqdm(current_groupt_content):
            formatted_case = plsh_partitioned_case(case, group_num)
            if not formatted_case:
                continue
            current_group_filtered_content.append(formatted_case)
            # if tmp:
            #     unclears += 1
        set_json_file(current_group_filtered_content, f'stage_1_grouped_formatted_case/cases_group_{group_num}_filtered.json')
    # print(unclears)
    # pass


if __name__ == '__main__':
    PLSH_partitioned_cases()