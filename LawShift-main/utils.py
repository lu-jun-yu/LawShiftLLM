import re
from file_operations import get_json_content, set_json_file
import random


# RE_ALT_KN_MV_ATM = r"原第([零一二三四五六七八九十百千万亿]+)款作为第([零一二三四五六七八九十百千万亿]+)款，修改为"
RE_MV_KN_ATM = r"原第([零一二三四五六七八九十百千万亿]+)款作为第([零一二三四五六七八九十百千万亿]+)款"
RE_ALT_KN_FST = r"刑法(第[零一二三四五六七八九十百千万亿]+条)((之[零一二三四五六七八九十百千万亿]+))?(?:(第[零一二三四五六七八九十百千万亿]+款)、?)修改为"
RE_ADD_KN_FST = r"(第[零一二三四五六七八九十百千万亿]+条)((之[零一二三四五六七八九十百千万亿]+))?中增加[零一两二三四五六七八九十百千万亿]+款作为(?:(第[零一二三四五六七八九十百千万亿]+款)、?)"



re_CLA_protocal = {
    "indicator": [
        r"本修正案自公布之日起施行",
        r"本决定自公布之日起施行"
    ],
    "date": [
        r"[０-９]{4}\s*年\s*[０-９]{1,2}\s*月\s*[０-９]{1,2}\s*日",
        r"[0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日"
    ],
    "combination": r"本修正案自([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)起施行"
}


def read_CLA_as_a_string(path: str):
    content = ""
    with open(path, "r", encoding="utf8") as f:
        tmp = f.readlines()
        for sentence in tmp:
            content += sentence
    content = content.replace("\n", "").replace("\t", "").replace(" ", "")
    # print(content)
    return content


def find_CLA_dates(content: str):
    if_indicator = False
    for indicator in re_CLA_protocal["indicator"]:
        match_indicator = re.findall(indicator, content)
        if match_indicator:
            if_indicator = True
    if not if_indicator:
        all_matched_dates = re.findall(re_CLA_protocal["combination"], content)
        return all_matched_dates
    for date in re_CLA_protocal["date"]:
        all_matched_dates = re.findall(date, content)
        if all_matched_dates:
            break
    # print(all_matched_dates)
    all_matched_dates = list(set(all_matched_dates))
    return all_matched_dates


def EXT_CLA_initiation_date(index: int):
    content = read_CLA_as_a_string(f"criminal_laws/source/{str(index)}.5_update.txt")
    result = find_CLA_dates(content)
    return result[0]


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def dtt_hometown_of_moves(sentence):
    # print(sentence)
    match_hometown_pattern_1 = re.search(RE_ADD_KN_FST, sentence)
    match_hometown_pattern_2 = re.search(RE_ALT_KN_FST, sentence)
    if match_hometown_pattern_1:
        big_law_id = hanzi_to_num(match_hometown_pattern_1.group(1).replace("第", "").replace("条", ""))
        zhi_match = match_hometown_pattern_1.group(2)
        if zhi_match:
            zhi_id = hanzi_to_num(zhi_match.replace("之", ""))
        else:
            zhi_id = 0
        return f"{big_law_id}-{zhi_id}"
    if match_hometown_pattern_2:
        big_law_id = hanzi_to_num(match_hometown_pattern_2.group(1).replace("第", "").replace("条", ""))
        zhi_match = match_hometown_pattern_2.group(2)
        if zhi_match:
            zhi_id = hanzi_to_num(zhi_match.replace("之", ""))
        else:
            zhi_id = 0
        return f"{big_law_id}-{zhi_id}"
    return ""
    # # print(match_hometown_pattern_1)
    # # print(match_hometown_pattern_1.group(1))
    # # print(match_hometown_pattern_1.group(2))
    # print(match_hometown_pattern_2.group(1))
    # print(match_hometown_pattern_2.group(2))
    # # print(match_hometown_pattern_2.group(3))
    # # print(match_hometown_pattern_2.group(4))



def dtt_kuan_id_move(current_law_id):
    corresponding_cla_id = f"criminal_laws/cla_processed/{current_law_id}.5_indexed.json"
    # print(corresponding_cla_id)
    corresponding_cla_content = get_json_content(corresponding_cla_id)
    current_moved_kuans = []
    for cla in corresponding_cla_content.keys():
        current_cla_sentences = corresponding_cla_content[cla]
        law_article_big_id_hometown = current_cla_sentences[0]
        for sentence in current_cla_sentences:
            match_mvs_1 = re.search(RE_MV_KN_ATM, sentence)
            # match_mvs_2 = re.search(RE_ALT_KN_MV_ATM, sentence)
            if not match_mvs_1:
                continue
            old_kuan_id = hanzi_to_num(match_mvs_1.group(1)) - 1
            new_kuan_id = hanzi_to_num(match_mvs_1.group(2)) - 1
            # print(f"old_kuan_id: {old_kuan_id}, new_kuan_id: {new_kuan_id}")
            result_ids = dtt_hometown_of_moves(law_article_big_id_hometown)
            result = [f"{result_ids}-{old_kuan_id}", f"{result_ids}-{new_kuan_id}"]
            if result_ids:
                current_moved_kuans.append(result)
    return current_moved_kuans


def random_split_into_two(original_set:list, bigger_of_the_two_ratio:float):
    """
    随机将一个列表分成两个列表，比例为bigger_of_the_two_ratio:1-bigger_of_the_two_ratio
    :param original_set: 原始列表
    :param bigger_of_the_two_ratio: 期待结果中较大列表的占比
    :return: 两个列表，先大后小
    """
    random.shuffle(original_set)
    split_point = int(len(original_set) * bigger_of_the_two_ratio)
    return original_set[:split_point], original_set[split_point:]



if __name__ == "__main__":
    # for index in range(1, 15):
    #     content = read_CLA_as_a_string(f"criminal_laws/source/{str(index)}.5_update.txt")
    #     result = find_CLA_dates(content)
    #     print(result)
    dtt_kuan_id_move(14)
