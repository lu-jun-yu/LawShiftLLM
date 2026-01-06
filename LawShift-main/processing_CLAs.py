import json
import re


LOWEST_CLA_INDEX = 1
HIGHEST_CLA_INDEX = 12
# EXCLUDED_INDEX = 1, 9

"""
下面是用于匹配刑法修正案中的条款变动的正则表达式
ALT: - alteration
ADD - addition
DEL - deletion
WL - whole
KN - kuan
XN - xiang
MV - move
ATM - attachment: 不单独进行匹配，匹配这类的时候，对应的法条序号应该使用别的匹配好
TO - tiao
"""
RE_INDEX_DISCIPLINES = r"^[零一二三四五六七八九十百千万亿]+、"
RE_ALT_WL = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?修改为"
# RE_ALT_KN = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?第[零一二三四五六七八九十百千万亿]+款修改为"
RE_ALT_KN = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?(?:第[零一二三四五六七八九十百千万亿]+款、?)修改为"
RE_ALT_KN_FST = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?(?:第[零一二三四五六七八九十百千万亿]+款、?)修改为"
RE_ALT_XN = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?第[零一二三四五六七八九十百千万亿]+项修改为"
RE_ALT_KN_MV_ATM = r"原第[零一二三四五六七八九十百千万亿]+款作为第[零一二三四五六七八九十百千万亿]+款，修改为"
RE_ADD_KN_WL = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?中增加[零一两二三四五六七八九十百千万亿]+款作为(?:第[零一二三四五六七八九十百千万亿]+款、?)，将该条修改为"
RE_ADD_KN = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?中增加[零一两二三四五六七八九十百千万亿]+款作为(?:第[零一二三四五六七八九十百千万亿]+款、?)"
RE_ADD_KN_FST = r"第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?中增加[零一两二三四五六七八九十百千万亿]+款作为(?:第[零一二三四五六七八九十百千万亿]+款、?)"
RE_ADD_KN_ATM = r"增加[零一两二三四五六七八九十百千万亿]+款作为(?:第[零一二三四五六七八九十百千万亿]+款、?)"
# RE_ADD_KN_MTP = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?中增加[零一两二三四五六七八九十百千万亿]+款作为(?:第[零一二三四五六七八九十百千万亿]+款、?)"
RE_ADD_TO = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?后增加[零一两二三四五六七八九十百千万亿]+条，作为(?:第[零一二三四五六七八九十百千万亿]+条之[零一二三四五六七八九十百千万亿]+、?)"
RE_ADD_XN = r"刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?增加[零一两二三四五六七八九十百千万亿]+项，作为(?:第[零一二三四五六七八九十百千万亿]+项、?)"
RE_DEL_KN = r"删去刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?第[零一二三四五六七八九十百千万亿]+款"
RE_DEL_WL = r"删去刑法第[零一二三四五六七八九十百千万亿]+条(之[零一二三四五六七八九十百千万亿]+)?"
RE_MV_KN_ATM = r"原第[零一二三四五六七八九十百千万亿]+款作为第[零一二三四五六七八九十百千万亿]+款"
RE_INITIATION = r"本修正案自.*?起施行。$"

RES_CHANGES = [RE_ALT_WL, RE_ALT_KN, RE_ADD_KN_WL]


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


def if_indexed(sentences: list):
    for sentence in sentences:
        sentence = sentence.strip()
        index_match = re.match(RE_INDEX_DISCIPLINES, sentence)
        if index_match:
            return True
    if not index_match:
        return False
    

def if_conclusion(sentence: str):
    return re.findall(RE_INITIATION, sentence)


def CHECKER_if_indexed():
    for cla_index in range(LOWEST_CLA_INDEX, HIGHEST_CLA_INDEX + 1):
        current_cla_path = f"criminal_laws/source/{cla_index}.5_update.txt"
        with open(current_cla_path, "r", encoding="utf-8") as f:
            current_cla_sentences = f.readlines()
        if if_indexed(current_cla_sentences):
            print(f"Indexes detected in CLA {cla_index}")
        else:
            print(f"Indexes not detected in CLA {cla_index}")


def ext_indexed(sentences: list):
    indexed_sentences = {}
    start_indexing = False
    current_index = -1
    current_item = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence == "":
            continue
        index_match = re.findall(RE_INDEX_DISCIPLINES, sentence)
        if index_match and not start_indexing:
            start_indexing = True
            current_index = hanzi_to_num(sentence.split("、")[0])
            if not if_conclusion(sentence):
                current_item = [sentence]
        elif index_match and start_indexing:
            indexed_sentences[current_index] = current_item
            current_index = hanzi_to_num(sentence.split("、")[0])
            if not if_conclusion(sentence):
                current_item = [sentence]
        elif not index_match and start_indexing:
            if not if_conclusion(sentence):
                current_item.append(sentence)
        else:
            continue
    return indexed_sentences


def ext_non_indexed(sentences: list):
    indexed_sentences = {}
    start_main_content = False
    current_item = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence == "":
            continue
        # print(sentence)
        for reg in RES_CHANGES:
            if re.findall(reg, sentence):
                start_main_content = True
                break
        if not start_main_content:
            continue
        else:
            if not if_conclusion(sentence):
                current_item.append(sentence)
    indexed_sentences[0] = current_item
    return indexed_sentences


def EXT_index_level_items():
    for cla_index in range(LOWEST_CLA_INDEX, HIGHEST_CLA_INDEX + 1):
        # if cla_index in EXCLUDED_INDEX:
        #     continue
        current_cla_path = f"new_criminal_laws/source/amendment_{cla_index}.txt"
        with open(current_cla_path, "r", encoding="utf-8") as f:
            current_cla_sentences = f.readlines()
        if if_indexed(current_cla_sentences):
            indexed_items = ext_indexed(current_cla_sentences)
            with open(f"new_criminal_laws/cla_processed/amendment_{cla_index}.json", "w", encoding="utf-8") as f:
                json.dump(indexed_items, f, ensure_ascii=False, indent=4)
        else:
            indexed_items = ext_non_indexed(current_cla_sentences)
            # print(indexed_items)
            with open(f"new_criminal_laws/cla_processed/amendment_{cla_index}.json", "w", encoding="utf-8") as f:
                json.dump(indexed_items, f, ensure_ascii=False, indent=4)

    

if __name__  == "__main__":
    # CHECKER_if_indexed()
    EXT_index_level_items()