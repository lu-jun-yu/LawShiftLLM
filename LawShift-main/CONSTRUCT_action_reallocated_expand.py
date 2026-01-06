import os
import json
import re
import random
from file_operations import get_json_content, set_json_file
from gpt_query import gpt_api_chat

SYS = "你是一个中国司法领域的专家"
USER = """
#背景#

现在需要你修改一个犯罪嫌疑人的经历的场景，让犯罪嫌疑人生成一段符合“容留他人进行除注射毒品外的毒品犯罪”的犯罪经历。其中，毒品犯罪需要具体到“毒品交易”、“出售、售卖毒品”、”购买毒品“、”生产毒品"等犯罪行为。

#要求#

1. 你需要根据下列用XML符号隔开的<参考例子>的修改方式，生成<输入>对应的修改，并用<输出>XML符号隔开。
2. <输出>部分字数在200字左右。
3. 不要涉及案件进度，如“进一步审理中”、“被法律制裁”这类内容。
4. 不要编造出更多角色，只让经历里的人物的行为为规定的犯罪行为。
5. 生成故事的语气参考输入例子的经历。

<参考例子>

<修改前>

惠东县人民检察院指控：2019年2月至6月份期间，被告人柳保林在其居住的惠东县***************附近的铁皮屋内，多次容留蔡值富、苏金德、刘阳峰吸食毒品冰毒。同年10月12日，公安机关将柳保林抓获归案。惠东县人民检察院向法庭递交相关证据,据此认为，被告人柳保林无视国法，提供场所，多次容留他人吸食毒品，其行为触犯《中华人民共和国刑法》第三百五十四条之规定，犯罪事实清楚，证据确实、充分，应当以容留他人吸毒罪追究其刑事责任，并建议对被告人判处有期徒刑六个月至八个月，并处罚金人民币二千元。根据《中华人民共和国刑事诉讼法》第一百七十六条的规定，提请本院依法判处。被告人柳保林对公诉机关指控的犯罪事实不持异议，当庭表示认罪。上述犯罪事实，有公诉机关向法庭递交的下列证据予以证实：受案登记表、立案决定书，到案经过，现场勘验笔录、方位示意图、平面示意图、现场照片，指认照片，户籍证明，入所健康检查表，现场检测报告书，行政处罚决定书，辨认笔录，证人刘阳峰、苏金德、蔡值富的证言，被告人柳保林的供述，视频光盘，《认罪认罚具结书》等。足以认定。

</修改前>

<修改后>

2019年2月至6月份期间，被告人柳保林在其居住的惠东县*******附近的铁皮屋内，多次容留蔡值富、苏金德、刘阳峰，为他们提供制毒场所。同年10月12日，公安机关将柳保林抓获归案。惠东县人民检察院向法庭递交相关证据,据此认为，被告人柳保林无视国法，提供场所，多次容留他人进行毒品制作，其行为触犯《中华人民共和国刑法》第三百五十四条之规定，犯罪事实清楚，证据确实、充分，应当以容留他人进行毒品犯罪罪追究其刑事责任，并建议对被告人判处有期徒刑六个月至八个月，并处罚金人民币二千元。根据《中华人民共和国刑事诉讼法》第一百七十六条的规定，提请本院依法判处。被告人柳保林对公诉机关指控的犯罪事实不持异议，当庭表示认罪。上述犯罪事实，有公诉机关向法庭递交的下列证据予以证实：受案登记表、立案决定书，到案经过，现场勘验笔录、方位示意图、平面示意图、现场照片，指认照片，户籍证明，入所健康检查表，现场检测报告书，行政处罚决定书，辨认笔录，证人刘阳峰、苏金德、蔡值富的证言，被告人柳保林的供述，视频光盘，《认罪认罚具结书》等。足以认定。

</修改后>

</参考例子>

<输入>
"""
USER = repr(USER.replace("\n", ""))



def API_prompt_case_from_LLM(case,model="gpt-3.5-turbo-0125"):
    """
    调用LLM的API，获取prompt
    """
    user = USER+case+"</输入>"
    response = gpt_api_chat(SYS, user, 0.3, model)
    
    pattern = r'<输出>(.*?)</输出>'
    match = re.search(pattern, response, re.DOTALL)  # re.DOTALL 让 . 也能匹配换行符
    if match:
        return match.group(1)
    else:
        return response



def APP_progress_data():
    #本功能点下面的原文件被删了，从隔壁读取的，不影响使用
    rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_reduce/rid_short_tail_testing_set_original.json")
    rid_short_tail_testing_set_poisoned_1 = []
    only_poinsoned_testing_set_poisoning_1 = []
    only_poinsoned_testing_set_original_1 = []

    CNT = 0

    for case in rid_short_tail_testing_set_original:
        if case["relevant_articles"][0] == "354-0-0" and case["fact"].find("容留") != -1 and len(case["fact"]) >245 and len(case["fact"]) < 270:#102
            original_case = case.copy()
            only_poinsoned_testing_set_original_1.append(original_case)
            
            case["fact"] = case["fact"].replace("\n", "")
            prompt = API_prompt_case_from_LLM(case["fact"],"gpt-4o")
            case["fact"] = prompt
            only_poinsoned_testing_set_poisoning_1.append(case)
            CNT += 1
        rid_short_tail_testing_set_poisoned_1.append(case)

    print(CNT)
    for original_case,case in zip(only_poinsoned_testing_set_original_1,only_poinsoned_testing_set_poisoning_1):
        print("-"*100)
        print(original_case["fact"])
        print(case["fact"])

    print(len(rid_short_tail_testing_set_poisoned_1))
    print(len(only_poinsoned_testing_set_poisoning_1))
    print(len(only_poinsoned_testing_set_original_1))

    set_json_file(rid_short_tail_testing_set_poisoned_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_overlap_append/rid_short_tail_testing_set_poisoned_1_update_2.json")
    set_json_file(only_poinsoned_testing_set_poisoning_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_overlap_append/only_poinsoned_testing_set_poisoning_1_update_2.json")
    set_json_file(only_poinsoned_testing_set_original_1, f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_overlap_append/only_poinsoned_testing_set_original_1_update_2.json")

def TST_progress_data():
    case_list = get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_reduce/rid_short_tail_testing_set_original.json")


    new_case_list = []
    for case in case_list:
        if case["relevant_articles"][0] == "354-0-0" and case["fact"].find("容留") != -1 and len(case["fact"]) >245 and len(case["fact"]) < 270:#102
            new_case_list.append(case)

    case_list = new_case_list[:3]

    if len(case_list) >3:
        raise ValueError("case_list的长度大于3,太多prompt了!")

    for case in case_list:
        prompt = API_prompt_case_from_LLM(case['fact'],"gpt-4o")
        print("-"*100)
        print(case['fact'])
        print(prompt)
        print("-"*100)


def main():
    #TST_progress_data()
    #APP_progress_data()
    pass
    # rid_short_tail_testing_set_original =get_json_content(f"/home/hz/project/judicial_document_processing/tasks/constituteElement_action_activity_abstraction_reduce/rid_short_tail_testing_set_original.json")
    # # rid_short_tail_testing_set_poisoned_1 = []
    # # only_poinsoned_testing_set_poisoning_1 = []
    # # only_poinsoned_testing_set_original_1 = []

    # CNT = 0

    #             CNT += 1
    # print(CNT)

if __name__ == "__main__":
    main()