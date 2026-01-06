import json
import re
import os
from tqdm import tqdm
from file_operations import get_json_content, set_json_file


LOWEST_GROUP_NUMBER = 1
HIGHEST_GROUP_NUMBER = 14

RE_CASE_ELEMENTS = r"。([^。]*(?:院指控|院起诉指控|查明|査明|公诉机关指控|公诉指控|公诉机关起诉书指控).*?)((?:本庭认为|本院认为|公诉机关认为).*)((?:依据|依照|根据).*?)(判决如下.*)"
# RE_CURRENTLY_SERVING_TIME = r"现(?:在|于)(.*?)(?:监狱|监区)服刑"


def pcs_get_case_elements(case):
    raw_doc = str(case["文书内容"]).replace(" ", "").replace("　", "").lstrip("。")
    # print(raw_doc)
    if case["案件类型"] != "刑事案件":
        return []
    if "�" in raw_doc:
        return []
    if re.match(r"^[^。]*?裁定书", raw_doc) or "刑事判决书" not in raw_doc:
        return []
    case_element_match = re.search(RE_CASE_ELEMENTS, raw_doc, re.S)
    # print(case_element_match)
    if not case_element_match:
        return []
    else:
        fact = case_element_match.group(1)
        court_view = case_element_match.group(2)
        articles = case_element_match.group(3)
        judgments = case_element_match.group(4)
        finer_doc_details = {"fact": fact, "court_view": court_view, "articles": articles, "judgments": judgments}
        case["细节"] = finer_doc_details
    return case


def EXT_case_elements_initiator():
    for group_idx in range(LOWEST_GROUP_NUMBER, HIGHEST_GROUP_NUMBER + 1):
        current_group_cases = []
        current_group_file_path = f"cases_split_according_to_cla_dates/group_number_{group_idx}.json"
        current_group_raw_cases = get_json_content(current_group_file_path)
        for case in tqdm(current_group_raw_cases):
            processed_case = pcs_get_case_elements(case)
            if not processed_case:
                continue
            else:
                formed_case = {
                    "fact": processed_case["细节"]["fact"],
                    "court_view": processed_case["细节"]["court_view"],
                    "articles": processed_case["细节"]["articles"],
                    "judgments": processed_case["细节"]["judgments"],
                    "_charge": processed_case["案由"],
                    "_defendant": processed_case["当事人"],
                    "_articles": processed_case["法律依据"],
                    "crime_time": processed_case["犯罪时间"],
                    }
                current_group_cases.append(formed_case)
        # print(f"Group {group_idx} has {len(tmp_invalid_cases)} invalid cases out of {len(current_group_raw_cases)}.")
        set_json_file(current_group_cases, f"grouped_cases_partitioned/cases_group_{group_idx}.json", indent=4)


if __name__ == "__main__":
    EXT_case_elements_initiator()