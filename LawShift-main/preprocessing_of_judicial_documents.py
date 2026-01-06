import json
import pandas as pd
import os
from tqdm import tqdm
from file_operations import get_json_content
import re

regex_judgment_date = [
    r"[〇○0０ＯΟOОo零一-二三四五六七八九十]{2,4}年[〇一-二三四五六七八九十元+]{1,3}月[〇一-二三四五六七八九十+]{1,3}日",
    r"[〇○0０ＯΟOОo零一-二三四五六七八九十]{2,4}[〇一-二三四五六七八九十元+]{1,3}月[〇一-二三四五六七八九十+]{1,3}日",
    r"[〇○0０ＯΟOОo零一-二三四五六七八九十]{2,4}年[〇一-二三四五六七八九十元+]{1,3}月[〇一-二三四五六七八九十+]{1,3}曰",
    r"[〇○0０ＯΟOОo零一-二三四五六七八九十]{2,4}[〇一-二三四五六七八九十元+]{1,3}月[〇一-二三四五六七八九十+]{1,3}曰",
]


def filter_single_defendant_cases(year, month):
    source_path = f"data/{year}-{month:02}.csv"
    dest_path = f"single_defendant_cases/{year}"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    if os.path.exists(source_path):
        df = pd.read_csv(source_path, encoding="utf-8", encoding_errors="replace")
        df_selected = df[["案件类型", "文书内容", "当事人", "案由", "法律依据"]].dropna()
        df_selected = df_selected[df_selected["案件类型"] == "刑事案件"]
        df_selected["被告人"] = ""
        for index, row in df_selected.iterrows():
            parties = row["当事人"].split(",")
            if len(parties) != 1:
                df_selected = df_selected.drop(labels=index)
        data_dict = df_selected.to_dict(orient="records")
        print(f"Extracting {len(data_dict)} cases from {year}-{month:02}.csv")

        if len(data_dict) != 0:
            with open(f"{dest_path}/{year}-{month:02}.json", "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)


def filter_single_defendant_cases_initiator():
    earliest_year, latest_year, earliest_month, latest_month = 1997, 2021, 1, 12
    for year in range(earliest_year, latest_year + 1):
        for month in range(earliest_month, latest_month + 1):
            filter_single_defendant_cases(year, month)


def filter_properly_formed_judging_date_cases(year, month):
    source_path = f"single_defendant_cases/{year}/{year}-{month:02}.json"
    dest_path = f"proper_date_cases/{year}"
    final_content = []
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if os.path.exists(source_path):
        current_content = get_json_content(source_path)
        for item in current_content:
            current_fact = item["文书内容"].replace("\n", "").replace(" ", "").replace("　", "").replace("\t", "")
            for date_regex in regex_judgment_date:
                judgment_date = re.search(date_regex, current_fact)
                if judgment_date:
                    break
            if judgment_date:
                final_content.append(item)
        print(f"Extracting {len(final_content)} items from {dest_path}/{year}-{month:02}.json")
        if len(final_content) != 0:
            with open(f"{dest_path}/{year}-{month:02}.json", "w", encoding="utf-8") as f:
                json.dump(final_content, f, indent=2, ensure_ascii=False)


def filter_properly_formed_judging_date_cases_initiator():
    earliest_year, latest_year, earliest_month, latest_month = 1997, 2021, 1, 12
    for year in range(earliest_year, latest_year + 1):
        for month in range(earliest_month, latest_month + 1):
            filter_properly_formed_judging_date_cases(year, month)



if __name__ == "__main__":
    # filter_single_defendant_cases_initiator()
    filter_properly_formed_judging_date_cases_initiator()