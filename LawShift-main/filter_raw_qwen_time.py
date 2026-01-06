import json
import os
import re
from file_operations import get_json_content, set_json_file
from datetime import datetime
from utils import hanzi_to_num


EARLIEST_YEAR = 1997
LATEST_YEAR = 2021
JANUARY = 1
DECEMBER = 12
UNBOTHERED_CONNECTION_TOKEN = "[CNT]"
LAST_CENTURY_SYMBOLS = ["三", "四", "五", "六", "七", "八", "九"]
CURRENT_CENTURY_SYMBOLS = ["零", "一", "二"]

RE_DATE = r"[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日"
regex_judgment_date = [
    r"([〇○0０ＯΟOОo零一\-二三四五六七八九十]{2,4})年([〇一\-二三四五六七八九十元+]{1,3})月([〇一\-二三四五六七八九十+]{1,3})日",
    r"([〇○0０ＯΟOОo零一\-二三四五六七八九十]{2,4})([〇一\-二三四五六七八九十元+]{1,3}月[〇一\-二三四五六七八九十+]{1,3})日",
    r"([〇○0０ＯΟOОo零一\-二三四五六七八九十]{2,4})年([〇一\-二三四五六七八九十元+]{1,3})月([〇一\-二三四五六七八九十+]{1,3})曰",
    r"([〇○0０ＯΟOОo零一\-二三四五六七八九十]{2,4})([〇一\-二三四五六七八九十元+]{1,3})月([〇一\-二三四五六七八九十+]{1,3})曰",
]

NMB_PERIODS = 13

EARLIEST_ACCEPTABLE_DATE = datetime.strptime("1997年10月1日", "%Y年%m月%d日")


class Date_Based_Groups():
    def __init__(self, date_str):
        self.input_date = datetime.strptime(date_str, "%Y年%m月%d日")
        self.timestamps = [
            # "1998年12月29日", # 1，0  第一位序号代表本时间点与前一个时间点组成的区间
            "1999年12月25日",   # 1      两头都有的按照区间的前闭后开算
            "2001年8月31日",    # 2
            "2001年12月29日",   # 3
            "2002年12月28日",   # 4
            "2005年2月28日",    # 5
            "2006年6月29日",    # 6
            "2009年2月28日",    # 7
            "2011年5月1日",     # 8
            "2015年11月1日",    # 9
            "2017年11月4日",    # 10
            "2021年3月1日",     # 11
            "2024年3月1日"      # 12
        ]
        self.dates = [datetime.strptime(date, "%Y年%m月%d日") for date in self.timestamps]

    def get_group_num(self):
        if self.input_date < self.dates[0]:
            return 1
        elif self.input_date > self.dates[-1]:
            return 13
        else:
            for i in range(len(self.dates) - 1):
                if self.dates[i] <= self.input_date <= self.dates[i + 1]:
                    if self.input_date == self.dates[i]:
                        return i + 1 + 1
                    elif self.input_date == self.dates[i + 1]:
                        return i + 2
                    else:
                        return i + 2
                    break


def flt_invalid_times(case):
    crime_time_raw = case["犯罪时间"].replace("<end_of_turn>", "")
    crime_time = re.findall(RE_DATE, crime_time_raw)
    if crime_time:
        return crime_time[-1]
    else:
        return []
    
def if_date_sane(date_str):
    try:
        date = Date_Based_Groups(date_str)
        return True
    except (ValueError, TypeError):
        return False
    
    
def ext_judgment_time(doc_content: str):
    formatted_fact = doc_content.replace("\n", "").replace(" ", "").replace("　", "").replace("\t", "")
    try:
        for date_regex in regex_judgment_date:
            judgment_date = re.search(date_regex, formatted_fact)
            if judgment_date:
                break
        if not judgment_date:
            return ""
        else:
            
            year = judgment_date.group(1)
            month = judgment_date.group(2)
            day = judgment_date.group(3)
            return f"{year}{UNBOTHERED_CONNECTION_TOKEN}{month}{UNBOTHERED_CONNECTION_TOKEN}{day}"
    except:
        return ""


def cvt_gibberish(extracted_judgment_date):
    formatted_str = extracted_judgment_date.replace("〇", "零"). \
            replace("○", "零"). \
            replace("0", "零"). \
            replace("０", "零"). \
            replace("Ｏ", "零"). \
            replace("Ο", "零"). \
            replace("O", "零"). \
            replace("О", "零"). \
            replace("o", "零"). \
                replace("-", "一").replace("+", "十").replace("元", "一")
    # print(extracted_judgment_date, formatted_str)
    year = formatted_str.split(UNBOTHERED_CONNECTION_TOKEN)[0]
    month = formatted_str.split(UNBOTHERED_CONNECTION_TOKEN)[1]
    day = formatted_str.split(UNBOTHERED_CONNECTION_TOKEN)[2]
    if len(year) < 4:
        if year[0] in LAST_CENTURY_SYMBOLS:
            year = "一九"[:4-len(year)] + year
        elif year[0] in CURRENT_CENTURY_SYMBOLS:
            year = "二零"[:4-len(year)] + year
    result = f"{year}{UNBOTHERED_CONNECTION_TOKEN}{month}{UNBOTHERED_CONNECTION_TOKEN}{day}"
    
    return result


def cvt_year_to_num(year_str):
    return year_str.replace("零", "0").replace("一", "1").replace("二", "2").replace("三", "3").replace("四", "4").replace("五", "5").replace("六", "6").replace("七", "7").replace("八", "8").replace("九", "9")


def if_judgment_and_crime_date_sane(doc_content, extracted_crime_date):
    current_judgment_time = ext_judgment_time(doc_content)
    # print(current_judgment_time)
    try:
        if current_judgment_time:
            formatted_judgment_time = cvt_gibberish(current_judgment_time)
            date_triplets = formatted_judgment_time.split(UNBOTHERED_CONNECTION_TOKEN)
            year_num_str = cvt_year_to_num(date_triplets[0])
            month_num_str = str(hanzi_to_num(date_triplets[1]))
            day_num_str = str(hanzi_to_num(date_triplets[2]))
            judgment_date_str = f"{year_num_str}年{month_num_str}月{day_num_str}日"
            # print(extracted_crime_date)
            # print(judgment_date_str)
            judgment_date = datetime.strptime(judgment_date_str, "%Y年%m月%d日")
            crime_date = datetime.strptime(extracted_crime_date, "%Y年%m月%d日")
            if judgment_date > crime_date:
                return True
            else:
                False
        else:
            return False
    except:
        return False
    
    
def if_under_new_system(extracted_crime_date):
    crime_date = datetime.strptime(extracted_crime_date, "%Y年%m月%d日")
    if crime_date >= EARLIEST_ACCEPTABLE_DATE:
        return True
    else:
        return False


def FLT_invalid_times():
    invalid_count = 0
    total_count = 0
    every_periods = [[] for _ in range(NMB_PERIODS)]
    for year in range(EARLIEST_YEAR, LATEST_YEAR + 1):
        for month in range(JANUARY, DECEMBER + 1):
            print(f"Currently doing {year}-{month}")
            current_file_path = f"date_extracted_cases_qwen/{year}/{year}-{month:02}.json"
            if os.path.exists(current_file_path):
                current_content = get_json_content(current_file_path)
                for case in current_content:
                    current_crime_time = flt_invalid_times(case)
                    if not if_date_sane(current_crime_time):
                        continue
                    if not current_crime_time:
                        continue
                    else:
                        if_correct_ordered_date_extracted = if_judgment_and_crime_date_sane(case["文书内容"], current_crime_time)
                        if_correct_system = if_under_new_system(current_crime_time)
                        if if_correct_ordered_date_extracted and if_correct_system:
                            current_date = Date_Based_Groups(current_crime_time)
                            current_group = current_date.get_group_num()
                            case["犯罪时间"] = current_crime_time
                            every_periods[current_group - 1].append(case)
                            # print(current_crime_time)
    for idx, list in enumerate(every_periods):
        current_file_path = f"cases_split_according_to_cla_dates/group_number_{idx + 1}.json"
        set_json_file(list, current_file_path, indent=4)
                    
                    
def TST_date_comparison_mechanism():
    test_0 = Date_Based_Groups("1997年11月3日")
    print(test_0.get_group_num())
    test_1 = Date_Based_Groups("1998年11月2日")
    print(test_1.get_group_num())
    test_2 = Date_Based_Groups("1998年12月29日")
    print(test_2.get_group_num())
    test_3 = Date_Based_Groups("1999年6月15日")
    print(test_3.get_group_num())
    test_4 = Date_Based_Groups("2001年8月31日")
    print(test_4.get_group_num())
    test_5 = Date_Based_Groups("2006年9月3日")
    print(test_5.get_group_num())
    test_6 = Date_Based_Groups("2023年7月31日")
    print(test_6.get_group_num())
    test_7 = Date_Based_Groups("2025年3月1日")
    print(test_7.get_group_num())
    
                        

if __name__ == "__main__":
    FLT_invalid_times()
    # TST_date_comparison_mechanism()
