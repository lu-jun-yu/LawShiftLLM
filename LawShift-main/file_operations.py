import json
import csv


def get_json_content(file_path):
    with open(file_path, 'r') as f:
        content = json.load(f)
    f.close()
    return content


def get_json_in_txt_content(file_path):
    file = open(file_path, 'r')
    lines = []
    for line in file.readlines():
        # dic = json.loads(line)
        lines.append(line)

    file.close()
    # lines为一个元素为字典的列表
    return lines


def set_json_file(content, file_path, indent=4):
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=indent)
    f.close()


# content为一个元素为字典的列表
# def set_txt_file_dic_each_line(content, file_path):
#     f = open(file_path, 'w')
#     for line in content:
#         f.write(json.dumps(line, ensure_ascii=False))
#         f.write("\n")
#     f.close()


# def get_csv_content(file_path):
#     data = []
#     with open(file_path) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         for row in csv_reader:
#             data.append(row)
#     return data


def get_jsonl_content(file_path):
    content = []
    with open(file_path, "r") as file:
        for line in file:
            obj = json.loads(line)
            content.append(obj)
    return content


def set_jsonl_file(content, file_path):
    with open(file_path, "w") as file:
        for obj in content:
            line = json.dumps(obj, ensure_ascii=False)
            file.write(line + "\n")


def get_txt_content(file_path):
    """
    输入文件路径，返回每行内容，列表形式
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines