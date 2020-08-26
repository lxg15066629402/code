# coding: utf-8

"""分析数据"""

import json


def load_json(json_file):
    # 读取json文件内容,返回字典格式
    with open(json_file, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        # print(json_data)
    return json_data


def get_check(check_json):

    check_data = load_json(check_json)
    check_all_file = []
    for i, index in enumerate(check_data):
        # print(index[0])
        check_all_file.append(index[0])

    print(len(check_all_file))
    return check_all_file


if __name__ == "__main__":
    check_json = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/check.json"
    get_check(check_json)