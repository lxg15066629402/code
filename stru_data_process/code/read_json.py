# coding: utf-8
"""
 读取数据，找出 ann 标注的id 起始位置

"""

import json


# save json file, obj 目标数据
def save_json(obj, folder, name):
    with open(folder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


def load_json(json_file):
    # 读取json文件内容,返回字典格式
    with open(json_file, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        # print(json_data)
    return json_data


if __name__ == "__main__":

    json_root_1 = '/dicom/Jone/code/data_pre/stru_data_process/same/yao_json1/annotations/yao_json.json'
    # json_root_1 = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/coco_1_json.json'
    # json_root_2 = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/liu_1_json.json'
    # json_root_3 = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/wan1_1_json.json'
    # json_root_4 = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/wan2_2_json.json'
    # json_root_5 = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/yao_1_json.json'

    dict_1 = load_json(json_root_1)
    # dict_2 = load_json(json_root_2)
    # dict_3 = load_json(json_root_3)
    # dict_4 = load_json(json_root_4)
    # dict_5 = load_json(json_root_5)

    print(len(dict_1["images"]))
    print(len(dict_1["annotations"]))

    # print(len(dict_2["images"]))
    # print(len(dict_2["annotations"]))

    # print(len(dict_3["images"]))
    # print(len(dict_3["annotations"]))

    # print(len(dict_4["images"]))
    # print(len(dict_4["annotations"]))

    # print(len(dict_5["images"]))
    # print(len(dict_5["annotations"]))
