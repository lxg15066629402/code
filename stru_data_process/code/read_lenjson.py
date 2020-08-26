# coding: utf-8
"""
 读取json数据，找出它的长度

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

    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/all_test1_nor_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/all_test1_nor_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/all_test2_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/all_test2_nor_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/all_test3_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/all_test3_nor_json.json'


    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/coco_1_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/liu_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/wan1_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/wan2_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/yao_json.json'
    # json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test/test_json.json'
    json_root = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_4/test_json.json'

    dict_ = load_json(json_root)

    print(len(dict_["images"]))
    print(len(dict_["annotations"]))
