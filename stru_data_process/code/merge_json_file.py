"""
# 合并字典
"""

import json
import os


# save json file, obj 目标数据
def save_json(obj, folder, name):
    with open(folder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


# load json file
def load_json(file_json):
    with open(file_json, 'r', encoding='utf8') as f:
        data = json.load(f)
        return data


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 创建文件夹
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test1'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test2'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test3'
    # save_json_name = 'all_test1_json'
    # save_json_name = 'all_test1_nor_json'
    # save_json_name = 'all_test2_json'
    # save_json_name = 'all_test2_nor_json'
    # save_json_name = 'all_test3_json'
    # save_json_name = 'all_test3_nor_json'

    # shuffle data
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1'
    # save_json_name = 'all_test1_json'
    #
    # create_dir_not_exists(save_json_path)
    # create_dir_not_exist(save_json_name)

    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1'
    # save_json_name = 'all_test1_nor_json'
    #
    # create_dir_not_exists(save_json_path)
    # create_dir_not_exist(save_json_name)

    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2'
    # save_json_name = 'all_test2_json'
    #
    # create_dir_not_exists(save_json_path)
    # create_dir_not_exist(save_json_name)

    save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2'
    save_json_name = 'all_test2_nor_json'

    create_dir_not_exists(save_json_path)
    create_dir_not_exist(save_json_name)

    # 01 al
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/first/first_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/check_json.json"

    # 01 nor
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/first/first_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/check_nor_json.json"

    # 02 al
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/all_test1_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/check_json.json"

    # 02 nor
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/all_test1_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/check_nor_json.json"

    # 03 al
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/all_test2_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/check_json.json"

    # 03 nor
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/all_test2_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/check_nor_json.json"

    # shuffle al
    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/first/first_json.json"
    # # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/check_json.json"
    #
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/check_nor_json.json"

    json_0 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/all_test1_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/check_json.json"

    json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/check_nor_json.json"

    dict_all = {}  # 定义一个空字典

    dict0 = load_json(json_0)
    dict1 = load_json(json_1)

    a = dict0["images"]
    b = dict1["images"]

    a_image = a + b

    al = dict0["annotations"]
    bl = dict1["annotations"]

    al_ann = al + bl

    # categories
    ac = dict0["categories"]

    ac_ca = ac

    dict_all["images"] = a_image
    dict_all["annotations"] = al_ann
    dict_all["categories"] = ac_ca

    dict_all = json.dumps(dict_all)
    save_json(dict_all, save_json_path, save_json_name)

    print("save success!!")