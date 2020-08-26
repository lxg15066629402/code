# coding: utf-8

"""
获取数据，在所有数据中，找出非主动学习的数据

"""
import os
import numpy as np
import shutil
import json


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data(org_folder, bst_folder):
    nor_al_name = []

    # print(sorted(os.listdir(org_folder))[:192])
    # exit()
    # for index, file_name in enumerate(sorted(os.listdir(org_folder))[:115]):
    # for index, file_name in enumerate(sorted(os.listdir(org_folder))[:200]):  # 第二次
    # for index, file_name in enumerate(sorted(os.listdir(org_folder))[:107]):  # 第三次
    # shuffle data
    for index, file_name in enumerate(sorted(os.listdir(org_folder))[:200]):  # 第三次

        nor_al_name.append(file_name)
        org_path = os.path.join(org_folder, file_name)
        bst_path = os.path.join(bst_folder, file_name)

        shutil.copyfile(org_path, bst_path)

    return nor_al_name


def save_json(get_data):
    # print(len(get_data))
    jsonArr = json.dumps(get_data, ensure_ascii=False)
    # with open("/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/nor.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/nor.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/nor.json", 'w') as f:
    # shuffle data
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/nor.json", 'w') as f:
    with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/nor.json", 'w') as f:
        # for line in jsonArr:
        f.write(jsonArr)
        f.close()


if __name__ == "__main__":
    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/lave"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/nor"

    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/lave"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/nor"

    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/lave"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/nor"

    # shuffle data
    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/lave"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/nor"

    org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/lave"
    bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/nor"

    create_dir_not_exists(bst_root)
    data_name = get_data(org_root, bst_root)
    save_json(data_name)




