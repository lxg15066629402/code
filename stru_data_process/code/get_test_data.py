# coding: utf-8

"""
获取数据，在所有数据中, 随机找出一部分数据进行测试

"""
import os
import numpy as np
import shutil
import json
from random import shuffle


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data(org_folder, bst_folder):
    nor_al_name = []
    # for index, file_name in enumerate(os.listdir(org_folder)[:150]):  # 01
    # for index, file_name in enumerate(os.listdir(org_folder)[:184]):  # 02
    # for index, file_name in enumerate(os.listdir(org_folder)[184:368]):  # 03
    # for index, file_name in enumerate(os.listdir(org_folder)[368:552]):  # 04
    for index, file_name in enumerate(os.listdir(org_folder)[552:736]):  # 05
        nor_al_name.append(file_name)
        org_path = os.path.join(org_folder, file_name)
        bst_path = os.path.join(bst_folder, file_name)

        shutil.copyfile(org_path, bst_path)

    return nor_al_name


def save_json(get_data):
    # print(len(get_data))
    jsonArr = json.dumps(get_data, ensure_ascii=False)
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test/test.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/test.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_2/test.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_3/test.json", 'w') as f:
    with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_4/test.json", 'w') as f:
        # for line in jsonArr:
        f.write(jsonArr)
        f.close()


if __name__ == "__main__":
    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test/data"

    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"

    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_2/data"

    # org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_3/data"

    org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_4/data"

    create_dir_not_exists(bst_root)
    data_name = get_data(org_root, bst_root)
    save_json(data_name)




