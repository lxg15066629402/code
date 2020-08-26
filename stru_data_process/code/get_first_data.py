# coding: utf-8

"""
获取数据，在所有数据中，选出一部分数据进行首次训练，得宠model1

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

    # print(shuffle(os.listdir(org_folder)))
    # print(os.listdir(org_folder))
    # exit()
    # for index, file_name in enumerate((shuffle(os.listdir(org_folder)))[:100]):
    for index, file_name in enumerate(os.listdir(org_folder)[:100]):
        nor_al_name.append(file_name)
        org_path = os.path.join(org_folder, file_name)
        bst_path = os.path.join(bst_folder, file_name)

        shutil.copyfile(org_path, bst_path)

    return nor_al_name


def save_json(get_data):
    # print(len(get_data))
    jsonArr = json.dumps(get_data, ensure_ascii=False)
    # with open("/dicom/Jone/code/data_pre/stru_data_process/test_al/first/first.json", 'w') as f:
    with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/first/first.json", 'w') as f:
        # for line in jsonArr:
        f.write(jsonArr)
        f.close()


if __name__ == "__main__":
    org_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/data"
    bst_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/first/data"

    create_dir_not_exists(bst_root)
    data_name = get_data(org_root, bst_root)
    save_json(data_name)




