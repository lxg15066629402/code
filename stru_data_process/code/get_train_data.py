# coding: utf-8
"""
# 进行文件的提取，如果在该文件中，则不提取，如果不在文件中，就将数据保存到另一个文件夹中
"""
import os
import shutil
import json


def search_file(sub_file, all_file):
    sub_file_name = []
    all_file_name = []
    lave_file_name = []
    for i in os.listdir(sub_file):
        sub_file_name.append(i)
    print(len(sub_file_name))

    for j in os.listdir(all_file):
        all_file_name.append(j)
    print(len(all_file_name))

    for filename in all_file_name:
        if filename not in sub_file_name:
            lave_file_name.append(filename)

    print(len(lave_file_name))
    return lave_file_name


def save_lave(org_root, lave_root, lave_file_name):

    for filename in lave_file_name:

        files = os.path.join(org_root, filename)
        lave_file = os.path.join(lave_root, filename)

        shutil.copyfile(files, lave_file)  # 文件的移动复制保存


def save_json(get_data):
    # print(len(get_data))
    jsonArr = json.dumps(get_data, ensure_ascii=False)
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train/train.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/train.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_2/train.json", 'w') as f:
    # with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_3/train.json", 'w') as f:
    with open("/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_4/train.json", 'w') as f:
        # for line in jsonArr:
        f.write(jsonArr)
        f.close()


if __name__ == "__main__":

    # 获取训练的结果
    # test_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # train_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train/data"

    # test_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # train_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/data"
    #
    # test_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_2/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # train_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_2/data"

    # test_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_3/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    # train_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_3/data"

    test_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_4/data"
    all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/data"
    train_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_4/data"

    lave_info = search_file(test_folder, all_folder)
    save_lave(all_folder, train_folder, lave_info)
    save_json(lave_info)