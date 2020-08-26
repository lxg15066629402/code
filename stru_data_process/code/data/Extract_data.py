# coding: utf-8
"""
# 进行文件的提取，如果在该文件中，则不提取，如果不在文件中，就将数据保存到另一个文件夹中
"""
import os
import shutil


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


if __name__ == "__main__":

    # 获取剩余第一次训练的结果
    # sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/first/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/data"
    # lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/lave"

    # 获取第二次训练的结果
    # sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/check"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/lave"
    # lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/lave"

    # 获取第三次训练的结果
    # sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/check"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/lave"
    # lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/lave"

    # 获取第四次训练的结果
    # sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/check"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/lave"
    # lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test4/lave"

    # shuffle data
    # sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/first/data"
    # all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/data"
    # lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/lave"

    sub_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/check"
    all_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/lave"
    lave_folder = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/lave"

    lave_info = search_file(sub_folder, all_folder)
    save_lave(all_folder, lave_folder, lave_info)