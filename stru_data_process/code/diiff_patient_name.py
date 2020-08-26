# coding: utf-8

'''
 判断两个医生标注的数据是否有重复的病例
'''

import os


def search_patient(folder1, folder2):
    # folder1_list = []
    # folder2_list = []
    # for index1, patient1 in enumerate(os.listdir(folder1)):
    #     folder1_list.append(patient1)
    #
    # for index2, patient2 in enumerate(os.listdir(folder2)):
    #     folder2_list.append(patient2)
    #
    # return

    folder_list = []
    i = 0
    for patient1 in os.listdir(folder1):
        for patient2 in os.listdir(folder2):
            if patient1 == patient2:
               i += 1
               folder_list.append(patient1)

    return i, folder_list


if __name__ == "__main__":
    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/liu_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/yao_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/liu_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/liu_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/yao_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/yao_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/diff_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_2/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_2/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_2/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_4/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_4/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/diff_json1/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/data"

    # root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/first/data"
    # root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"

    root1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/nor_data"
    root2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/test_1/data"

    num, same_name = search_patient(root1, root2)

    print(num, same_name)