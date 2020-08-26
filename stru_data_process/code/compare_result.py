# coding: utf-8

"""
实现标注数据与AI 数据的配置

"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import cv2


def get_data(data_path, pre_path, gt_path, compare_doctor_floder):

    for pre_file in os.listdir(pre_path):
        if pre_file in os.listdir(gt_path):
            data_file_path = os.path.join(data_path, pre_file)
            pre_file_path = os.path.join(pre_path, pre_file)
            gt_file_path = os.path.join(pre_path, pre_file)
            pre_data = cv2.imread(pre_file_path, 0)
            gt_data = cv2.imread(gt_file_path, 0)
            data = cv2.imread(data_file_path)
            pre_con = deal_data(pre_data)
            gt_con = deal_data(gt_data)

            cv2.drawContours(data, pre_con, -1, (0, 0, 255), 4)  # 红色
            cv2.drawContours(data, gt_con, -1, (255, 255, 0), 4)  # 黄色
            # rgb
            # bgr opencv

            cv2.imwrite(f'{compare_doctor_floder}/{pre_file}', data)

            exit()


def deal_data(img):
    # 二值化处
    # print(img.shape)
    binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 255, 255, cv2.THRESH_BINARY)
    # print(binary.shape)
    # exit()
    binary = np.array(binary, np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


if __name__ == "__main__":
    data_root = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/lave"
    pre_root = "/dicom/Jone/Dection_mass/test_data/test0/Doctor_result_AI_"
    gt_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/ann"
    compare_doctor_root = "/dicom/Jone/code/data_pre/stru_data_process/same/compare"

    get_data(data_root, pre_root, gt_root, compare_doctor_root)






