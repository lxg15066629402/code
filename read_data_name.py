# coding: utf-8

'''
将数据名存到文件中

'''

import os
import numpy as np


def save_name(path, type):
    with open(os.path.join(path, 'train_' + type + '_.txt'), "a") as f:
        for label in sorted(os.listdir(os.path.join(path, type)), key=lambda x: int(x[:8])):
            f.write(os.path.join(path, type, label))
            f.write("\n")
    f.close()


# read txt file
def read_file(file_path):
    # image_list = [line.strip() for line in open(file_path, 'r')]  # 去掉 \n
    image_list = [line.strip() for line in open(file_path, 'r')]
    print(image_list)
    np.random.shuffle(image_list)  # 打乱顺序
    print(image_list)


if __name__ == "__main__":
    # root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data"
    # type_list = ['ann', 'data']
    # save_name(root, type_list[0])
    # save_name(root, type_list[1])
    txt_file_root = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/train_ann_.txt"
    read_file(txt_file_root)