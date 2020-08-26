# coding: utf-8

"""
获取路径下的文件个数
"""

import os
path = "/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_data_png"

a = len(os.listdir(path))
print(a)