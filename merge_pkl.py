# coding: utf-8
import _pickle as cPickle

import pickle


def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


pkl0 = open(r'/dicom/Jone/code/data_pre/stru_data_process/save_data/diff/train_ann_file/differ-learn.pkl', 'rb')   # 二进制格式读文件
pkl1 = open(r'/dicom/Jone/code/data_pre/stru_data_process/same/liu/train_ann_file/differ-learn.pkl', 'rb')
pkl2 = open(r'/dicom/Jone/code/data_pre/stru_data_process/same/wan1/train_ann_file/differ-learn.pkl', 'rb')
pkl3 = open(r'/dicom/Jone/code/data_pre/stru_data_process/same/wan2/train_ann_file/differ-learn.pkl', 'rb')
pkl4 = open(r'/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_ann_file/differ-learn.pkl', 'rb')
# 使用python的cPickle库中的load函数，可以读取pkl文件的内容
inf0 = cPickle.load(pkl0)
inf1 = cPickle.load(pkl1)
inf2 = cPickle.load(pkl2)
inf3 = cPickle.load(pkl3)
inf4 = cPickle.load(pkl4)

inf_all = inf0 + inf1 + inf2 + inf3 + inf4
f = "/dicom/Jone/Dection_mass/merge_data/01-test"
pkl_name = "/all_learn"
save_pkl(inf_all, f, pkl_name)

# print(inf_all)
