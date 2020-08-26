# coding: utf-8
# load pkl file
# 打开.pkl文件代码
# pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西。正确的操作方式是使用Pickle模块。Pickle模块将任意一个Python对象转换成一系统字节，这个操作过程叫做串行化对象。


import _pickle as cPickle

import pickle

# get a pkl file
# f = open(r'/dicom/Jone/code/data_pre/stru_data_process/save_data/diff/train_ann_file/differ-learn.pkl', 'rb')   # 二进制格式读文件
f = open(r'/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_ann_file/differ-learn.pkl', 'rb')   # 二进制格式读文件
# f = open(r'/dicom/Jone/code/data_pre/stru_data_process/train_ann_file/differ-learn.pkl', 'rb')   # 二进制格式读文件
# 使用python的cPickle库中的load函数，可以读取pkl文件的内容
inf = cPickle.load(f)
# print(inf)
# print((len(inf)))
# print(type(inf[0]['filename']))
name = []
c = 0
for i in range(len(inf)):
    if inf[i]['filename'] in name:
        print(inf[i]['filename'])
        c += 1
    name.append(inf[i]['filename'])
print(c)
print(len(inf))

# print(type(inf))  # <class 'list'>
# print(len(inf[:120]))
# a = inf[:120]
# # 保存 再打开一个txt文件，向内写入刚才读取的信息
# # inf = str(a)
# # print(len(a))
# # print(a)
# print(type(a))  # <class 'list'>
# print((a[0]))  # 文件 + 标注信息
#
# # {'filename': '20110224_1587675_LCC.png', 'width': 2016, 'height': 2816, 'ann': {'bboxes': array([[  16., 1420.,  504., 1876.], [ 212., 1142.,  442., 1312.]], dtype=float32), 'labels': array([0, 0])}}
# # 使用cPickle.dump来将对象(a)序列化到文件
#
# with open('trian_filename.pkl', 'wb') as f:
#     pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)

# save txt
# with open('test_2.txt', 'w') as f:
#     for line in a:
#         f.write(line + '\n')
# f.close()



