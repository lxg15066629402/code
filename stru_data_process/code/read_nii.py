# coding: utf-8


# read nii file

import nibabel as nii

a = nii.load("/dicom/Adam/data/structural/raw_data/20200619/double_same/liuyuanyuan_vs_shangxiaojing/"
             "20130426_1979555/0002_000001_2.16.840.1.113669.632.25.1.610314.20130426093635.shangxiaojing.nii.gz")

# a_img = a.get_data()
# print(a.shape)


# 最长子序列
len1 = 4
len2 = 5
res = [[0 for i in range(len1+1)] for j in range(len2+1)]
print(res)
print(len(res))

'''
动态规划问题
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0]]
'''
