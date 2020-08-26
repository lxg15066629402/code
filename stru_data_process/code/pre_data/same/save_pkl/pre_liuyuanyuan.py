"""
# 读取刘医生组的数据，刘医生组的数据中，有医生是单序列标注，另外的一位医生是连续序列进行标注
# 处理的思路是：
# 首先根据文件名分别读取出两位医生对应的dcm 文件和 dicom 文件，之后对应的序列进行添加

"""

import nibabel as nib
import cv2
import numpy as np
import pydcm2png
import os
import pydicom
import pandas as pd
import pickle
import json
import time
from shapely.geometry import Polygon  # 多边形
import scipy.io as io
import shutil


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pkl(obj, folder, name):
    with open(folder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# 读取dcm 文件
def dcm2png(dcm_file):
    '''
    将dcm文件转为png文件，并获取dcm信息
    :param dcm_folder_path:例如20110304（该文件夹的目录为20110304/ExamImage/images.dcm）
    :return: png_dic 形式为png_dic[索引] = [拍摄位,img]，保存img(保存在png_img/20110304/images.png）
    '''

    img = pydcm2png.get_pixel_data(dcm_file)

    # 获取拍摄位信息
    ds = pydicom.read_file(dcm_file)

    ImageLaterality = ds.ImageLaterality

    ViewPosition = ds.ViewPosition

    pos = ImageLaterality + ViewPosition

    return pos, img


# 读取 nii file
def get_one_nii(nii_file):
    # 加载nii文件
    img_data = nib.load(nii_file)
    img = img_data.get_fdata()

    return img


# 获取bounding box
def get_one_nii_bounding_box(img, save_mask=False):
    '''
    解析nii文件的信息
    :param nii_floder:
    :param nii_name:
    :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
             contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
    '''

    bounding_box = []
    contours_list = []
    mask_list = []
    for i in range(int(img.shape[-1])):
        # 获取掩码信息
        # 注意：nibabel读出的image的data的数组顺序为：Width，Height，Channel
        # SimpleITK读出的image的data的数组顺序为：Channel,Height，Width
        mask = np.transpose(img[:, :, i])  # 在标注保存时掩码进行来转置，因此这里需要transpose

        # 二值化处理
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        binary = np.array(binary, np.uint8)
        mask_list.append(binary)
        # 获取掩码边
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_list.append(contours)

        new_contours = []
        # 获取最大边框
        if contours != []:
            box = []  # 存在一张图中有多个结构扭曲区域，i代表结构扭曲个数
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > 8:  #
                    x_max = int(np.max(contours[i][:, :, 0]))
                    y_max = int(np.max(contours[i][:, :, 1]))
                    x_min = int(np.min(contours[i][:, :, 0]))
                    y_min = int(np.min(contours[i][:, :, 1]))
                    box.append([x_min, y_min, x_max, y_max])
                    new_contours.append(contours[i])
            bounding_box.append(box)
            contours_list.append(new_contours)
        else:
            bounding_box.append([[0, 0, 0, 0]])
            contours_list.append(new_contours)

    if save_mask == True:
        return bounding_box, contours_list, mask_list
    else:
        return bounding_box, contours_list


# dcm 文件与 nii 文件进行对应
def dcm_nii_con(bounding_box_list, png_dic, contours_list, folder, png_path, ann_png_path, save=True, draw=True,
                mask_list=None, save_mask=False):

    """

    :param bounding_box_list:
    :param png_dic:
    :param contours_list:
    :param floder:
    :param png_path:
    :param ann_png_path:
    :param save:
    :param draw:
    :param mask_list:
    :param save_mask:
    :return:
    """

    label_list = []
    png_list = []

    for index in range(len(bounding_box_list)):
        dict_data = {}
        ann = {}
        # 获取对应信息
        coords = bounding_box_list[index]
        if coords == [[0, 0, 0, 0]]:  # 未标注数据不进行保存
            continue

        contours = contours_list[index]
        pos, image = png_dic  #

        # 文件命名
        filename = folder + '_' + pos + '.png'

        height, width = image.shape
        # 在得到bboxes， labels 的时候需要注意对应类型，该类型要与深度学习所使用的框架对应，同时根据使用的 开发环境的操作系统的内存，也有关系，
        # 这时，需要进行更大的关注
        # bboxes = np.array(coords, dtype=np.float64)
        bboxes = np.array(coords, dtype=np.float32)

        # labels = np.ones(len(bboxes), dtype=np.int64)   # **
        labels = np.zeros(len(bboxes), dtype=np.int64)   # **

        # 是否保存掩码信息
        if save_mask == True:
            mask = np.array(mask_list, dtype=np.int64)  # **
            ann['mask'] = mask[index]

        # 标签填充
        ann['bboxes'] = bboxes
        ann['labels'] = labels

        dict_data['filename'] = filename
        dict_data['width'] = width
        dict_data['height'] = height
        dict_data['ann'] = ann

        label_list.append(dict_data)
        png_list.append(image)

        # 保存dcm2png 文件
        if save == True:
            cv2.imwrite(f"{png_path}/{filename}", image)

        # 绘制标签文件
        if draw == True:
            # print(len(coords))
            for i in range(len(coords)):
                if len(contours) != len(coords):
                    continue
                else:
                    cv2.rectangle(image, (coords[i][0], coords[i][1]),
                                  (coords[i][2], coords[i][3]), color=60000, thickness=5)
                    cv2.drawContours(image, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(ann_png_path, filename), image)

    return label_list, png_list


# 合并两个字典数据
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def merge_nii(doctor0_nii, doctor1_nii):
    # iou_all = cv2.bitwise_or(doctor0_nii['img'], doctor1_nii['img'])
    # iou_all = np.add(doctor0_nii['img'], doctor1_nii['img'])
    # if doctor0_nii.shape != doctor1_nii.shape:
    # iou_all = np.add(doctor0_nii, doctor1_nii)
    # print(doctor0_nii)
    # print(doctor1_nii)
    index_0 = doctor0_nii.keys()
    index_1 = doctor1_nii.keys()

    # print(index_0)
    # print(index_1)
    # exit()

    iou_all = {}
    # print(doctor0_nii[0]['img'])
    # exit()
    for index in index_1:
        # print(doctor0_nii[index]['img'].shape)
        # a = (doctor0_nii[index]['img'][:, :, np.newaxis])
        # print(a.shape)
        # exit()
        iou_all[index] = np.add(doctor0_nii[index]['img'][:, :, np.newaxis], doctor1_nii[index]['img'])

    # print(iou_all)
    add_index = iou_all.keys()
    # print(add_index)
    # remain_index = []
    iou_remain = {}
    remain_index = [x for x in index_0 if x not in add_index]
    # print(remain_index)
    # exit()
    for index in remain_index:
        iou_remain[index] = doctor0_nii[index]['img'][:, :, np.newaxis]

    # print(iou_remain)
    # exit()
    all_dict = Merge(iou_remain, iou_all)

    # 排序
    data = dict(sorted(all_dict.items(), key=lambda x: x[0]))
    return data


def compare_double(folders):

    # 获取病人的dcm和nii
    png_dic = {}
    nii_dic = {}
    # 获取病人的dcm和nii
    # exit()
    # print(folders)
    for file in os.listdir(folders):
        # print(file)
        # exit()
        end_name = file.split('.')[-1]
        type_name = int(file.split('_')[0])
        split_name = file.split('.')[-3]
        # nii_mask_index = int(file.split('_')[1])
        # print(end_name)
        # print(type_name)
        # print(split_name)
        # exit()

        # 获取dcm信息
        # if end_name == '3' and type_name == 2:  # 连续序列标注
        if end_name == '3' and type_name == 2:  # 连续序列标注
            dcm_index = int(file.split('_')[1])
            # print(dcm_index)
            # exit()
            dcm_file = os.path.join(folders, file)
            pos, img = dcm2png(dcm_file)
            png_dic[dcm_index-1] = [pos, img]

        # 获取nii信息。
        elif end_name == 'gz':

            # shangxiaojing or sxj是进行单标注
            if 'sxj' in split_name or 'shangxiaojing' in split_name:
                nii_file = os.path.join(folders, file)
                nii_type = int(file.split('_')[0])
                nii_mask_index = int(file.split('_')[1])
                if nii_type == 2:
                    if 'doctor_sxj' not in nii_dic.keys():
                        nii_dic['doctor_sxj'] = {}

                    img = get_one_nii(nii_file)
                    # nii_dic['doctor_sxj'][f'{nii_type}_{nii_mask_index - 1}'] = {'img': img}
                    nii_dic['doctor_sxj'][f'{nii_mask_index - 1}'] = {'img': img}

            elif split_name[-1] == 'A':  #
                nii_file = os.path.join(folders, file)
                if 'doctor_lyy' not in nii_dic.keys():
                    nii_dic['doctor_lyy'] = {}

                img = get_one_nii(nii_file)
                for i in range(img.shape[-1]):
                    nii_dic['doctor_lyy'][f'{i}'] = {'img': img[:, :, i]}

    # print(nii_dic['doctor_lyy'][f'{nii_type}_{nii_mask_index - 1}'])
    # exit()
    doctor_name = list(nii_dic.keys())
    # all_nii_dic = {}
    # print(nii_dic['doctor_lyy'].keys())
    # exit()
    if len(doctor_name) != 2:
        print("只进行了一次标注")

    elif len(doctor_name) == 2:
        all_nii_dic = merge_nii(nii_dic['doctor_lyy'], nii_dic['doctor_sxj'])
    # all_nii_dic = merge_nii(doctor_name[1], doctor_name[0])

        return png_dic, all_nii_dic


def main():

    # 初始化路径
    dcm_floders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/liuyuanyuan_vs_shangxiaojing'
    png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu/train_data_png'
    pkl_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu/train_ann_file'
    ann_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu/train_ann_png'
    pkl_name = '/differ-learn'

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)
    all_label_alllist = []
    n = 0

    for doctor in sorted(os.listdir((dcm_floders))):

        num = len(os.listdir(dcm_floders))
        n += 1
        # sorted() 排序
        patient_path = os.path.join(dcm_floders, doctor)
        # remove '.DS_Store'
        for i in os.listdir(patient_path):
            if i.split(".")[-1] == 'DS_Store':
                # print('remove')
                os.remove(os.path.join(patient_path, i))

        # if :
        # print(patient_path)
        png_dic, all_nii_dic = compare_double(patient_path)

        # else:
        #     continue
        # print(png_dic)
        # print(all_nii_dic)
        # exit()
        # print(png_dic[0])
        # print(all_nii_dic.keys())
        # exit()
        label_alllist = []
        for idx in list(all_nii_dic.keys()):  # 以label 为主
            # print(idx)
            # exit()

            bounding_box_list, contours_list = get_one_nii_bounding_box(all_nii_dic[idx])

            label_list, png_list = dcm_nii_con(bounding_box_list, png_dic[int(idx)], contours_list, doctor, png_path,
                                               ann_png_path)

            # 添加label
            label_alllist += label_list
        print(f'Save {doctor} successful,Residual:{num - n}')  # 打印输出结果
        all_label_alllist += label_alllist
    # 将label存储为pkl格式
    save_pkl(all_label_alllist, pkl_path, pkl_name)


if __name__ == '__main__':
    main()