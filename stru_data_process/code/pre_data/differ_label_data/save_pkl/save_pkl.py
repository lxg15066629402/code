# coding: utf-8
"""
# 读取dicom file ,nii file, 并实现数据与标签的对应
# 该批数据对应的是通过处理后双标不一致的数据，经过返回得到的数据
"""

import pydicom
import nibabel as nib
import numpy as np
import cv2
import pickle
import os
import pydcm2png


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load pkl file
def load_pkl(floder, name):
    with open(floder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# .dcm 文件 转化为 .png 文件
# 获取连续的dcm文件数据
def dcm_png(dcm_folder):
    """

    :param dcm_folder: input dcm file
    :return: png file
    """
    png_dic = {}  # 字典类型
    # sorted 排序
    for filename in sorted(os.listdir(dcm_folder)):

        if filename.split(".")[-1] == 'gz' or filename.split(".")[-1] == 'bdc-downloading':
            continue

        file_name = os.path.join(dcm_folder, filename)
        # 获取文件编号(用于和标签的匹配)
        png_name = filename.split(".")[:-1]

        index = int(png_name[0].split('_')[1])

        # dcm转png
        img = pydcm2png.get_pixel_data(file_name)  # 转化
        # print(img)

        # 读取.dcm文件，获取拍摄位信息
        ds = pydicom.read_file(file_name)

        ImageLaterality = ds.ImageLaterality

        ViewPosition = ds.ViewPosition

        pos = ImageLaterality + ViewPosition

        png_dic[index - 1] = [pos, img]  # 文件名序号是从1开始，index-1使其从0开始

    return png_dic


# .dcm 文件 转化为 .png 文件
# 获取单序列的dcm数据
def dcm_png_single(dcm_folder, name_style):

    """
    :param dcm_folder: input dcm file
    :return: png file
    """
    png_dic = {}
    for i, value in enumerate(name_style):
        # print(name_style[0])
        filename = str(value[:-10])  # 检索文件名

        file_name = os.path.join(dcm_folder, filename)

        img = pydcm2png.get_pixel_data(file_name)  # 转化

        # 读取.dcm文件，获取拍摄位信息
        ds = pydicom.read_file(file_name)

        ImageLaterality = ds.ImageLaterality

        ViewPosition = ds.ViewPosition

        pos = ImageLaterality + ViewPosition

        png_dic[i] = [pos, img]   #

    return png_dic


# 读取nii 文件，并获得标注区域信息
def get_nii_bounding_box(nii_floder, save_mask=False):
    '''
        解析nii文件的信息
        :param nii_floder:
        :param nii_name:
        :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
                 contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
    '''
    single_nii = []
    for nii_file in os.listdir(nii_floder):
        nii_file_name = os.path.join(nii_floder, nii_file)
        if nii_file.split('.')[-3] == 'ZB':
            img_data = nib.load(nii_file_name)
            img = img_data.get_fdata()
            # print(img.shape)
            # 如果是单序列标注
            if img.shape[-1] == 1:
                single_nii.append(nii_file)
            a = single_nii
            bounding_box = []
            contours_list = []
            mask_list = []
            for i in range(img.shape[-1]):
                # 在标注保存时掩码进行来转置，因此这里需要transpose
                mask = np.transpose(img[:, :, i])
                # 二值化处理
                ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                # 转化类型
                binary = np.array(binary, np.uint8)
                mask_list.append(binary)

                #  获取mask信息
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print(len(contours))
                # contours_list.append(contours)

                new_contours = []
                # 获取最大边框
                if contours != []:
                    box = []
                    # 存在一张图中有多个结构扭曲区域，i 表示结构扭曲个数
                    for i in range(len(contours)):
                        area = cv2.contourArea(contours[i])
                        if area > 8:  # 区别信息
                            x_max = int(np.max(contours[i][:, :, 0]))
                            y_max = int(np.max(contours[i][:, :, 1]))
                            x_min = int(np.min(contours[i][:, :, 0]))
                            y_min = int(np.min(contours[i][:, :, 1]))
                            box.append([x_min, y_min, x_max, y_max])
                            new_contours.append(contours[i])  # 一张图中的结构扭曲个数
                    bounding_box.append(box)
                    contours_list.append(new_contours)
                else:
                    bounding_box.append([[0, 0, 0, 0]])  # 如何没有结构扭曲，则进行补充为 [[0, 0, 0, 0]
                    contours_list.append(new_contours)

            if save_mask == True:  # 保存掩码
                # return bounding_box, contours_list, mask_list, single_nii
                return bounding_box, contours_list, mask_list, a
            else:  # 不保存掩码
                # return bounding_box, contours_list, single_nii
                return bounding_box, contours_list, a


# nii 文件与con 文件进行合并操作
def dcm_nii_con(bounding_box_list, png_dic, contours_list, floder, png_path, ann_png_path, save=True, draw=True,
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
    for index in range(len(bounding_box_list)):  # .nii.gz 多少层
        # print(len(bounding_box_list))
        dict_data = {}
        ann = {}
        # 获取对应信息
        coords = bounding_box_list[index]
        if coords == [[0, 0, 0, 0]]:  # 未标注数据不进行保存
            continue
        contours = contours_list[index]
        pos, image = png_dic[index]  #

        # 文件命名
        filename = floder + '_' + pos + '.png'

        height, width = image.shape
        # bboxes = np.array(coords, dtype=np.float64)
        bboxes = np.array(coords, dtype=np.float32)

        # labels = np.ones(len(bboxes), dtype=np.int64)   # **
        labels = np.zeros(len(bboxes), dtype=np.int64)   # **
        #
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
            for i in range(len(coords)):
                if len(contours) != len(coords):
                    continue
                else:
                    cv2.rectangle(image, (coords[i][0], coords[i][1]), (coords[i][2], coords[i][3]),
                              color=60000, thickness=5)
                    cv2.drawContours(image, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(ann_png_path, filename), image)

    return label_list, png_list


def main():
    # 初始化路径
    dcm_floders = '/dicom/Adam/data/structural/raw_data/20200619/double_differ'
    nii_floders = '/dicom/Adam/data/structural/raw_data/20200619/double_differ'
    png_path = '/dicom/Jone/code/data_pre/stru_data_process/save_data/diff/train_data_png'
    pkl_path = '/dicom/Jone/code/data_pre/stru_data_process/save_data/diff/train_ann_file'
    ann_png_path = '/dicom/Jone/code/data_pre/stru_data_process/save_data/diff/train_ann_png'
    pkl_name = '/differ-learn'

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)
    all_label_alllist = []

    for doctor in sorted(os.listdir((dcm_floders))):
        label_alllist = []
        n = 0
        num = len(os.listdir(os.path.join(dcm_floders, doctor)))

        # sorted() 排序
        patient_path = os.path.join(dcm_floders, doctor)
        # remove '.DS_Store'
        for i in os.listdir(patient_path):
            if i.split(".")[-1] == 'DS_Store':
                os.remove(os.path.join(patient_path, i))

        for floder in sorted(os.listdir(patient_path)):  # 标签与数据对应

            n += 1
            dcm_floder = os.path.join(patient_path, floder)
            nii_floder = os.path.join(patient_path, floder)

            # 处理nii文件
            bounding_box_list, contours_list, single_nii_name = get_nii_bounding_box(nii_floder)

            # 匹配dcm和nii文件的信息
            # print(len(bounding_box_list))
            if len(bounding_box_list) > 1:
                png_dic = dcm_png(dcm_floder)
                # print(len(png_dic))
                label_list, png_list = dcm_nii_con(bounding_box_list, png_dic, contours_list, floder,
                                                   png_path, ann_png_path)
                label_alllist += label_list

            elif len(bounding_box_list) == 1:
                png_dic = dcm_png_single(dcm_floder, single_nii_name)
                label_list, png_list = dcm_nii_con(bounding_box_list, png_dic, contours_list, floder,
                                                          png_path, ann_png_path)

                label_alllist += label_list
            print(f'Save {floder} successful,Residual:{num - n}')  # 打印输出结果

        all_label_alllist += label_alllist

    # 将label存储为pkl格式
    # save_pkl(label_alllist, pkl_path, pkl_name)
    save_pkl(all_label_alllist, pkl_path, pkl_name)


if __name__ == "__main__":
    main()
