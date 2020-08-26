# coding: utf-8
"""
# 处理万云医生数据
# 该数据都是进行双标数据，另外该标注数据单序列标注，需要根据文件名进行对应处理
# 读双人标记一致的文件
# 思路： 对于双人标注的文件进行合并，即并操作
# 读取dicom file ,nii file, 并实现数据与标签的对应
"""

import nibabel as nib
import cv2
import numpy as np
import pydcm2png
import os
import pydicom
import pickle
import shutil


# 创建文件夹
def create_dir_not_exist(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)


def get_one_nii(nii_file):
    # 加载nii文件
    img_data = nib.load(nii_file)
    img = img_data.get_fdata()

    return img


def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load pkl file
def load_pkl(floder, name):
    with open(floder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_one_nii_bounding_box(nii_all, save_mask=False):
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
    # print(nii_all)
    # exit()
    for i in range(int(nii_all.shape[-1])):

        # 获取掩码信息
        # print(nii_all[:, :, i].shape)
        mask = np.transpose(nii_all[:, :, i])  # 在标注保存时掩码进行来转置，因此这里需要transpose
        # print("***********")
        # print(mask.shape)
        # exit()
        # 二值化处理
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        binary = np.array(binary, np.uint8)
        mask_list.append(binary)
        # 获取掩码边
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_contours = []
        # 获取最大边框
        if contours != []:
            box = []  # 存在一张图中有多个结构扭曲区域，i代表结构扭曲个数
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > 10:
                    x_max = int(np.max(contours[i][:, :, 0]))
                    y_max = int(np.max(contours[i][:, :, 1]))
                    x_min = int(np.min(contours[i][:, :, 0]))
                    y_min = int(np.min(contours[i][:, :, 1]))
                    # box.append([x_min, y_min, x_max - x_min, y_max - y_min])
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


# 读取dcm文件并转为png文件
def dcm2png(dcm_file):
    '''
    将dcm文件转为png文件，并获取dcm信息
    :param dcm_folder_path:例如20110304（该文件夹的目录为20110304/ExamImage/images.dcm）
    :return: png_dic 形式为png_dic[索引] = [拍摄位,img]，保存img(保存在png_img/20110304/images.png）
    '''
    # 读取dcm 数据
    img = pydcm2png.get_pixel_data(dcm_file)

    # 获取拍摄位信息
    ds = pydicom.read_file(dcm_file)
    ImageLaterality = ds.ImageLaterality
    ViewPosition = ds.ViewPosition
    pos = ImageLaterality + ViewPosition

    return pos, img


# 比较nii文件, 医生标注的是同一影像
def merge_nii(doctor0_nii, doctor1_nii):
    # iou_all = cv2.bitwise_or(doctor0_nii['img'], doctor1_nii['img'])
    iou_all = np.add(doctor0_nii['img'], doctor1_nii['img'])

    return iou_all


# 比较双标数据
def compare_double(folders):

    # 获取病人的dcm和nii
    png_dic = {}
    nii_dic = {}
    # 获取病人的dcm和nii
    # exit()
    for file in os.listdir(folders):
        patient = os.path.join(folders, file)
        end_name = file.split('.')[-1]
        # 获取dcm信息,检索信息
        if end_name == '3':

            dcm_type = int(file.split('_')[0])
            dcm_index = int(file.split('_')[1])
            pos, img = dcm2png(patient)
            png_dic[f'{dcm_type}_{dcm_index - 1}'] = [pos, img]

        # 获取nii信息
        elif end_name == 'gz':
            doctor = file.split('.')[-3]  # 获取医生
            if f'doctor{doctor}' not in nii_dic.keys():  #
                nii_dic[f'doctor{doctor}'] = {}

            nii_type = int(file.split('_')[0])
            nii_mask_index = int(file.split('_')[1])

            img = get_one_nii(patient)
            nii_dic[f'doctor{doctor}'][f'{nii_type}_{nii_mask_index - 1}'] = {"img": img}

    doctor_name = list(nii_dic.keys())
    # print(doctor_name)
    # exit()
    all_nii_dic = {}

    if len(doctor_name) == 2:
        nii_index_list = nii_dic[doctor_name[0]].keys()
        # print(nii_dic[doctor_name[0]][index_nii])
        for index_nii in nii_index_list:
            # print(index_nii)
            # exit()
            if nii_dic[doctor_name[0]].keys() != nii_dic[doctor_name[1]].keys():
                # print('医生标注的不是同一影像')
                continue
            else:
                nii_all = merge_nii(nii_dic[doctor_name[0]][index_nii], nii_dic[doctor_name[1]][index_nii])

                all_nii_dic[index_nii] = nii_all
    else:
        print("只有一个医生标注或没有医生标注")
    # print(all_nii_dic)
    # exit()
    return png_dic, all_nii_dic


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
                cv2.rectangle(image, (coords[i][0], coords[i][1]), (coords[i][2], coords[i][3]),
                              color=60000, thickness=5)
                cv2.drawContours(image, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(ann_png_path, filename), image)

    return label_list, png_list


def main():

    # 初始化路径
    dcm_floders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/wanyun_vs_huangyan2'
    png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2/train_data_png'
    pkl_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2/train_ann_file'
    ann_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2/train_ann_png'
    pkl_name = '/differ-learn'

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)
    all_label_alllist = []
    n = 0

    for doctor in sorted(os.listdir((dcm_floders))):

        label_alllist = []
        num = len(os.listdir(dcm_floders))
        n += 1
        # sorted() 排序
        patient_path = os.path.join(dcm_floders, doctor)
        # remove '.DS_Store'
        for i in os.listdir(patient_path):
            if i.split(".")[-1] == 'DS_Store':
                # print('remove')
                os.remove(os.path.join(patient_path, i))

        png_dic, all_nii_dic = compare_double(patient_path)

        for idx in list(all_nii_dic.keys()):  # 以label 为主

            bounding_box_list, contours_list = get_one_nii_bounding_box(all_nii_dic[idx])

            label_list, png_list = dcm_nii_con(bounding_box_list, png_dic[idx], contours_list, doctor, png_path, ann_png_path)

            # 添加label
            label_alllist += label_list
        print(f'Save {doctor} successful,Residual:{num-n}')  # 打印输出结果
        all_label_alllist += label_alllist
        # 将label存储为pkl格式
    save_pkl(all_label_alllist, pkl_path, pkl_name)


def pre_path():
    ori_path = '/dicom/Adam/data/structural/double/wy_hy2/'
    new_path = '/dicom/Adam/data/structural/double/wanyun_vs_huangyan06.24/'
    floders = os.listdir(ori_path)
    erro = []

    for floder in floders:
        print(floder)
        if floder == '.DS_Store':
            continue
        patients = os.path.join(ori_path, floder)
        for in_patient_file in os.listdir(patients):
            if in_patient_file == '.DS_Store':
                os.remove(os.path.join(patients, in_patient_file))
                continue
            else:
                other_floder = os.path.join(patients, in_patient_file)

                other_index = in_patient_file.split('_')[0]

                for file in os.listdir(other_floder):
                    if file != '.DS_Store':
                        file_path = f'{other_floder}/{file}'

                        save_path = f'{new_path}{floder}_{other_index}/{file}'


                        create_dir_not_exist(f'{new_path}{floder}_{other_index}')

                        shutil.copyfile(file_path, save_path)


if __name__ == '__main__':
    # pre_path()
    # compare_double()
    main()