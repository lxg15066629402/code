# 构建结构扭曲数据， 保存成模型需要的数据
""""
 这批数据共有557例数据，其中双标正确数据 441 列
 双标不对应，进行处理的数据为 116例。
 现在已有数据文件夹  557例所有数据   116不同数据
 功能：  找出除去不同数据，剩余双标正确的 441 例数据
"""


import os
import numpy as np
import shutil
import nibabel as nib
import cv2

# path_all, path_differ
doctor = []


def create_dir_not_exist(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            # os.makedirs(path)
        except OSError:
            pass
        # os.mkdir(path)


def get_doctor(path):
    for doctor_name in os.listdir(path):
        doctor.append(doctor_name)

    return doctor


patients_all = []
patients_diff = []


def get_patients(doctors, folder_all, folder_diff):

    for doctor in sorted(doctors):
        patient_all = sorted(os.listdir(os.path.join(folder_all, doctor)))
        patient_diff = sorted(os.listdir(os.path.join(folder_diff, doctor)))

        patients_all.append(patient_all)
        patients_diff.append(patient_diff)

    return patients_all, patients_diff


# patient_same_all = []


# def get_same_patient(patient_all, patient_diff):
#     # x for x in a if x in b
#     for x in patient_all:
#         if x not in patient_diff:
#             patient_same_all.append(x)
#     return patient_same_all

patient_same_all = []


def get_same_patient(patients_all, patients_diff):
    for i in range(len(patients_all)):
        patient_same_all.append([x for x in patients_all[i] if x not in patients_diff[i]])
    return patient_same_all


def save_same(org_root, save_root, doctors, patients):
    # root, doctor, patient
    # os.walk()
    # for i in range(len(sorted(doctors))):
    for i in range(len(doctors)):
    # for i in range(len(sorted(doctors, key=lambda i:i[0]))):
        # print(sorted(doctors))
        # exit()
        # print(doctors[0], doctors[1], doctors[2], doctors[3])
        for patient in sorted(patients[i][:]):
            # print(patients[1][:])
            # exit()

            files = os.path.join(org_root, sorted(doctors)[i], patient)
            # print(files)

            same_file = os.path.join(save_root, sorted(doctors)[i], patient)
            # create_dir_not_exist(same_file)
            # exit()
            # if os.path.exists(same_file):
            if not os.path.exists(same_file):
                os.mkdir(same_file)
                shutil.copytree(files, same_file)  # 文件的移动复制保存


# # 合并数据，即将医生标注的数据进行合并操作
# def read_label(path, doctor, patient, nii_floder, save_mask=False):
#     '''
#         解析nii文件的信息
#         :param nii_floder:
#         :param nii_name:
#         :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
#                  contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
#     '''
#
#     nii_floder = os.path.join(path, doctor, patient, nii_floder)
#     img_all = []
#     for nii_file in os.listdir(nii_floder):
#         if nii_file.split('.')[-1] == 'gz':
#             img_data = nib.load(nii_file)
#             img = img_data.get_fdata()
#             img_all.append(img)
#
#     bounding_box = []
#     contours_list = []
#     mask_list = []
#     for i in range(len(img_all)):
#         for i in range(img_all[i].shape[-1]):
#             # 在标注保存时掩码进行来转置，因此这里需要transpose
#             mask = np.transpose(img[:, :, i])
#             # 二值化处理
#             ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
#             # 转化类型
#             # print(type(binary))  # <class 'numpy.ndarray'>
#             binary = np.array(binary, np.uint8)
#             # print(type(binary))  # <class 'numpy.ndarray'>
#             mask_list.append(binary)
#
#             #  获取mask信息
#             contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours_list.append(contours)
#
#             # 获取最大边框
#             if contours !=[]:
#                 box = []  # 存在一张图中有多个结构扭曲区域，i 表示结构扭曲个数
#                 for i in range(len(contours)):
#                     print(contours[i].shape)  # (176, 1, 2)
#                     # exit()
#                     x_max = int(np.max(contours[i][:, :, 0]))
#                     y_max = int(np.max(contours[i][:, :, 1]))
#                     x_min = int(np.min(contours[i][:, :, 0]))
#                     y_min = int(np.min(contours[i][:, :, 1]))
#                     box.append([x_min, y_min, x_max, y_max])
#                     print(len(box))  # 一张图中的结构扭曲个数
#                 bounding_box.append(box)
#             else:
#                 bounding_box.append([[0, 0, 0, 0]])  # 如何没有结构扭曲，则进行补充为 [[0, 0, 0, 0]]
#
#             if save_mask == True:  # 保存掩码
#                 return bounding_box, contours_list, mask_list
#             else:  # 不保存掩码
#                 return bounding_box, contours_list
#
#         # save nii
#         nib.Nifti1Image(img_merge, img_affine).to_filename('xxxxx.ZB.nii.gz')


def main():
    path_all = "/dicom/Adam/data/structural/raw_data/20200619/double_all"
    path_diff = "/dicom/Adam/data/structural/raw_data/20200619/double_differ"
    path_same_save = "/dicom/Adam/data/structural/raw_data/20200619/double_same"

    create_dir_not_exist(path_same_save)

    doctors = get_doctor(path_all)
    # print(doctor)
    patients_all, patients_diff = get_patients(doctors, path_all, path_diff)

    # print(patients_all)
    # print(len(patients_all))
    # print(np.array(patient_all).shape)
    # exit()

    # patient_same_all = []
    # print(patients_all)
    # print(patients_diff)
    # exit()

    # for i in range(len(patients_all)):
    #     # print(patients_all[i], patients_diff[i])
    #     patient_same = get_same_patient(patients_all[i], patients_diff[i])
    # patient_same_all.append(patient_same)

    # return patient_same_all

    patient_same = get_same_patient(patients_all, patients_diff)
    # print(patient_same)
    # exit()

    save_same(path_all, path_same_save, doctors, patient_same)
    # return patient_same

# def read_label(path, doctors, patients):
#
#     for doctor in sorted(doctors):
#         for patient in sorted(patients):
#             pass
#     return 0


if __name__ == "__main__":
    main()
    # print(a)
    # print(len(a))
