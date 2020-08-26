# 处理姚医生数据
# 该数据都是进行双标数据
# 读双人标记一致的文件
# 思路： 对于双人标注的文件进行合并，即并操作

# 读取dicom file ,nii file, 并实现数据与标签的对应
import pydicom
import nibabel as nib
import numpy as np
import cv2
import pickle
import os
import pydcm2png
from pydicom.dicomdir import DicomDir


# .dcm 文件 转化为 .png 文件
def dcm_png(dcm_folder):
    """

    :param dcm_folder: input dcm file
    :return: png file
    """
    png_dic = {}  # 字典类型
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


def read_nii(nii_file):
    img_data = nib.load(nii_file)
    img = img_data.get_fdata()

    return img


def get_nii(nii_floder):
    nii_dict = {}
    nii_index = 0
    for nii_file in os.listdir(nii_floder):
        if nii_file.split('.')[-1] == '.DS_Store':
            os.remove(os.path.join(nii_floder, nii_file))
            continue
        # print(nii_file)
        if nii_file.split(".")[-1] == 'gz':
            nii_dict[f'doctor{nii_index}'] = os.path.join(nii_floder, nii_file)
            nii_index += 1

    return nii_dict


def get_nii_bounding_box(nii1, nii2, save_mask=False):

    # print([nii_file for nii_file in os.listdir(nii_floder) if nii_file.split(".") == "gz"])
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
    nii1_data = read_nii(nii1)
    nii2_data = read_nii(nii2)
    for i in range(nii1_data.shape[-1]):
        if nii1_data.shape[-1] != nii2_data.shape[-1]:
            continue
        else:
        # 在标注保存时掩码进行来转置，因此这里需要transpose
            mask1 = np.transpose(nii1_data[:, :, i])
            mask2 = np.transpose(nii2_data[:, :, i])
        # print(nii2)
        # print(nii2_data.shape)
        # print(nii2_data.shape[-1])

        # 二值化处理
        ret1, binary1 = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY)
        ret2, binary2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY)
        # 转化类型
        binary = cv2.bitwise_or(binary1, binary2)
        binary = np.array(binary, np.uint8)
        mask_list.append(binary)

        #  获取mask信息
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_list.append(contours)


        new_contours = []
        # 获取最大边框
        if contours != []:
            box = []  # 存在一张图中有多个结构扭曲区域，i 表示结构扭曲个数
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
            contours_list.append(new_contours)  # 如何没有结构扭曲，则进行补充为 [[0, 0, 0, 0]

    if save_mask == True:  # 保存掩码
        return bounding_box, contours_list, mask_list
    else:  # 不保存掩码
        return bounding_box, contours_list


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
                cv2.rectangle(image, (coords[i][0], coords[i][1]), (coords[i][2], coords[i][3]),
                              color=60000, thickness=5)
                cv2.drawContours(image, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(ann_png_path, filename), image)

    return label_list, png_list


def create_dir_not_exist(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)


def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load pkl file
def load_pkl(floder, name):
    with open(floder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():

    # 初始化路径
    dcm_floders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/yaoguoyan_vs_chenlijun'
    png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_data_png'
    pkl_path = '/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_ann_file'
    ann_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/yao/train_ann_png'
    pkl_name = '/differ-learn'

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)
    label_alllist = []
    n = 0
    for doctor in sorted(os.listdir((dcm_floders))):

        n += 1
        num = len(os.listdir(os.path.join(dcm_floders)))

        # sorted() 排序
        patient_path = os.path.join(dcm_floders, doctor)

        png_dic = dcm_png(patient_path)

        dict_nii = get_nii(patient_path)
        print(dict_nii)
        # exit()

        # bounding_box_list, contours_list = get_nii_bounding_box(patient_path)
        bounding_box_list, contours_list = get_nii_bounding_box(dict_nii['doctor0'], dict_nii['doctor1'])

        # 匹配dcm和nii文件的信息
        label_list, png_list = dcm_nii_con(bounding_box_list, png_dic, contours_list, doctor, png_path, ann_png_path)

        # 添加label
        label_alllist += label_list
        print(f'Save {doctor} successful,Residual:{num - n}')  # 打印输出结果

    # 将label存储为pkl格式
    save_pkl(label_alllist, pkl_path, pkl_name)


if __name__ == "__main__":
    main()
