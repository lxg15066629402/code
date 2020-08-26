# 读取dicom file ,nii file, 并实现数据与标签的对应
import pydicom
import nibabel as nib
import numpy as np
import cv2
import pickle
import os
import pydcm2png


# .dcm 文件 转化为 .png 文件
def dcm_png(dcm_folder):
    """

    :param dcm_folder: input dcm file
    :return: png file
    """
    png_dic = {}  # 字典类型
    # sorted 排序
    for filename in sorted(os.listdir(dcm_folder)):

        if filename == 'Untitled.nii.gz':
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


def get_nii_bounding_box(nii_floder, nii_name='Untitled.nii.gz', save_mask=False):
    '''
        解析nii文件的信息
        :param nii_floder:
        :param nii_name:
        :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
                 contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
    '''

    nii_file_path = os.path.join(nii_floder, nii_name)
    # load nii file()
    img_data = nib.load(nii_file_path)
    img = img_data.get_fdata()

    bounding_box = []
    contours_list = []
    mask_list = []
    for i in range(img.shape[-1]):
        print(img.shape)
        # 在标注保存时掩码进行来转置，因此这里需要transpose
        mask = np.transpose(img[:, :, i])
        # 二值化处理
        # print(i)
        # cv2.imwrite(f"/dicom/Jone/{i}.png", mask)
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f"/dicom/Jone/{i}.png", binary)
        # cv2.save("/dicom/Jome/1.png", mask)
# '''
        # 转化类型
        # print(type(binary))  # <class 'numpy.ndarray'>
        binary = np.array(binary, np.uint8)
        # print(type(binary))  # <class 'numpy.ndarray'>
        mask_list.append(binary)

        #  获取mask信息
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

        # 获取最大边框
        if contours !=[]:
            box = []  # 存在一张图中有多个结构扭曲区域，i 表示结构扭曲个数
            for i in range(len(contours)):
                # print(contours[i].shape)  # (176, 1, 2)
                # exit()
                x_max = int(np.max(contours[i][:, :, 0]))
                y_max = int(np.max(contours[i][:, :, 1]))
                x_min = int(np.min(contours[i][:, :, 0]))
                y_min = int(np.min(contours[i][:, :, 1]))
                box.append([x_min, y_min, x_max, y_max])
                # print(len(box))  # 一张图中的结构扭曲个数
            bounding_box.append(box)
        else:
            bounding_box.append([[0, 0, 0, 0]])  # 如何没有结构扭曲，则进行补充为 [[0, 0, 0, 0]]

        if save_mask == True:  # 保存掩码
            return bounding_box, contours_list, mask_list
        else:  # 不保存掩码
            return bounding_box, contours_list
            # '''

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
        bboxes = np.array(coords, dtype=np.float64)

        labels = np.ones(len(bboxes), dtype=np.int32)

        if save_mask == True:
            mask = np.array(mask_list, dtype=np.int32)
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
        os.mkdir(path)


def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load pkl file
def load_pkl(floder, name):
    with open(floder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():

    # 初始化路径
    dcm_floders = '/dicom/Jone/code/data_pre/data/structural/train/'
    nii_floders = '/dicom/Jone/code/data_pre/data/structural/train/'
    png_path = '/dicom/Jone/code/data_pre/data/structural/train_data_png'
    pkl_path = '/dicom/Jone/code/data_pre/data/structural/train_ann_file'
    ann_png_path = '/dicom/Jone/code/data_pre/data/structural/train_ann_png'
    pkl_name = '/20200520_142-learn'

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)

    label_alllist = []
    n = 0
    num = len(os.listdir(nii_floders))

    # sorted() 排序
    for floder in sorted(os.listdir(nii_floders)):  # 标签与数据对应

        n += 1
        dcm_floder = os.path.join(dcm_floders, floder)
        nii_floder = os.path.join(nii_floders, floder)

        # 处理dcm文件
        png_dic = dcm_png(dcm_floder)
        # print(png_dic)
        # exit()

        # 处理nii文件
        bounding_box_list, contours_list = get_nii_bounding_box(nii_floder)

        # 匹配dcm和nii文件的信息
        label_list, png_list = dcm_nii_con(bounding_box_list, png_dic, contours_list, floder, png_path, ann_png_path)

        # 添加label
        label_alllist += label_list
        print(f'Save {floder} successful,Residual:{num - n}')  # 打印输出结果

    # 将label存储为pkl格式
    save_pkl(label_alllist, pkl_path, pkl_name)


if __name__ == "__main__":
    main()
