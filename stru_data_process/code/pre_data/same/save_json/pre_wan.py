# coding: utf-8
"""
# 数据来自：处理万医生组数据
# 数据信息：进行双标数据，且标注的原则是单序列标注
# 思路： 对于双人标注的文件进行合并，即并操作
# 功能： 实现数据转化，保存成coco 格式的数据，便于mmdetection 深度学习目标检测任务使用
"""

import pydicom
import nibabel as nib
import numpy as np
import cv2
import os
import pydcm2png
import json


# save json file
# obj 目标数据
def save_json(obj, floder, name):
    with open(floder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


# load json file
def load_json(floder, name):
    with open(floder + '/' + name + '.json', 'rb') as f:
        return json.load(f)


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 创建文件夹
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def get_one_nii(nii_file):
    # 加载nii文件
    img_data = nib.load(nii_file)
    img = img_data.get_fdata()

    return img


# 获取标注数据中的标签信息
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
                    box.append([x_min, y_min, x_max - x_min, y_max - y_min])
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


# 比较nii文件, 医生标注的是同一影像
def merge_nii(doctor0_nii, doctor1_nii):
    # iou_all = cv2.bitwise_or(doctor0_nii['img'], doctor1_nii['img'])
    iou_all = np.add(doctor0_nii['img'], doctor1_nii['img'])

    return iou_all


# 比较双标数据
def compare_double(folders):

    png_dic = {}
    nii_dic = {}
    # 获取病人的dcm和nii
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
    all_nii_dic = {}

    if len(doctor_name) == 2:
        nii_index_list = nii_dic[doctor_name[0]].keys()
        for index_nii in nii_index_list:
            # print(index_nii)
            # exit()
            if nii_dic[doctor_name[0]].keys() != nii_dic[doctor_name[1]].keys():
                print('医生标注的不是同一影像')
                continue
            else:
                nii_all = merge_nii(nii_dic[doctor_name[0]][index_nii], nii_dic[doctor_name[1]][index_nii])

                all_nii_dic[index_nii] = nii_all
    else:
        print("只有一个医生标注或没有医生标注")
    return png_dic, all_nii_dic


# 将dcm文件和nii文件的信息进行匹配
def connect_pngs_boxs(bounding_box_list, png_dic, contours_list, floder, png_path, mask_png_path, img_id, ann_id,
                      save_png=False, save_mask=False):
    '''
    将dcm文件和nii文件的信息进行匹配
    :param save_png:
    :param ann_id:
    :param image_id:
    :param mask_png_path:
    :param png_path:
    :param floder:
    :param png_dic:
    :param contours_list:
    :param save_mask:
    :param bounding_box_list:
    :param png_img_list:
    :return:
    '''

    png_list = []
    images = []
    annotations = []
    # len(bounding_box_list) = dcm文件夹内的图片个数
    for index_img in range(len(bounding_box_list)):
        image = {}

        # 获取当前图片的bbbox
        coords = bounding_box_list[index_img]

        # 获取当前图片的标注
        contours = contours_list[index_img]

        # 当该图片内没有标注数据，则跳过该图片
        if coords == [[0, 0, 0, 0]]:
            continue

        # 获取图片，即获取数据dcm
        pos, img = png_dic
        file_name = floder + '_' + pos + '.png'

        height, width = img.shape  # 获取图像的尺度信息

        # 添加标签
        image['file_name'] = file_name
        image['height'] = int(height)
        image['width'] = int(width)
        image['id'] = int(img_id)

        # 遍历该图片标注内的每个框
        for index_box in range(len(coords)):
            annotation = {}

            # 获取掩码区域的面积
            area = int(cv2.contourArea(contours[index_box]))
            if area < 10:
                continue

            # 对边缘做变形，得到coco所需要的segmentation_label
            segmentation = contours[index_box].flatten().tolist()  # flatten 操作

            # 获取该标注的bbox
            bboxe = coords[index_box]

            # 获取当前标注区域的分类
            category = 0  # ...

            # 添加标注
            annotation['segmentation'] = [segmentation]
            annotation['area'] = area
            annotation['iscrowd'] = 0
            annotation['image_id'] = img_id
            annotation['bbox'] = bboxe
            annotation['category_id'] = category
            annotation['id'] = ann_id  # 标签的id
            ann_id += 1
            annotations.append(annotation)

        img_id += 1
        images.append(image)
        png_list.append(img)

        # 保存dcm2png图像
        if save_png == True:
            cv2.imwrite(f'{png_path}/{file_name}', img)

        # 绘制标注结果 # **
        if save_mask == True:
            for i in range(len(coords)):
                cv2.rectangle(img, (coords[i][0], coords[i][1]), ((coords[i][0]+coords[i][2]),
                                (coords[i][1]+coords[i][3])), color=60000, thickness=5)
                cv2.drawContours(img, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(mask_png_path, file_name), img)

    return png_list, images, annotations, img_id, ann_id


def main():

    # 初始化路径
    # ***wan1
    # dcm_folders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/wanyun_vs_huangyan1'
    # 01
    # save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json/data'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json/annotations'
    # mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json/ann'
    save_json_name = 'wan1_json'

    # 02
    # save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/data'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations'
    # mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/ann'
    # save_json_name = 'wan1_1_json'


    # ***wan2
    dcm_folders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/wanyun_vs_huangyan2'
    # 01
    save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json/data'
    save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json/annotations'
    mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json/ann'
    save_json_name = 'wan2_json'

    # 02
    # save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/data'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations'
    # mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/ann'
    # save_json_name = 'wan2_2_json'

    # 创建文件保存路径
    create_dir_not_exists(save_png_path)
    create_dir_not_exists(save_json_path)
    create_dir_not_exists(mask_png_path)
    create_dir_not_exist(save_json_name)

    images_list = []
    annotations_list = []
    categories_list = [{'id': 0, 'name': 'struct'}, ]  #
    result = {}
    # 01
    image_id = 0
    ann_id = 0

    # 02
    # **wan1
    # image_id = 636
    # ann_id = 636
    # **wan2
    # image_id = 712
    # ann_id = 712

    for doctor in sorted(os.listdir(dcm_folders)):

        patient_path = os.path.join(dcm_folders, doctor)
        png_dic, all_nii_dic = compare_double(patient_path)

        for idx in list(all_nii_dic.keys()):  # 以label 为主

            bounding_box_list, contours_list = get_one_nii_bounding_box(all_nii_dic[idx])

            png_list, images, annotations, image_id, ann_id = connect_pngs_boxs(bounding_box_list, png_dic[idx],
                                                contours_list, doctor, save_png_path, mask_png_path,image_id, ann_id,
                                                                                save_png=True, save_mask=True)

            images_list.extend(images)
            annotations_list.extend(annotations)

        print(f'Save {doctor} successful,ID:{image_id}')

    result['images'] = images_list
    result['annotations'] = annotations_list
    result['categories'] = categories_list
    result = json.dumps(result)
    save_json(result, save_json_path, save_json_name)


if __name__ == "__main__":
    main()
