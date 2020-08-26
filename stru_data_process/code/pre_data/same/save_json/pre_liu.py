# coding: utf-8
"""
# 数据来自：处理刘医生组数据
# 数据信息：进行双标数据，且标注的原则是一位医生是连续序列标注，一位医生是单序列标注
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


# save json file, obj 目标数据
def save_json(obj, folder, name):
    with open(folder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


# load json file
def load_json(folder, name):
    with open(folder + '/' + name + '.json', 'rb') as f:
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
    # print(nii_all.shape)
    # exit()
    for i in range(int(nii_all.shape[-1])):

        # 获取掩码信息
        mask = np.transpose(nii_all[:, :, i])  # 在标注保存时掩码进行来转置，因此这里需要transpose
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
    for file in os.listdir(folders):
        end_name = file.split('.')[-1]
        type_name = int(file.split('_')[0])
        split_name = file.split('.')[-3]
        # nii_mask_index = int(file.split('_')[1])

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

    doctor_name = list(nii_dic.keys())

    if len(doctor_name) != 2:
        print("只进行了一次标注")

    elif len(doctor_name) == 2:
        all_nii_dic = merge_nii(nii_dic['doctor_lyy'], nii_dic['doctor_sxj'])
    # all_nii_dic = merge_nii(doctor_name[1], doctor_name[0])

        return png_dic, all_nii_dic


# 将dcm文件和nii文件的信息进行匹配
def connect_pngs_boxs(bounding_box_list, png_dic, contours_list, folder, png_path, mask_png_path, img_id, ann_id,
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
        file_name = folder + '_' + pos + '.png'

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
    dcm_folders = '/dicom/Adam/data/structural/raw_data/20200619/double_same/liuyuanyuan_vs_shangxiaojing'
    # 01
    save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu_json/data'
    save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu_json/annotations'
    mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/liu_json/ann'
    save_json_name = 'liu_json'

    # 02
    # save_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/data'
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations'
    # mask_png_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data/ann'
    # save_json_name = 'liu_1_json'

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
    # image_id = 200
    # ann_id = 200
    for doctor in sorted(os.listdir(dcm_folders)):
        patient_path = os.path.join(dcm_folders, doctor)
        for i in os.listdir(patient_path):
            if i.split(".")[-1] == 'DS_Store':
                # print('remove')
                os.remove(os.path.join(patient_path, i))

        # if :
        # print(patient_path)
        png_dic, all_nii_dic = compare_double(patient_path)
        # 以label 为主
        for idx in list(all_nii_dic.keys()):
            # print(all_nii_dic.keys())
            # exit()
            bounding_box_list, contours_list = get_one_nii_bounding_box(all_nii_dic[idx])

            png_list, images, annotations, image_id, ann_id = connect_pngs_boxs(bounding_box_list, png_dic[int(idx)],
                                                                                contours_list, doctor, save_png_path,
                                                                                mask_png_path, image_id, ann_id,
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
