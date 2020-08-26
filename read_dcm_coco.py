# 构建coco 格式数据标签, 为使用 mmdetection 深度学习框架

import json
import os
import pydicom
import pydcm2png
import cv2
import nibabel as nib
import numpy as np


# save json file
def save_json(obj, floder, name):
    with open(floder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


def load_json(floder, name):
    with open(floder + '/' + name + '.json', 'rb') as f:
        return json.load(f)


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def dcm2png(folder):

    dict_data = {}
    for filename in sorted(os.listdir(folder)):

        if filename.split(".")[-1] != 'gz':

            # print(filename)
            file_name = os.path.join(folder, filename)  # 连接全路径
            # 获取文件编号(用于和标签的匹配)
            png_name = filename.split(".")[:-1]

            index = int(png_name[0].split('_')[1])

            # dcm转png
            img = pydcm2png.get_pixel_data(file_name)  # 转化

            # 读取图像位置信息
            ds = pydicom.read_file(file_name)

            ImageLaterality = ds.ImageLaterality

            ViewPosition = ds.ViewPosition

            pos = ImageLaterality + ViewPosition

            dict_data[index - 1] = [pos, img]  # 文件名序号是从1开始，index-1使其从0开始

        return dict_data


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
    img = img_data.get_fdata()  # 注意get_fdata() 与 get_data() 的区别

    bounding_box = []
    contours_list = []
    mask_list = []
    for i in range(img.shape[-1]):
        # 在标注保存时掩码进行来转置，因此这里需要transpose
        mask = np.transpose(img[:, :, i])
        # 二值化处理
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # 转化类型
        # print(type(binary))  # <class 'numpy.ndarray'>
        binary = np.array(binary, np.uint8)
        # print(type(binary))  # <class 'numpy.ndarray'>
        mask_list.append(binary)

        # 获取mask信息
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

        # 获取最大边框
        if contours !=[]:
            box = []  # 存在一张图中有多个结构扭曲区域，i 表示结构扭曲个数
            for i in range(len(contours)):
                print(contours[i].shape)  # (176, 1, 2)
                # exit()
                x_max = int(np.max(contours[i][:, :, 0]))
                y_max = int(np.max(contours[i][:, :, 1]))
                x_min = int(np.min(contours[i][:, :, 0]))
                y_min = int(np.min(contours[i][:, :, 1]))
                box.append([x_min, y_min, x_max, y_max])
                print(len(box))  # 一张图中的结构扭曲个数
            bounding_box.append(box)
        else:
            bounding_box.append([[0, 0, 0, 0]])  # 如何没有结构扭曲，则进行补充为 [[0, 0, 0, 0]]

        if save_mask == True:  # 保存掩码
            return bounding_box, contours_list, mask_list
        else:  # 不保存掩码
            return bounding_box, contours_list


# 将dcm文件和nii文件的信息进行匹配
def connect_pngs_boxs(bounding_box_list, png_dic, contours_list, floder, png_path, mask_png_path, image_id, ann_id,
                      save_png = False, save_mask = False):
    '''
    将dcm文件和nii文件的信息进行匹配
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
        pos, img = png_dic[index_img]
        file_name = floder + '_' + pos + '.png'

        height, width = img.shape  # 获取图像的尺度信息

        # 添加标签
        image['file_name'] = file_name
        image['height'] = int(height)
        image['width'] = int(width)
        image['id'] = int(image_id)

        # 遍历该图片标注内的每个框
        for index_box in range(len(coords)):
            annotation = {}

            # 获取掩码区域的面积
            area = int(cv2.contourArea(contours[index_box]))
            if area < 10:
                continue

            # 对边缘做变形，得到coco所需要的segmentation_label
            segmentation = contours[index_box].flatten().tolist()

            # 获取该标注的bbox
            bboxe = coords[index_box]

            # 获取当前标注区域的分类
            category = 0  #  ...

            # 添加标注
            annotation['segmentation'] = [segmentation]
            annotation['area'] = area
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = bboxe
            annotation['category_id'] = category
            annotation['id'] = ann_id  # 标签的id
            ann_id += 1
            annotations.append(annotation)

        image_id += 1
        images.append(image)
        png_list.append(img)

        # 保存dcm2png图像
        if save_png == True:
            cv2.imwrite(f'{png_path}/{file_name}', img)

        # 绘制标注结果
        if save_mask == True:
            for i in range(len(coords)):
                cv2.rectangle(img, (coords[i][0], coords[i][1]), ((coords[i][0]+coords[i][2]), (coords[i][1]+coords[i][3])), color=60000, thickness=5)
                cv2.drawContours(img, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(mask_png_path, file_name), img)

    return png_list, images, annotations, image_id, ann_id


def main(type):

    # 初始化路径信息
    floders = []
    test_floders = '/dicom/Jone/code/data_pre/data/structural/test'
    train_floders = '/dicom/Jone/code/data_pre/data/structural/train'

    save_png_path = f'/dicom/Jone/code/data_pre/data/structural/maskrcnn_{type}'
    save_json_path = '/dicom/Jone/code/data_pre/data/structural/maskrcnn/annotations'
    mask_png_path = f'/dicom/Jone/code/data_pre/data/structural/maskrcnn/maskrcnn_maskpng_{type}'
    save_json_name = f'mask_coco_{type}'
    path = ''

    # 创建文件保存路径信息
    create_dir_not_exist(save_png_path)
    create_dir_not_exist(save_json_path)
    create_dir_not_exist(mask_png_path)
    create_dir_not_exist(save_json_name)

    # 进行判断测试，验证与训练过程
    if type == 'test':
        floders = os.listdir(test_floders)[:125]
        path = test_floders
    elif type == 'val':
        floders = os.listdir(test_floders)[125:]
        path = test_floders
    elif type == 'train':
        floders = os.listdir(train_floders)
        path = train_floders

    # 初始化数据
    images_list = []
    annotations_list = []
    categories_list = [{'id': 0, 'name': 'struct'}, ]  #
    result = {}
    image_id = 0
    ann_id = 0
    for floder in floders:
        patient = os.path.join(path, floder)

        # 处理dcm文件
        png_dic = dcm2png(patient)
        # print(patient)
        # exit()

        # 获取nii信息
        bounding_box_list, contours_list = get_nii_bounding_box(patient)

        # 匹配dcm和nii文件的信息
        png_list, images, annotations, image_id, ann_id = connect_pngs_boxs(bounding_box_list,
                                                                            png_dic,
                                                                            contours_list,
                                                                            floder,
                                                                            save_png_path,
                                                                            mask_png_path,
                                                                            image_id, ann_id,
                                                                            save_png=True, save_mask=True)

        images_list.extend(images)
        annotations_list.extend(annotations)

        print(f'Save {floder} successful,ID:{image_id}')

        # if image_id>2:
    result['images'] = images_list
    result['annotations'] = annotations_list
    result['categories'] = categories_list
    result = json.dumps(result)
    save_json(result, save_json_path, save_json_name)


if __name__ == '__main__':

    # type = 'train'
    # main(type)

    # type = 'val'
    # main(type)

    type = 'test'
    main(type)
