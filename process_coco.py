# 构建coco 格式数据标签
import nibabel as nib
import cv2
import numpy as np
import pydcm2png
import os
import pydicom
import json


def save_json(obj, floder, name):
    with open(floder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


def load_json(floder, name):
    with open(floder + '/' + name + '.json', 'rb') as f:
        return json.load(f)


def get_nii_bounding_box(nii_file, save_mask=False):
    '''
    解析nii文件的信息
    :param nii_floder:
    :param nii_name:
    :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
             contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
    '''

    # 加载nii文件
    img_data = nib.load(nii_file)
    img = img_data.get_fdata()

    bounding_box = []
    contours_list = []
    mask_list = []
    for i in range(int(img.shape[-1])):

        # 获取掩码信息
        mask = np.transpose(img[:, :, i])  # 在标注保存时掩码进行来转置，因此这里需要transpose

        # 二值化处理
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        binary = np.array(binary, np.uint8)
        mask_list.append(binary)
        # cv2.imwrite(f'/home/ubuntu/Adam/code/mmdetection/liu_result/mask{i}.png',binary)
        # 获取掩码边
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    box.append([x_min, y_min, x_max-x_min, y_max-y_min])  #
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


# 将dcm文件和nii文件的信息进行匹配
def connect_pngs_boxs(bounding_box_list,
                   png_dic,
                   contours_list,
                   floder,
                   png_path,
                   mask_png_path,
                   image_id, ann_id,
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

        # 获取图片
        pos, img = png_dic[index_img]
        file_name = floder + '_' + pos + '.png'

        height, width = img.shape

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
            segmentation=contours[index_box].flatten().tolist()

            # 获取该标注的bbox
            bboxe = coords[index_box]

            # 获取当前标注区域的分类
            category = 0

            # 添加标注
            annotation['segmentation'] = [segmentation]
            annotation['area'] = area
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = bboxe
            annotation['category_id'] = category
            annotation['id'] = ann_id  #
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
    # 初始化路径
    floders = []
    test_floders = '/dicom/Adam/data/structural/test'
    train_floders = '/dicom/Adam/data/structural/train'

    save_png_path = f'/dicom/Adam/code/mmdetection/data/structrual/maskrcnn_{type}'
    save_json_path = '/dicom/Adam/code/mmdetection/data/structrual/annotations'
    mask_png_path = f'/dicom/Adam/code/mmdetection/data/structrual/maskrcnn_maskpng_{type}'
    save_json_name = f'mask_coco_{type}'
    path = ''

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
    categories_list = [{'id': 0, 'name': 'struct'}, ]
    result = {}
    image_id = 0
    ann_id = 0
    for floder in floders:
        png_dic = {}
        patient = os.path.join(path, floder)

        # 获取dcm信息
        for file in os.listdir(patient):
            if file.split('.')[-1] != 'gz':  # 判断
                index = int(file.split('_')[1])
                dcm_file = os.path.join(patient, file)
                pos, img = dcm2png(dcm_file)  # 读取dcm 文件
                png_dic[index - 1] = [pos, img]

        # 获取nii信息
        nii_file = os.path.join(patient, 'Untitled.nii.gz')
        bounding_box_list, contours_list = get_nii_bounding_box(nii_file)

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

            # print('AllSave')
            # break


if __name__ == '__main__':
    # type = 'train'
    # main(type)
    type = 'val'
    main(type)
    type = 'test'
    main(type)

    # save_json_path = '/dicom//Adam/code/mmdetection/data/structrual/annotations'
    # save_json_name = f'mask_coco_{type}_0527'
    # f = load_json(save_json_path,save_json_name)
    # for i in f['annotations']:
    #     print(i['id'])
