# data gain processing pkl file ways
# coding: utf-8
import nibabel as nib
import cv2
import numpy as np
import pydcm2png
import os
import pydicom
import pickle
import json
from pathlib import Path
import shutil


def remove_empty_floder(floder_path, file_name='Untitled.nii.gz', floder_name='ExamImage',
                        type='file_exit'):
    """
    function: remove empty floder
    :param floder_path: 文件夹路径
    :param file_name:
    :param floder_name:
    :param type:
    :return:
    """

    if type == 'file_exit':
        file = Path(os.path.join(floder_path, file_name))
        if file.exists():  # 判断当前路径是否是文件或者文件夹
            pass
        else:
            shutil.rmtree(floder_path)
            print(f'nii remove {floder_path}')

    if type == 'dir_empty':
        dir = os.path.join(floder_path, floder_name)
        if os.listdir(dir) == [] :
            shutil.rmtree(floder_path)
            print(f'dcm remove {floder_path}')
        else:
            pass


def remove_empty_floders(nii_floders, dcm_floders):

    for floder in os.listdir(nii_floders):
        if floder == '.DS_Store':
            os.remove(os.path.join(nii_floders, floder))
            continue
        nii_path = os.path.join(nii_floders, floder)
        remove_empty_floder(nii_path, type='file_eixt')

    for floder in os.listdir(dcm_floders):
        if floder == '.DS_Store':
            os.remove(os.path.join(dcm_floders, floder))
            continue
        dcm_path = os.path.join(dcm_floders, floder)
        remove_empty_floder(dcm_path, type='dir_empty')


def remove_pkl_label():
    '''
    从pkl文件中删除标注有问题的erro文件，并重新保存
    :return:
    '''
    pkl_path = '/dicom/Jone/code/data_pre/data/structural/annotations'
    pkl_name = '20200520_142'
    pkl_file = load_pkl(pkl_path, pkl_name)
    erro =['20101222_1534497_RLM.png',
           '20110627_2017440_LCC.png',
           '20110726_2070608_LMLO.png',
           '20110823_2121759_RLM.png']
    for label in pkl_file:
        if label['filename'] in erro:
            pkl_file.remove(label)
            print(f'remove {label}')

    save_pkl(pkl_file, pkl_path, pkl_name)


def remove_png():
    '''
    删除标注有问题的png图像
    :return:
    '''
    png_path = '/dicom/Jone/code/data_pre/data/structrual/train'
    erro =['20101222_1534497_RLM.png',
           '20110627_2017440_LCC.png',
           '20110726_2070608_LMLO.png',
           '20110823_2121759_RLM.png']
    for file in os.listdir(png_path):
        if file in erro:
            os.remove(os.path.join(png_path, file))
            print(f'remove {file}')


def remove_double():
    """

    :return:  保存pkl file
    """
    path = "/dicom/Jone/code/data_pre/data/structural/train"
    print(f'train image:{len(os.listdir(path))}')

    pkl_path = '/dicom/Jone/code/data_pre/data/structural/train/annotations'
    pkl_name = '20200520_142'
    pkl_file = load_pkl(pkl_path, pkl_name)

    new_pklname = []
    dic = {}
    for label in pkl_file:
        if label['filename'] not in dic.keys():
            new_pklname.append(label)
        dic[label['filename']] = 0
    print(len(new_pklname))
    save_pkl(new_pklname, pkl_path, pkl_name)


# save pkl file
def save_pkl(obj, floder, name):
    with open(floder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load pkl file
def load_pkl(floder, name):
    with open(floder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# 创建不存在的文件夹信息
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


# 统一文件名
def unify_file_name(floders, split_str = '_'):
    '''
    对floders中的所有文件夹重命名，用于处理结构扭曲返回的标注文件中名称不一致的问题
    :param floders:
    :return:
    '''

    for old_floder in os.listdir(floders):
        new_floder = old_floder.split(split_str)[0] + split_str + old_floder.split(split_str)[-1]
        os.rename(os.path.join(floders, old_floder), os.path.join(floders, new_floder))


# 获得bounding box
def get_nii_bounding_box(nii_floder, nii_name='Untitled.nii.gz', save_mask=False):
    '''
    解析nii文件的信息
    :param nii_floder:
    :param nii_name:
    :return: bounding_box[n,x,2,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 2代表左上和右下角的点 2代表每个点的x坐标和y坐标
             contours_list[n,x,i,2] n代表图像的个数 x代表每个图像中结构扭曲的个数 i代表有每个结构扭曲线段的个数 2代表每个点的x坐标和y坐标
    '''

    nii_file_path = os.path.join(nii_floder, nii_name)

    # 加载nii文件
    img_data = nib.load(nii_file_path)  # nib.load()读取文件，会将图像向左旋转90°，坐标顺序为W*H*C
    img = img_data.get_fdata()  # 获取数据信息 （x, y, z）
    # print(img.shape)  #
    # exit()

    bounding_box = []
    contours_list = []
    mask_list = []
    for i in range(int(img.shape[-1])):

        # 获取掩码信息
        # print(img[:, :, i].shape)
        mask = np.transpose(img[:, :, i])  # 在标注保存时掩码进行来转置，因此这里需要transpose
        # print(mask.shape)
        # exit()
        # 二值化处理
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # 转化为二值图像
        print(binary)
        binary = np.array(binary, np.uint8)  # 转换为array类型
        # print(binary)
        # exit()
        mask_list.append(binary)

        # 获取掩码边
        # cv2.findContours()函数来查找检测物体的轮廓
        # 三个返回值：图像，轮廓，轮廓的层析结构 或者  2个返回值 轮廓，轮廓的层析结构
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

        # 获取最大边框
        if contours != []:
            box = []  # 存在一张图中有多个结构扭曲区域，i代表结构扭曲个数
            for i in range(len(contours)):
                # print(contours)
                # exit()
                x_max = int(np.max(contours[i][:, :, 0]))
                y_max = int(np.max(contours[i][:, :, 1]))
                x_min = int(np.min(contours[i][:, :, 0]))
                y_min = int(np.min(contours[i][:, :, 1]))
                box.append([x_min, y_min, x_max, y_max])  # 获得box 框
            bounding_box.append(box)
        else:
            bounding_box.append([[0, 0, 0, 0]])

    if save_mask == True:
        return bounding_box, contours_list, mask_list
    else:
        return bounding_box, contours_list


# dcm file 转化为 png file
def dcm2png(dcm_folder, dcm_name = 'ExamImage'):
    '''
    将dcm文件转为png文件，并获取dcm信息
    :param dcm_folder_path:例如20110304（该文件夹的目录为20110304/ExamImage/images.dcm）
    :return: png_dic 形式为png_dic[索引] = [拍摄位,img]，保存img(保存在png_img/20110304/images.png）
    '''

    # 获取路径
    filepath, filename = os.path.split(dcm_folder)
    # print(dcm_folder)
    # print(filepath, filename)
    dcm_folder_path = os.path.join(dcm_folder, dcm_name)
    # print(dcm_folder_path)
    # exit()

    png_dic = {}  # 字典类型

    for file_name in os.listdir(dcm_folder_path):
        if file_name == '.DS_Store':
            continue

        # 获取文件编号(用于和标签的匹配)
        # print(file_name)
        png_name = file_name.split(".")[:-1]
        # print(png_name)
        index = int(png_name[0].split('_')[1])
        # print(index)
        # exit()

        # dcm转png
        file_path = os.path.join(dcm_folder_path, file_name)
        img = pydcm2png.get_pixel_data(file_path)  # 转化

        # 获取拍摄位信息
        ds = pydicom.read_file(file_path)
        ImageLaterality = ds.ImageLaterality
        # print(ImageLaterality)
        ViewPosition = ds.ViewPosition
        # print(ViewPosition)
        pos = ImageLaterality + ViewPosition
        # print(pos)
        # exit()

        # 存入字典
        png_dic[index-1] = [pos, img]  # 文件名序号是从1开始，index-1使其从0开始
        # print(png_dic)
        # exit()
    return png_dic


def dcm2png_liu(dcm_folder, dcm_name):
    '''
    将dcm文件转为png文件，并获取dcm信息
    :param dcm_folder_path:例如20110304（该文件夹的目录为20110304/ExamImage/images.dcm）
    :return: png_dic 形式为png_dic[索引] = [拍摄位,img]，保存img(保存在png_img/20110304/images.png）
    '''

    # 获取路径
    dcm_folder_path = os.path.join(dcm_folder, dcm_name)

    # dcm转png
    img = pydcm2png.get_pixel_data(dcm_folder_path)

    # 获取拍摄位信息
    ds = pydicom.read_file(dcm_folder_path)
    ImageLaterality = ds.ImageLaterality
    ViewPosition = ds.ViewPosition
    pos = ImageLaterality + ViewPosition

    # 存入字典
    pos, img  # 文件名序号是从1开始，index-1使其从0开始
    return pos, img


def show_pngs_boxs(bounding_box_list,
                   png_dic,
                   contours_list,
                   floder,
                   png_path,
                   ann_png_path,
                   save = True,
                   draw = True,
                   is_liu = False,
                   mask_list = None,
                   save_mask = False):

    '''
    将dcm文件和nii文件的信息进行匹配
    :param bounding_box_list:  目标框
    :param png_img_list:
    :return:
    '''

    label_list = []
    png_list = []

    for index in range(len(bounding_box_list)):
        dic_data = {}
        ann = {}
        # 获取对应信息
        coords = bounding_box_list[index]
        if coords == [[0, 0, 0, 0]]:
            continue
        contours = contours_list[index]
        pos, image = png_dic[index]  #
        if is_liu == True:
            _, tempfilename = os.path.split(floder)

            # print(tempfilename)  # 保存的文件名太长，在这里修改
            print(tempfilename)
            file_name = tempfilename.split('_')[0] + '_' + pos + '.png'
            print(file_name)
            # exit()
        else:
            file_name = floder + '_' + pos + '.png'
            print(floder)  # 20110224_1587675
            exit()

        height, width = image.shape
        bboxes = np.array(coords, dtype=np.float64)

        labels = np.ones(len(bboxes), dtype=np.int32)
        # print(len(bboxes))  # 2
        # print(labels)  # [1 1]
        # exit()
        if save_mask == True:
            mask = np.array(mask_list, dtype=np.int32)
            ann['masks'] = mask[index]

        # 标签填充
        ann['bboxes'] = bboxes
        ann['labels'] = labels

        dic_data['filename'] = file_name
        dic_data['width'] = width
        dic_data['height'] = height
        dic_data['ann'] = ann

        label_list.append(dic_data)
        png_list.append(image)
        # 保存dcm2png图像
        if save == True:
            cv2.imwrite(f'{png_path}/{file_name}', image)
        # 绘制标注结果
        if draw == True:
            for i in range(len(coords)):
                cv2.rectangle(image, (coords[i][0], coords[i][1]), (coords[i][2], coords[i][3]), color=60000, thickness=5)
                cv2.drawContours(image, contours[i], -1, color=60000, thickness=5)
            cv2.imwrite(os.path.join(ann_png_path, file_name), image)

    return label_list, png_list


def main():
    # 初始化路径
    dcm_floders = '/dicom/Jone/code/data_pre/data/structural/20200520_142/dcm/'
    nii_floders = '/dicom/Jone/code/data_pre/data/structural/20200520_142/nii/'
    png_path = '/dicom/Jone/code/data_pre/data/structural/train-learn'
    pkl_path = '/dicom/Jone/code/data_pre/data/structural/annotations'
    ann_png_path = '/dicom/Jone/code/data_pre/data/structural/ann_png'
    pkl_name = '/20200520_142-learn'

    # 统一dcm和nii的文件夹名
    # unify_file_name(dcm_floders)
    # unify_file_name(nii_floders)

    # 创建文件保存路径
    create_dir_not_exist(png_path)
    create_dir_not_exist(pkl_path)
    create_dir_not_exist(ann_png_path)

    # 删除空文件夹
    # remove_empty_floders(nii_floders, dcm_floders)

    label_alllist = []
    n = 0
    num = len(os.listdir(nii_floders))

    for floder in os.listdir(nii_floders):  # 标签与数据对应

        n += 1
        if floder == '.DS_Store':
            continue

        dcm_floder = os.path.join(dcm_floders, floder)
        nii_floder = os.path.join(nii_floders, floder)

        # 处理dcm文件
        png_dic = dcm2png(dcm_floder)

        # 处理nii文件
        bounding_box_list, contours_list = get_nii_bounding_box(nii_floder)

        # 匹配dcm和nii文件的信息
        label_list, png_list = show_pngs_boxs(bounding_box_list, png_dic, contours_list, floder, png_path, ann_png_path)

        # 添加label
        label_alllist += label_list
        print(f'Save {floder} successful,Residual:{num - n}')  # 打印输出结果

    # 将label存储为pkl格式
    save_pkl(label_alllist, pkl_path, pkl_name)


def liu_main():

    # 初始化路径
    path = '/dicom/Adam/data/structural/2930liu'  # 主路径
    pkl_path = '/dicom/Adam/code/mmdetection/data/structrual/pkl/'# pkl文件的保存路径
    pkl_name = '20200525_liu'# 保存pkl的文件名
    png_path = '/dicom/Adam/code/mmdetection/data/structrual/val/'# dcm2png文件的保存路径
    ann_png_path = '/dicom/Adam/code/mmdetection/data/structrual/ann_liu/'# nii文件的标注绘制到dcm2png的图像保存路径
    # floder = '/home/ubuntu/Adam/data/structural/2930liu/20120604/1601757_1.2.840.31314.14143234.20120604080302.2690404'
    # with open(f'{pkl_path}{pkl_name}.pkl', 'wb') as f:
    #     f.close()

    label_alllist = []
    num = len(os.listdir(path))
    n = 0
    erro = []
    # with open(f'{pkl_path}{pkl_name}.pkl', 'ab') as f:
    for patient in os.listdir(path):  # 获取文件夹内每个患者的信息

        n += 1
        patient_path = os.path.join(path, patient)

        # try:
        for floders in os.listdir(patient_path):  # 获取每个患者中的每个文件夹

            floder = os.path.join(patient_path, floders)

            # 初始化数据
            png_dic = {}
            nii_files = []
            dcm_files = []
            bounding_box_list = 0
            contours_list = 0

            # 获取文件夹内的dcm和nii文件，将其分类保存。
            for file in os.listdir(floder):

                # 将xxxA.nii.gz的文件分类为nii文件
                if file.split('.')[-1] == 'gz':
                    if file.split('.')[-3][-1] == 'A':
                        nii_files.append(file)
                # 其余分类为dcm文件
                else:
                    dcm_files.append(file)

            # 获取nii文件的信息
            for nii_name in nii_files:
                bounding_box_list,contours_list,mask_list = get_nii_bounding_box(floder, nii_name=nii_name,save_mask=True)

            # 获取dcm的信息
            for index, dcm_name in enumerate(dcm_files):

                # 分解文件名
                dcm_first_name = dcm_name.split('.')[0]
                dcm_type = int(dcm_first_name.split('_')[0])

                # 选择2号文件进行解析
                if dcm_type == 2:
                    dcm_index = int(dcm_first_name.split('_')[1])
                    pos, img = dcm2png_liu(floder, dcm_name = dcm_name)
                    png_dic[dcm_index - 1] = [pos, img]

            # 将nii信息和dcm信息进行匹配
            label_list, png_list = show_pngs_boxs(bounding_box_list, png_dic, contours_list, floder, png_path, ann_png_path, save=True, draw=True,is_liu=True,mask_list=mask_list,save_mask=False)
            # print(len(label_list))
            # pickle.dump(label_list, f)

            # 添加label
            label_alllist += label_list
            print(f'Save {patient} successful,Residual:{num - n}')
            # if num - n ==65:
            #     print('prepearsave')
            #     save_pkl(label_alllist, pkl_path, pkl_name)
            #     print('savepkl')
        # except:
        #     erro.append(patient)
        #     print('erro:',patient)
        # 将label存储为pkl格式

    print(erro)
    val = label_alllist[:250]
    test = label_alllist[250:]
    save_pkl(val, pkl_path, 'val')
    save_pkl(test, pkl_path, 'test')
    print('save')
    # save_pkl(label_alllist, pkl_path, pkl_name)


if __name__ == '__main__':
    # liu_main()

    main()

    # remove_pkl_label()

    # # remove_png()
    # new_floder ='/dicom/Adam/code/mmdetection/data/structrual/pkl/'
    # new_png_path = '/dicom/Adam/code/mmdetection/data/structrual/val/'
    # # new_name ='20200525_liu'
    # new_name = 'val'
    # new_file = load_pkl(new_floder,new_name)

    # print(len(os.listdir(png_path)))
    # print(len(file))
    # floder = '/dicom/Adam/code/mmdetection/data/structrual/annotations/'
    # name = '20200520_142'
    # file = load_pkl(floder,name)
    # print(file[0]['ann']['bboxes'])
    # test= new_file[:250]
    # val = new_file[250:]
    # print(new_file[0])
    # exit()
    # save_pkl(test,new_floder,'test')
    # save_pkl(val, new_floder, 'val')
    # print(new_file[0]['ann']['bboxes'])
    # print()