"""
# 合并字典
"""

import json
import os


# save json file, obj 目标数据
def save_json(obj, folder, name):
    with open(folder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


# load json file
def load_json(file_json):
    with open(file_json, 'r', encoding='utf8') as f:
        data = json.load(f)
        return data


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 创建文件夹
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations'
    save_json_name = 'all_all_json'

    create_dir_not_exists(save_json_path)
    create_dir_not_exist(save_json_name)

    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/save_data/diff_json/annotations/coco_json.json"
    # # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/liu_json/annotations/liu_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/liu_json/annotations/liu_1_json.json"
    # # json_2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json/annotations/wan1_json.json"
    # json_2 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan1_json/annotations/wan1_1_json.json"
    # # json_3 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json/annotations/wan2_json.json"
    # json_3 = "/dicom/Jone/code/data_pre/stru_data_process/same/wan2_json/annotations/wan2_1_json.json"
    # # json_4_4 = "/dicom/Jone/code/data_pre/stru_data_process/same/yao_json/annotations/yao_json.json"
    # json_4 = "/dicom/Jone/code/data_pre/stru_data_process/same/yao_json/annotations/yao_1_json.json"

    # json_0 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/coco_1_json.json"
    # json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/liu_1_json.json"
    # json_2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/wan1_1_json.json"
    # json_3 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/wan2_2_json.json"
    # json_4 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/yao_1_json.json"
    # test = "/dicom/Jone/Dection_mass/data/structrual/annotations/mask_coco_train.json"
    # #
    # test1 = load_json(test)
    #
    # print(len(test1["annotations"]))
    # print(len(test1["images"]))
    # # print((test1["images"]))
    # print(test1["annotations"][1])
    # exit()
    # dict_4 = load_json(json_4_4)
    # test_4 = dict_4["annotations"]
    # print((dict_4["annotations"][1]))
    #
    # exit()

    dict_all = {}  # 定义一个空字典

    # dict0 = load_json(json_0)
    # print('这是文件中的json数据：', dict0)
    # print(type(dict0))
    # print(dict0["images"])
    # print(len(dict_all))
    # print(len(dict0["images"]))
    # print(len(dict0["annotations"]))
    # print(len(dict0["categories"]))
    # exit()

    # merge all 815 张数据
    json_0 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/coco_1_json.json"
    json_1 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/liu_json.json"
    json_2 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/wan1_json.json"
    json_3 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/wan2_json.json"
    json_4 = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/annotations/yao_json.json"

    dict0 = load_json(json_0)
    dict1 = load_json(json_1)
    dict2 = load_json(json_2)
    dict3 = load_json(json_3)
    dict4 = load_json(json_4)


    # dict_all.update(dict0)
    # dict_all.update(dict1)
    # dict_all.update(dict2)
    # dict_all.update(dict3)
    # dict_all.update(dict4)

    #
    # print(len(dict0))
    # print("\n")
    # print(dict1["images"])
    a = dict0["images"]
    b = dict1["images"]
    c = dict2["images"]
    d = dict3["images"]
    e = dict4["images"]

    a_image = a + b + c + d + e
    # a_image = b + c + d + e

    al = dict0["annotations"]
    bl = dict1["annotations"]
    cl = dict2["annotations"]
    dl = dict3["annotations"]
    el = dict4["annotations"]

    # al_ann = bl + cl + dl + el
    al_ann = al + bl + cl + dl + el
    # al_ann = al

    # categories

    # ac = dict0["categories"]
    ac = dict1["categories"]
    # bc = dict1["categories"]
    # cc = dict2["categories"]
    # dc = dict3["categories"]
    # ec = dict4["categories"]

    # ac_ca = ac + bc + cc + dc + ec
    ac_ca = ac

    dict_all["images"] = a_image
    dict_all["annotations"] = al_ann
    dict_all["categories"] = ac_ca

    # print(len(dict_all))
    # # print(len(dict_all["images"]))
    # name = []
    # print(dict_all["images"])
    # c = 0
    # for i in range(len(dict_all["images"])):
    #     # print(dict_all["images"][i])
    #     if dict_all["images"][i]['file_name'] in name:
    #         print(dict_all["images"][i]['file_name'])
    #         c += 1
    #     name.append(dict_all["images"][i]['file_name'])
    #
    # print(c)
    # print(len(dict_all))
    # print(dict_all)
    # print(len(dict_all["annotations"]))
    # print((dict_all["annotations"][1165]))
    # print(len(dict_all["categories"]))

    dict_all = json.dumps(dict_all)
    save_json(dict_all, save_json_path, save_json_name)

    print("save success!!")