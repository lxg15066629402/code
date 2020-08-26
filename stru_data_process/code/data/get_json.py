# coding: utf-8
"""
处理json 文件，进行数据整合
"""
import json


# save json file, obj 目标数据
def save_json(obj, folder, name):
    with open(folder + '/' + name + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json.loads(obj), ensure_ascii=False, indent=2))


def load_json(json_file):
    # 读取json文件内容,返回字典格式
    with open(json_file, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        # print(json_data)
    return json_data


def get_check(check_json):

    check_data = load_json(check_json)
    check_all_file = []
    for i, index in enumerate(check_data):

        check_all_file.append(index[0])

    # print(len(check_all_file))
    return check_all_file


def analyze_json(json_data, check_all):

    json_keys = list(json_data.keys())
    # print(json_keys)  # ['image', 'annotations', 'categories']
    # namelist = [200]
    image = json_data[json_keys[0]]
    ann = json_data[json_keys[1]]
    # 定义一个字典，用于保存符合的json 文件
    dict_check = {}  #
    dict_check['images'] = []
    dict_check['annotations'] = []
    for idx, data in enumerate(image):
        # print(index)
        name = data['file_name']
        # print(name)
        if name in check_all:
            dict_check['images'].append(data)
            # record_id.append(data['id'])
            record_id = data['id']
            for ann_idx, ann_data in enumerate(ann):
                id = ann_data["image_id"]
                if record_id == id:
                    dict_check['annotations'].append(ann_data)

    dict_check['categories'] = {'id': 0, 'name': 'struct'}

    # print(dict_check)
    # print(len(dict_check['image']))
    # print(len(dict_check['annotations']))
    # exit()
    return dict_check


# # if '20130507_1987422_RML.png' in img_index["file_name"]:
#         #     # print(index['id'])
#         #     record_id.append(img_index['id'])
#         #     # save img
#         #     json_data[json_keys[0]] =
# def analyze_ann_json(json_data, record_id):
#     json_keys = list(json_data.keys())
#     for ann_index in json_data[json_keys[1]]:
#         for id_index in record_id:
#             if id_index in ann_index['image_id']:
#                 # save ann
#                 pass


if __name__ == "__main__":
    # 01 al
    # json_file = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/all_json.json"
    # check_json = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test1/check.json"
    # # #
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test1'
    #
    # save_json_name = 'check_json'

    # 02
    # json_file = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/all_json.json"
    # check_json = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test2/check.json"
    #
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test2'
    #
    # save_json_name = 'check_json'

    # 03
    # json_file = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data/annotations/all_json.json"
    # check_json = "/dicom/Jone/code/data_pre/stru_data_process/test_al/test3/check.json"
    #
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/test_al/test3'
    #
    # save_json_name = 'check_json'

    # shuffle data
    # json_file = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/train_json.json"
    # check_json = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1/check.json"
    #
    # save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test1'
    #
    # save_json_name = 'check_json'

    json_file = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/train_json.json"
    check_json = "/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2/check.json"

    save_json_path = '/dicom/Jone/code/data_pre/stru_data_process/same/all_data_all/train_1/AL1/test2'

    save_json_name = 'check_json'

    check_all_file = get_check(check_json)
    json_data = load_json(json_file)
    dict_check = analyze_json(json_data, check_all_file)
    dict_check = json.dumps(dict_check)
    save_json(dict_check, save_json_path, save_json_name)
