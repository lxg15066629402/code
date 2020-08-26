# coding: utf-8

'''
python+pandas 保存list到本地excel
'''

# 一维列表
# import pandas as pd
#
#
# def deal():
#     # 列表
#     company_name_list = ['腾讯', '阿里巴巴', '字节跳动', '腾讯']
#
#     # list转dataframe
#     df = pd.DataFrame(company_name_list, columns=['company_name'])
#
#     # 保存到本地excel
#     df.to_excel("company_name_li.xlsx", index=False)
#
#
# if __name__ == '__main__':
#     deal()

# # 二维列表
# import pandas as pd
#
#
# # python+pandas 保存list到本地
# def deal():
#     # 二维list
#     company_name_list = [['腾讯', '北京'], ['阿里巴巴', '杭州'], ['字节跳动', '北京']]
#
#     # list转dataframe
#     df = pd.DataFrame(company_name_list, columns=['company_name', 'local'])
#
#     # 保存到本地excel
#     df.to_excel("company_name_li.xlsx", index=False)
#
#
# if __name__ == '__main__':
#     deal()


# #  读取excel 到 list， 一维
# import pandas as pd
#
#
# def excel_one_line_to_list():
#     df = pd.read_excel("文件.xlsx", usecols=[1],
#                        names=None)  # 读取项目名称列,不要列名
#     df_li = df.values.tolist()
#     result = []
#     for s_li in df_li:
#         result.append(s_li[0])
#     print(result)
#
#
# if __name__ == '__main__':
#     excel_one_line_to_list()

# 读取两列
import pandas as pd


def excel_one_line_to_list():
    df = pd.read_excel("文件.xlsx", usecols=[1, 2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
    df_li = df.values.tolist()  # 将值转化为list 形式
    print(df_li)


if __name__ == '__main__':
    excel_one_line_to_list()


# os.chdir() 方法用于改变当前工作目录到指定的路径

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys

path = "/tmp"

# 查看当前工作目录
retval = os.getcwd()
print("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir(path)

# 查看修改后的工作目录
retval = os.getcwd()

print("目录修改成功 %s" % retval)
# 执行以上程序输出结果为：
#
# 当前工作目录为 /www
# 目录修改成功 /tmp
