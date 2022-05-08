import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd

# # 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
np.set_printoptions(threshold = np.inf)
# 若想不以科学计数显示:
np.set_printoptions(suppress = True)

# os.chdir("C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data/")  # 设置工作目录
#
# file_chdir = os.getcwd()  # 获得工作目录
# print(file_chdir)
# filename_npy = []  # 文件名列表
# print(filename_npy)
# file_npy = []  # 数据列表
# print(file_npy)
# for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
#     for file in files:
#         if os.path.splitext(file)[-1] == '.npy':  # 判断文件格式是否符合npy格式
#             filename_npy.append(file)  # 存储文件名
#             file_npy.append(np.load(file))  # 存储数据
#
# data = file_npy  # data就是所有数据的存储
# print(data)
# np.save('test.npy', data)

tmp_item = np.load("C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data/20220413005400.npy")

tmp_item = np.array(tmp_item)

# tmp_item = tmp_item * 80
z_item = 10 ** (tmp_item / 10)
out = np.exp(1000 / 1369 * np.log((z_item / 315.6) + 1e-5))

# out = out.tolist()
#
#
# print(out)

# out.to_csv('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/rain.txt.csv',encoding='utf-8')
# test = pd.DataFrame(data=out)#数据有三列，列名分别为one,two,three
# test.to_csv('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/rain.csv',encoding='utf-8')


rain = np.rint(out)
print(rain.shape)
rain = rain.reshape((116, 116))
print(rain)
np.savetxt('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/rain.txt', rain)

