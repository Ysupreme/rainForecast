import os
import struct

from PIL import Image
import os
import numpy as np
import tensorflow as tf

grdPath = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/GUI_grd/'

def anadata(path):

    print("path:", path)

    with open(path, 'rb') as p:# 解析.grd雷达数据
        data = p.read()
        data_raw = struct.unpack("f" * (len(data) // 4), data)
        data_new = np.asarray(data_raw).reshape(1500, 1800)
        data_np = data_new
        data_np[data_np < 0] = 0 #小于0的雷达回拨数据记为0
        data_np[data_np > 80] = 80# 大于八十的记为八十

        list1 = []
        list2 = []
        for j1 in range(len(data_np[0])):
            if j1 % 5 != 0:
                list1.append(j1)
        for j2 in range(len(data_np)):
            if j2 % 5 != 0:
                list2.append(j2)
        data_5km = np.delete(data_np, list1, 1)
        data_5km = np.delete(data_5km, list2, 0)
        data_5km = data_5km[133:249, 90:206]

        global fileName
        fileName = path[-18:-4]
        print(data_5km)
        return data_5km

def get_datas(filePath):
    tmp = anadata(filePath)
    tmp = np.expand_dims(tmp, axis=2)
    return tmp

def save_files(filePath):
    os.chdir(filePath)  # 设置工作目录
    file_chdir = os.getcwd()  # 获得工作目录
    # print(file_chdir)
    filename_npy = []  # 文件名列表
    # print(filename_npy)
    # file_npy = []  # 数据列表
    # print(file_npy)
    for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
        for file in files:
            if os.path.splitext(file)[-1] == '.grd':  # 判断文件格式是否符合grd格式
                filename_npy.append(file)  # 存储文件名
                print(filename_npy)
                tmp_item = get_datas(filename_npy[-1])
                # print(tmp_item.shape)
                # print(tmp_item)
                # print('---------')
                # file_npy.append(np.load(file))  # 存储数据
                outfile = filePath + "/" + fileName + ".npy"
                np.save(outfile, tmp_item)

def load_data(filePath):
    WIDTH = 116
    HEIGHT = 116
    NUMBER = 0
    save_files(filePath)
    os.chdir(filePath)  # 设置工作目录
    file_chdir = os.getcwd()  # 获得工作目录
    filename_npy = []  # 文件名列表
    file_npy = []  # 数据列表
    for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
        for file in files:
            if os.path.splitext(file)[-1] == '.npy':  # 判断文件格式是否符合npy格式
                filename_npy.append(file)  # 存储文件名
                file_npy.append(np.load(file))  # 存储数据
                NUMBER += 1
    print(NUMBER)

    DATA_SEQUENCE = np.array(file_npy)  # data就是所有数据的存储

    print(DATA_SEQUENCE.shape)

    # FRAMES = 10

    DATA_SEQUENCE = DATA_SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)

    print(DATA_SEQUENCE.shape)

    return DATA_SEQUENCE


if __name__ == '__main__':
    load_data(grdPath)

