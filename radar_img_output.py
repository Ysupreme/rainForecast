import numpy as np
import struct
import os
from matplotlib import pyplot as plt
from PIL import Image
from pylab import *

# with open('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/GUI_grd/Z_MCR_20220315031200.grd', 'rb') as p:  # 解析.grd雷达数据
#     data = p.read()
#     print(len(data))
#     data_raw = struct.unpack("f" * (len(data) // 4), data)
#     data_new = np.asarray(data_raw).reshape(1500, 1800)
#     data_np = data_new
#     data_np[data_np < 0] = 0  # 小于0的雷达回拨数据记为0
#     data_np[data_np > 80] = 80  # 大于八十的记为八十


def MatrixToImage(data, sp):
    data = data.reshape(116 * 116)  # 将index层的（二维）数据变成一维
    # data = data.reshape(1500 * 1800)  # 将index层的（二维）数据变成一维
    d = []
    # 上色
    for num in data:  # num是data里的每个点
        if num <= 3:
            d.append([0, 0, 0])  # 如果该点的雷达回波强度小于等于0，则设置黑色
        if 3 < num <= 5:
            d.append([0, 157, 255])
        if 5 < num <= 10:
            d.append([3, 0, 244])
        if 10 < num <= 15:
            d.append([1, 160, 246])
        if 15 < num <= 20:
            d.append([0, 236, 236])
        if 20 < num <= 25:
            d.append([0, 216, 0])
        if 25 < num <= 30:
            d.append([1, 144, 0])
        if 30 < num <= 35:
            d.append([255, 255, 0])
        if 35 < num <= 40:
            d.append([231, 192, 0])
        if 40 < num <= 45:
            d.append([255, 144, 0])
        if 45 < num <= 50:
            d.append([255, 0, 0])
        if 50 < num <= 55:
            d.append([214, 0, 0])
        if 55 < num <= 60:
            d.append([192, 0, 0])
        if 60 < num <= 65:
            d.append([255, 0, 240])
        if num > 65:
            d.append([150, 0, 180])
    d = np.asarray(d).reshape(116, 116, 3)
    # d = np.asarray(d).reshape(1500, 1800, 3)
    # print(d)
    d = array(d)
    new_im = Image.fromarray(d.astype('uint8')).convert('RGB')
    new_im.save(sp + 'tmp_item.jpg')

if __name__ == '__main__':
    data = np.load("C:/Users/84342/PycharmProjects/PrecipitationNowcasting/GUI_grd/20220415193600.npy")
    MatrixToImage(data, "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/")