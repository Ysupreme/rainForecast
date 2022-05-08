import os
import random
import shutil

def moveFile(train_data_Dir):
        data_pathDir = os.listdir(train_data_Dir)                    # 提取图片的原始路径
        filenumber = len(data_pathDir)
        # 自定义test的数据比例
        test_rate = 0.1                                            # 如0.1，就是10%的意思
        test_picknumber = int(filenumber*test_rate)                # 按照test_rate比例从文件夹中取一定数量图片
        # 自定义val的数据比例
        val_rate = 0.1
        val_picknumber = int(filenumber*val_rate)                  # 按照val_rate比例从文件夹中取一定数量图片
        # 选取移动到test中的样本
        sample1 = random.sample(data_pathDir, test_picknumber)      # 随机选取picknumber数量的样本图片
        print(sample1)
        for i in range(0, len(sample1)):
            sample1[i] = sample1[i][:-4]                           # 去掉图片的拓展名，移动标注时需要这个列表
        for name in sample1:
            src_data_name1 = train_data_Dir + name
            dst_data_name1 = test_data_Dir + name
            shutil.move(src_data_name1 + '.npy', dst_data_name1 + '.npy')     # 加上图片的拓展名，移动图片
        # 选取移动到val中的样本
        data_pathDir = os.listdir(train_data_Dir)                    # 这时图片目录里的文件数目会变
        sample2 = random.sample(data_pathDir, val_picknumber)       # 但是抽出来的数目，还是用之前算的
        print(sample2)
        for i in range(0, len(sample2)):
            sample2[i] = sample2[i][:-4]
        for name in sample2:
            src_img_name2 = train_data_Dir + name
            dst_img_name2 = val_data_Dir + name
            shutil.move(src_img_name2 + '.npy', dst_img_name2 + '.npy')
        return

if __name__ == '__main__':
    # train 从train中移动
    train_data_Dir = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data_split/train/'
    # test路径：图片和标注目录
    test_data_Dir = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data_split/test/'
    # val路径：图片和标注文目录
    val_data_Dir = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data_split/val/'
    # 运行划分数据集函数
    moveFile(train_data_Dir)