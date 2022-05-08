import shutil
import os
import time

def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    number = 1
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)

        print(number)
        if number%2 == 0:
            print("-----------sleep-----------")
            time.sleep(20)
        number += 1


if __name__ == '__main__':
    remove_file("C:/Users/84342/Desktop/MCR/MCR", "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/MCR")