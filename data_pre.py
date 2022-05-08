# coding=gbk
import os
import time

import datetime
import struct
from goto import with_goto

# from mim import MIM

import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Subset, Dataset
import os
import argparse
import numpy as np
# from core.models.model_factory import Model
# from core.utils import preprocess

filePath = "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/MCR/"

storePath = "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/rain/"

dataPath = "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data/"

temporary = "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/img/"


def anadata(path):

    print("path:", path)

    with open(path, 'rb') as p:# 解析.grd雷达数据
        data = p.read()
        print(len(data))
        data_raw = struct.unpack("f" * (len(data) // 4), data)
        data_new = np.asarray(data_raw).reshape(1500, 1800)
        data_np = data_new
        data_np[data_np < 0] = 0 #小于0的雷达回拨数据记为0
        data_np[data_np > 80] = 80# 大于八十的记为八十

        # print(data_np.shape)

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

        print(data_5km.shape)
        img = Image.fromarray(data_5km)
        IM = img.convert("L")
        IM.save(temporary +"/"+ path[-18:-4] + ".png")

        global fileName
        fileName = path[-18:-4]

        return data_5km


def getDatas(filePaths):
    data = None
    for filePath in filePaths:
        tmp = anadata(filePath)
        tmp = np.expand_dims(tmp, 0)
        if data is None:
            data = tmp
        else:
            data = np.vstack((data, tmp))
    data = np.expand_dims(data, 1)
    data = np.expand_dims(data, 0)
    return data


def np2Tensor(data, devices):
    return torch.from_numpy(data).to(devices).unsqueeze(0).float()


def time_add(stime, rtimeint):
    stime = stime[0: 4] + "-" + stime[4: 6] + "-" + stime[6: 8] + "-" + stime[8: 10]
    dTime = datetime.datetime.strptime(stime, '%Y-%m-%d-%H')

    dTime += datetime.timedelta(hours=rtimeint)
    stime = dTime.strftime("%Y%m%d%H")
    return stime


def create_m4(stime, prehour, result):
    rtimeint = int(prehour)
    timestr = time_add(stime, rtimeint)[4:]
    global storePath
    with open(storePath + 'RAT_' + stime + '.' + prehour, 'w') as file:
        file.write('diamond 4 rain_0-1_hours_predict_' + stime + '.' + prehour + ':' + timestr + '\n')
        file.write('' + stime[:4] + ' ' + stime[4:6] + ' ' + stime[6:8] + ' ' + stime[8:10] + ' 24 0.0\n')
        file.write('0.05 0.05 108.65 114.4 24.5 30.25 116 116 0.1 -0.1 1.1 1 0\n')
        for i in range(116):
            if i == 115:
                for j in range(116):
                    if j == 115:
                        file.write(str(result[i][j]))
                    else:
                        file.write(str(result[i][j]) + '  ')
            else:
                for j in range(116):
                    if j == 115:
                        file.write(str(result[i][j]))
                    else:
                        file.write(str(result[i][j]) + '  ')
                file.write('\n')


@with_goto
def watch_dir(path):
    added_list = []
    sleepTime = 1
    before = dict([(f, None) for f in os.listdir(path)])
    print("before:", before)
    print("watch:", path)
    count = 0
    while 1:
        after = dict([(f, None) for f in os.listdir(path)])
        print("after:", after)
        added = [f for f in after if not f in before]
        print("added:", added)
        if added:
            # print("added:",added, added[0])
            added_list = added
            # print("hour time:",str(added_list[0][11:-4]))
            start_file = added_list[0]
            start_time = start_file.split('.')[0]
            start_time = str(start_time[6:])
            print("hour list:", added_list[0])
            print("hour time:", start_time)
            hour_time = time.strptime(start_time, "%Y%m%d%H%M%S")
            hour_time = time.mktime(hour_time)
            hour_time = int(hour_time / 60 / 60) * 60 * 60
            # print("hour time:", hour_time)
            for i in range(0, 60, 6):  # range(start, stop [,step])
                tmp = hour_time + 60 * i
                tmp = time.localtime(tmp)
                tmp_ctl = added_list[0][:6] + time.strftime("%Y%m%d%H%M%S", tmp) + ".ctl"
                if tmp_ctl not in added_list:
                    if os.path.exists(os.path.join(path, tmp_ctl)) is True:
                        print(tmp_ctl)
                        added_list.append(tmp_ctl)
                tmp_grd = added_list[0][:6] + time.strftime("%Y%m%d%H%M%S", tmp) + ".grd"
                if tmp_grd not in added_list:
                    if os.path.exists(os.path.join(path, tmp_grd)) is True:
                        print(tmp_grd)
                        added_list.append(tmp_grd)

            if len(added_list) != 20:
            # if len(added_list) != 10:
                # added_list = []
                count += 1
                if count <= 10:
                    print("-----------sleep-----------")
                    time.sleep(sleepTime)
                    continue

            else:
                goto.begin
                print("goto.begin")
        # print("count:", count)
        print("pya", added_list)
        if count > 10:
            label.begin
            print("label.begin")
            for added_list_item in added_list[:]:
                if added_list_item[-4:] == ".ctl":
                    # print("rm:", added_list_item)
                    added_list.remove(added_list_item)
            # print("added_list---2:", added_list)
            hour_time = time.strptime(str(added_list[0][6:-4]), "%Y%m%d%H%M%S")
            hour_time = time.mktime(hour_time)
            hour_time = int(hour_time / 60 / 60) * 60 * 60
            tmp_list = []
            for i in range(0, 60, 6):
                tmp = hour_time + 60 * i
                tmp = time.localtime(tmp)
                tmp_grd = added_list[0][:6] + time.strftime("%Y%m%d%H%M%S", tmp) + ".grd"
                if tmp_grd not in added_list or os.path.getsize(os.path.join(path, tmp_grd)) < 1 * 1024 * 1024:
                    if tmp_grd in added_list and os.path.getsize(os.path.join(path, tmp_grd)) < 1 * 1024 * 1024:
                        added_list.remove(tmp_grd)
                    if i != 0:
                        added_list.append(tmp_list[-1])
                    else:
                        tmp_time = -6
                        while True:
                            print("pya")
                            tmp = hour_time + 60 * tmp_time
                            tmp = time.localtime(tmp)
                            tmp_grd = added_list[0][:6] + time.strftime("%Y%m%d%H%M%S", tmp) + ".grd"
                            if os.path.exists(os.path.join(path, tmp_grd)) is True and os.path.getsize(
                                    os.path.join(path, tmp_grd)) > 1 * 1024 * 1024:
                                tmp_list.append(tmp_grd)
                                added_list.append(tmp_grd)
                                break
                            else:
                                tmp_time -= 6
                else:
                    tmp_list.append(tmp_grd)
            count = 0
            break
    added_list.sort()
    for i in range(len(added_list)):
        added_list[i] = os.path.join(path, added_list[i])
    for added_list_item in added_list[:]:
        if added_list_item[-4:] == ".ctl":
            # print("rm:", added_list_item)
            added_list.remove(added_list_item)
    while len(added_list) > 10:
        added_list.pop()

    print(added_list)
    return added_list
    # before = after


if __name__ == '__main__':

    datasLength = 12

    device = torch.device("cpu")

    # shape = [1, 10, 1, 116, 116]
    # numlayers = 4
    # net = NewNet(UNet(64, 2)).to(device)
    # model = RNN(shape, numlayers, [64, 64, 64, 64], 10, 64, 2, net, True).to(device)
    # model = None
    # statePath = "/home/shida_user/hdy_competition/rain/loss_record/pred/trainnewep100_bs1_hs64_nl4/epoch99loss_train_best=0.0007833254220796251.pkl"

    parser = argparse.ArgumentParser(description='PyTorch video prediction model - MIM')
    parser.add_argument('--model_name', type=str, default='mim')
    parser.add_argument('--pretrained_model', type=str,
                        default='')
    parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--decouple_beta', type=float, default=0.1)
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--img_width', type=int, default=116)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--Loader', type=str, default='csc')
    args = parser.parse_args()
    # model = Model(args)
    # model.load(args.pretrained_model)
    # model.eval()

    while (True):

        filePaths = watch_dir(filePath)

        if filePaths is None:
            continue

        print("reading data")
        datas = getDatas(filePaths)
        # print(datas.shape)
        try:
            tensorDatas = np2Tensor(datas, device)
        except Exception as e:
            print(e)
            continue
        if len(tensorDatas) != datasLength:
            print("data error, abandon prediction")
            print(len(tensorDatas), datasLength, tensorDatas[0].size())

        print("predicting data")
        # modelRes = tensorDatas
        modelRes = []
        for tensorData in tensorDatas:
            # print("input shape:",tensorData.shape)
            data = tensorData.permute(0, 1, 3, 4, 2)
            print(data.shape)
            # ims = data.cpu().numpy()
            # ims = preprocess.reshape_patch(ims, args.patch_size)
            # tmp = model.test(ims)
            # # tmp = torch.nn.functional.softmax(tmp, dim=1)
            # # tmp = torch.max(tmp, 1)[1].data.to('cpu').numpy().squeeze()
            # tmp = preprocess.reshape_patch_back(tmp, args.patch_size)  # (1,19,116,116,1)
            # tmp = np.transpose(tmp, (0, 1, 4, 2, 3))  # (1,19,1,116,116)
            # tmp = np.squeeze(tmp)  # (19,116,116)
            # tmp = tmp[-10:]  # (10,116,116)
            final = np.zeros((116, 116))
            for i in range(10):
                tmp_item = data[0, i, :, :]

                tmp_item[tmp_item < 0] = 0
                tmp_item[tmp_item > 80] = 80

                radar_data = tmp_item.cpu().detach().numpy()  # tensor转换为ndarray
                print(fileName)
                outfile = dataPath + "/" + fileName + ".npy"
                # np.save(outfile, tmp_item)
                np.save(outfile, radar_data)

                # radar_data = np.transpose(radar_data, (2, 0, 1))
                # im = Image.fromarray(tmp_item.numpy())
                # im = im.convert('L')
                # im.save(temporary + "/" + i + ".png")

                tmp_item = tmp_item * 80
                z_item = 10 ** (tmp_item / 10)
                out = np.exp(1000 / 1369 * np.log((z_item / 315.6) + 1e-5))

                final = np.add(final, out)

            # tmp = torch.squeeze(tmp, dim=0)
            # tmp = tmp.cpu().detach().numpy()
            final = np.where(final > 20, -1, 63)
            print("final shape:", final.shape)
            modelRes.append(final)

        print("saving data")
        for i in range(len(modelRes)):
            _, fileName = os.path.split(filePaths[0])

            stime, ext = os.path.splitext(fileName)
            stime = stime[6:-4]

            stime = time_add(stime, 1)

            ext = "001"

            create_m4(stime, ext, modelRes[i])


