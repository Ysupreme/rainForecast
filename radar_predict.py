import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from radar_data_pre import *
from radar_img_output import MatrixToImage

def radar_predict():
    # 加载模型结构
    json_file = open('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/model/ConvLSTM_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # 加载模型权重
    model.load_weights("C:/Users/84342/PycharmProjects/PrecipitationNowcasting/model/ConvLSTM_model.h5")
    print("[INFO] Loaded model from disk")

    savepath = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/'

    DATA_SEQUENCE = load_data(grdPath)

    # 十张预测一张
    track = DATA_SEQUENCE[::, ::, ::, ::]

    new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    print(new_pos.shape)
    new = new_pos[::, -1, ::, ::, ::]
    print(new.shape)
    track = np.concatenate((track, new), axis=0)
    print(track.shape)

    # 保存预测图像与实际图像
    toplot = track[-1, ::, ::, 0]
    print('predict', toplot)
    outpath = savepath + 'predict_'
    MatrixToImage(toplot, outpath)
    print("save")

    tmp_item = np.array(toplot, dtype=np.float64)

    # 保存降雨数据
    z_item = 10 ** (tmp_item / 10)
    out = np.exp(1000 / 1369 * np.log((z_item / 315.6) + 1e-5))
    rain = np.rint(out)
    print(rain.shape)
    rain = rain.reshape((116, 116))
    print(rain)
    np.savetxt('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/rain.txt', rain)

    pic_path = outpath + 'tmp_item.jpg'

    return pic_path


if __name__ == '__main__':
    radar_predict()