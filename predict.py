from train import *
from radar_img_output import MatrixToImage

# 加载模型结构
json_file = open('C:/Users/84342/PycharmProjects/PrecipitationNowcasting/model/ConvLSTM_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# 加载模型权重
model.load_weights("C:/Users/84342/PycharmProjects/PrecipitationNowcasting/model/ConvLSTM_model.h5")
print("[INFO] Loaded model from disk")

# savepath = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/predict/'
savepath = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/test/'

def predict():
    # 十张预测一张
    which = 8
    track = INPUT_SEQUENCE[which][::, ::, ::, ::]

    # for j in range(FRAMES):
    #     new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    #     print(new_pos.shape)
    #     new = new_pos[::, -1, ::, ::, ::]
    #     print(new.shape)
    #     track = np.concatenate((track, new), axis=0)
    #     print(track.shape)

    new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    print(new_pos.shape)
    new = new_pos[::, -1, ::, ::, ::]
    print(new.shape)
    track = np.concatenate((track, new), axis=0)
    print(track.shape)

    # 保存预测图像与实际图像
    toplot = track[-1, ::, ::, 0]
    print('predict', toplot)
    outpath = savepath + '_predict_'
    MatrixToImage(toplot, outpath)
    toplot = NEXT_SEQUENCE[which][-1, ::, ::, 0]
    print('ground', toplot)
    outpath = savepath + '_real_'
    MatrixToImage(toplot, outpath)
    print("save")

    # for i in range(FRAMES):
    #     toplot = track[i, ::, ::, 0]
    #     if i >= 0:
    #       print('predict', i, toplot)
    #       outpath = savepath + str(i+1) + '_predict_'
    #       MatrixToImage(toplot, outpath)
    #     if i >= 0:
    #       toplot = NEXT_SEQUENCE[which][i - 1, ::, ::, 0]
    #       print('ground', i, toplot)
    #       outpath = savepath + str(i+1) + '_real_'
    #       MatrixToImage(toplot, outpath)
    #       print("save")

if __name__ == '__main__':
    predict()