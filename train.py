import numpy as np
from PIL import Image
import os
import time
import pylab as plt
import pickle
import tensorflow as tf
from tensorflow import keras
# from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import Input, Add, Conv2D, Conv3D, Concatenate, ConvLSTM2D, Dropout, BatchNormalization, \
    LeakyReLU, MaxPooling2D, UpSampling2D, TimeDistributed
from tensorflow.keras.optimizers import *
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging

# from werkzeug.datastructures import K

logging.getLogger('tensorflow').setLevel(logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NO WARNINGS!
print("[INFO] Imports loaded.")

# 数据加载和预处理 #
# DATA LOADING AND PROCESSING INTO NUMPY ARRAY #

DATA_PATH = 'C:/Users/84342/PycharmProjects/PrecipitationNowcasting/data/'

WIDTH = 116
HEIGHT = 116
# DATA_SEQUENCE = np.array([])
INPUT_SEQUENCE = np.array([])
NEXT_SEQUENCE = np.array([])
NUMBER = 0

os.chdir("data/")  # 设置工作目录

file_chdir = os.getcwd()  # 获得工作目录
# print(file_chdir)
filename_npy = []  # 文件名列表
# print(filename_npy)
file_npy = []  # 数据列表
# print(file_npy)
for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
    for file in files:
        if os.path.splitext(file)[-1] == '.npy':  # 判断文件格式是否符合npy格式
            filename_npy.append(file)  # 存储文件名
            file_npy.append(np.load(file))  # 存储数据
            NUMBER += 1
print(NUMBER)

DATA_SEQUENCE = np.array(file_npy)  # data就是所有数据的存储

print(DATA_SEQUENCE.shape)

# 格式序列 #
# LOADING OF NUMPY ARRAY INTO SEQUENCES #

FRAMES = 10  # frames to process

# DATA_SEQUENCE = DATA_SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)

INPUT_SEQUENCE = np.zeros((NUMBER - FRAMES, FRAMES, WIDTH, HEIGHT, 1), dtype=float)

NEXT_SEQUENCE = np.zeros((NUMBER - FRAMES, FRAMES, WIDTH, HEIGHT, 1), dtype=float)

for i in range(FRAMES):
    # print(i)
    INPUT_SEQUENCE[:, i, :, :, :] = DATA_SEQUENCE[i:i + NUMBER - FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = DATA_SEQUENCE[i + 1:i + NUMBER - FRAMES + 1]
print(INPUT_SEQUENCE.shape)
print(NEXT_SEQUENCE.shape)

x_train = INPUT_SEQUENCE[:int(0.8 * (NUMBER - FRAMES))]
y_train = NEXT_SEQUENCE[:int(0.8 * (NUMBER - FRAMES))]

x_val = INPUT_SEQUENCE[int(0.9 * (NUMBER - FRAMES)):]
y_val = NEXT_SEQUENCE[int(0.9 * (NUMBER - FRAMES)):]

# print('[INFO] InputSeq Statistics=%.3f (%.3f)' % (INPUT_SEQUENCE.mean(), INPUT_SEQUENCE.std()))
# print('[INFO] NextSeq Statistics=%.3f (%.3f)' % (NEXT_SEQUENCE.mean(), NEXT_SEQUENCE.std()))

print("[INFO] Input sequence ready")


# 模型 #
# NEW NETWORK #
def mainmodel():
    # Inputs
    dtype = 'float32'

    contentInput = Input(shape=(None, WIDTH, HEIGHT, 1), name='content_input', dtype=dtype)

    # Encoding Network
    x1 = ConvLSTM2D(4, (3, 3), padding='same', return_sequences=True, kernel_initializer='normal', activation='relu',
                    name='layer1')(
        contentInput)
    x2 = ConvLSTM2D(8, (3, 3), padding='same', return_sequences=True, kernel_initializer='normal', activation='relu',
                    name='layer2')(
        x1)

    # Forecasting Network
    x3 = ConvLSTM2D(8, (3, 3), padding='same', return_sequences=True, kernel_initializer='normal', activation='relu',
                    name='layer3')(
        x1)
    add1 = Add()([x3, x2])
    x4 = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True, kernel_initializer='normal', activation='relu',
                    name='layer4')(
        add1)

    # Prediction Network
    conc = Concatenate()([x4, x3])  #
    predictions = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='prediction')(
        conc)  # sigmoid original

    model = Model(inputs=contentInput, outputs=predictions)

    return model


# 模型
# NEW NETWORK
def test_model():
    # ConvLSTM Model Architecture
    model = Sequential()

    model.add(ConvLSTM2D(filters=2, kernel_size=(3, 3), input_shape=(None, WIDTH, HEIGHT, 1), padding='same',
                         return_sequences=True))

    # model.add(BatchNormalization())  # 使样本正则化符合正态分布，防止过拟合

    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    # model.add(Dropout(0.5))  # 让神经元有0.5的概率不激活 防止过拟合
    model.add(BatchNormalization())
    model.add(
        Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last'))

    return model


# 性能函数
# PLOT LOSS vs EPOCHS
def performance():
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig('results_Test.png', dpi=100)


# 训练函数
# Train model
def train(main_model=True, batchsize=5, epochs=30, save=False):
    smooth = 1e-9

    # Additional metrics: SSIM, PSNR, POD, FAR
    def ssim(x, y, max_val=1.0):
        return tf.image.ssim(x, y, max_val)

    # recall
    def CSI(x, y):
        y_pred_pos = keras.backend.clip(y, 0, 1)
        y_pos = keras.backend.clip(x, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_neg = 1 - y_pos
        tp = keras.backend.sum(y_pos * y_pred_pos)
        fn = keras.backend.sum(y_pos * y_pred_neg)
        fp = keras.backend.sum(y_neg * y_pred_pos)
        return (tp + smooth) / (tp + fn + fp + smooth)

    def POD(x, y):
        y_pos = keras.backend.clip(x, 0, 1)
        y_pred_pos = keras.backend.clip(y, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        tp = keras.backend.sum(y_pos * y_pred_pos)
        fn = keras.backend.sum(y_pos * y_pred_neg)
        return (tp + smooth) / (tp + fn + smooth)

    def FAR(x, y):
        y_pred_pos = keras.backend.clip(y, 0, 1)
        y_pos = keras.backend.clip(x, 0, 1)
        y_neg = 1 - y_pos
        tp = keras.backend.sum(y_pos * y_pred_pos)
        fp = keras.backend.sum(y_neg * y_pred_pos)
        return (fp) / (tp + fp + smooth)

    metrics = ['accuracy', CSI, POD, FAR, ssim]

    global history, model

    if main_model:
        model = mainmodel()
        model.summary()
        print("[INFO] Compiling Main Model...")
        optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='logcosh', optimizer=optim,
                      metrics=metrics)  # logcosh gives better results than crossentropy or mse
        print("[INFO] Compiling Main Model: DONE")
        print("[INFO] Training Main Model...")
        history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, validation_data=(x_val, y_val),
                            verbose=1, use_multiprocessing=True)
        print("[INFO] Training of Main Model: DONE")
        # Save trained model
        if save:
            print("[INFO] Saving Model...")
            # serialize model to JSON
            model_json = model.to_json()
            with open('ConvLSTM_model.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights('ConvLSTM_model.h5')
            print("[INFO] Model Saved")
            performance()
        else:
            print("[INFO] Model not saved")

    else:
        model = test_model()
        model.summary()
        print("[INFO] Compiling Test Model...")
        optim = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
        model.compile(loss='mse', optimizer=optim, metrics=metrics)
        print("[INFO] Compiling Test Model: DONE")
        print("[INFO] Training Test Model...:")
        # history = model.fit(INPUT_SEQUENCE[:40], NEXT_SEQUENCE[:40], batch_size=5, epochs=180, validation_split=0.05, verbose=1, use_multiprocessing=True)
        history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, validation_data=(x_val, y_val),
                            verbose=1, use_multiprocessing=True)
        print("[INFO] Training of Test Model: DONE")
        # Save trained model
        if save:
            print("[INFO] Saving Test Model...")
            model_json = model.to_json()
            with open('test_model.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save('test_model.h5')
            print("[INFO] Model Saved")
            performance()
        else:
            print("[INFO] Model not saved")


if __name__ == '__main__':
    train(main_model=False, batchsize=16, epochs=300, save=True)
