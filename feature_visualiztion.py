import os
import matplotlib
import tensorflow
import numpy as  np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
#加载模型
def load_model():
    model_vgg16=tensorflow.keras.applications.vgg16.VGG16(weights='imagenet')
    return model_vgg16

#查询卷积层的名称
def block_layers_16():
    model_vgg16=load_model()
    layer_outputs = [layer.output for layer in model_vgg16.layers[:20]] #前20个conv输出
    model = Model(inputs=model_vgg16.input, outputs=layer_outputs) #构建能够输出前20个conv层的模型
    return model

def featureVisualiztion(img_path):
    block_name=[layer.name for layer in load_model().layers[1:5]]
    model_vgg16 = block_layers_16()
    #将图像缩放到固定大小
    img=image.load_img(img_path,target_size=(224,224))
    #将图片转换为向量
    img=image.img_to_array(img)
    #对其维度进行扩充
    img=np.expand_dims(img,axis=0)
    #对输入到网络的图像进行处理
    output_img=preprocess_input(img)
    #预测图像
    features=model_vgg16.predict(output_img)
    feature1_shape=np.shape(features[0])
    print(feature1_shape)
    # model_vgg16.summary()
    #显示前五层的特征图
    for step,feature in enumerate(features[1:5]):
        #特征图的通道数//16=画图的行数
        rows=feature.shape[-1]//16
        subplots=np.zeros((int(feature.shape[1]*rows),int(16*feature.shape[1])))
        for row in range(rows):
            for col in range(16):
                #每一个通道的特征图
                feature_image=feature[0,:,:,row*16+col]
                subplots[row*feature.shape[1]:(row+1)*feature.shape[1],col*feature.shape[1]:(col+1)*feature.shape[1]]=feature_image
        scale=1.0/feature.shape[1]
        plt.figure(figsize=(scale*subplots.shape[1],scale*subplots.shape[0]))
        plt.title(block_name[step])
        plt.grid(False)
        plt.imshow(subplots,aspect='auto',cmap='viridis')
        plt.show()


if __name__ == '__main__':
    print('Pycharm')
    featureVisualiztion('cat.jpg')