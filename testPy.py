

import numpy as np

import tensorflow as tf
import cv2 
from models.common import C3,StemBlock,Conv,SPP,Concat,MyDenseLayer
from tensorflow.keras.layers import Dense,Conv2D,BatchNormalization,UpSampling2D
from tensorflow.keras.regularizers import L2
import torch.backends.cudnn
from torch.nn import Upsample
from torch import nn
import torch
from models import yolo
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch.distributed as dist
from icecream import ic
from utils import tfrecord_utils
from utils.tensorflow_utils import initiaLize_weights

import pynvml

from tensorflow.keras import Sequential

a=np.zeros([1,2,3],dtype=np.int32)
b=np.zeros([4,5,6],dtype=np.int32)

print(a[b].shape)

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

bce = tf.losses.BinaryCrossentropy(from_logits = True)


write =tf.io.TFRecordWriter('./tf2.tfrecords')


x = tf.train.Example(features=tf.train.Features(feature={

    "width":tf.train.Feature(float_list=tf.train.FloatList(value= [1.1])),
    "height":tf.train.Feature(float_list=tf.train.FloatList(value =  [1.1])),
    "channels":tf.train.Feature(float_list=tf.train.FloatList(value =  [1.1])),
}))



exampel = tf.train.Example(features=tf.train.Features(feature={
    
    # 长、宽、通道数
    "height": tf.train.Feature(float_list=tf.train.FloatList(value=[28.])),
    "width": tf.train.Feature(float_list=tf.train.FloatList(value=[28.,28.])),
    "channels": tf.train.Feature(float_list=tf.train.FloatList(value=[3.,3.])),
}))

def decode_fn(record_byte):
    
    # print(type(record_byte))
    
    # e = {'image': tf.FixedLenFeature((), tf.string),
    #                'height': tf.FixedLenFeature((), tf.int64),
    #                'width': tf.FixedLenFeature((), tf.int64),
    #                'channels': tf.FixedLenFeature((), tf.int64),
    #                'label': tf.FixedLenFeature((), tf.int64)
    #                }
    feature_internal = {
        "width":tf.io.FixedLenFeature([1],dtype=tf.float32,default_value=0),
        "height":tf.io.FixedLenFeature([1],dtype=tf.float32,default_value=0),
        "channels":tf.io.FixedLenFeature([1],dtype=tf.float32,default_value=0)
        }
    # print(record_byte)
    return tf.io.parse_single_example(record_byte,feature_internal)

write.write(x.SerializeToString())
write.write(x.SerializeToString())
write.write(x.SerializeToString())
write.close()
print('打印')
ash=[]
for b in tf.data.TFRecordDataset(['./tf2.tfrecords']).map(decode_fn):
    ash.append(b['height'])
    
    
    print(b['height'].numpy())
    # write.remove(b)
    # print(b)


c={}

c['x'] = 'x'
print(c['x'])
print(pynvml.nvmlDeviceGetMemoryInfo(handle).used/ 1E9)
# 1268502528
# 2147483648
# 875573248
import math

from random import uniform

from utils.face_datasets import LoadFaceImagesAndLabels

# l = LoadFaceImagesAndLabels()
# print(len(l))
import torch.optim as optim

from tensorflow.keras.optimizers import Adam,SGD

x = tf.constant([1.0472, 0.7853])
# print(x)
y = tf.math.atan(x) # [1.731261, 0.99920404]
# print(y)
# print(tf.math.atan(y)) # [1.047, 0.785] = x


xp = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
yp = [[10,2,1,0],[10,2,1,0],[10,2,1,0],[10,2,1,0],[10,2,1,0]]

bce(np.array(xp,np.float32),np.array(yp,np.float32),(1.,0.5,2.0,2.0,2.0))

print(tf.atan(np.array(yp,dtype=np.float32)))
tf.math.square
print(tf.square(np.array(xp,dtype = np.float32)))
print( np.array(xp,dtype=np.float32) ** 2)


print(xp > [5,4,3,2] )
print(xp[True])

# print(np.interp(1.5,xp,yp))


x = 150
s = 32
 
n1 = np.eye(3,1)

decay=0.999999
decay = lambda x : decay * (1 - math.exp(-x / 2000))



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# 用这个替代expand
c      = tf.convert_to_tensor(np.array(range(16)))



c      = tf.reshape(c,(1,1,2,8))
output = tf.tile(c,(1,3,3,1))


cv = cv2.imread('d:/1.jpg')

# print(np.max(cv))
print("打印")
r = torch.from_numpy(cv)
print(1. /r )
print(np.max(np.maximum(cv, 1. /cv),2).shape)


cv = cv2.resize(cv,[4,4])
c = 1<2
print(cv[:,:,False::True])
print(cv[:,:,False::True].shape)


print('测试')
print(cv)
print(cv.shape)
print('测试')
c1=UpSampling2D(eval('[2,2]'),eval('None'),'bilinear')

cv = np.reshape(cv,(1,4,4,3))


print(c1.call(tf.convert_to_tensor(cv)))
print(c1.call(tf.convert_to_tensor(cv)).shape)

print(eval('None'))
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

a=eval('UpSampling2D')
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.weight = torch.nn.Parameter(torch.zeros(input_dim, output_dim))
        self.linear.bias = torch.nn.Parameter(torch.ones(output_dim))
        
 
    def forward(self, input_array):
        output = self.linear(input_array)
        return output


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)



n= np.array(cv,dtype=float)
# print(n)


n = tf.zeros((245,245,2))




x = Conv2D(10, (1,1), (1,1), 'same', use_bias=False)

q=['xih1','asd','asdzxc','ads']
q.insert(2,'测试')



# print(q)
n=3
n= max(round(n * 1.0), 1) if n > 1 else n
# print(n)


import yaml
# with open('F:\demopy\yolov5-face-tensorflow\models\yolov5s.yaml') as f:
# ya = yaml.load(f,Loader=yaml.FullLoader)



y=yolo.Model(
    # cfg=ya
    cfg='F:/demopy/yolov5-face-tensorflow/models/yolov5s.yaml'
    )
# for layer in y.layers:
#     	layer.trainable = False
# y.add_loss(tf.reduce_sum(outputs))


# 配置图片混淆器，把图片搞的乱七八糟
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# y.compile(loss='sparse_categorial_crossentropy', optimizer='adam')
# plot_model(y, to_file='VGG16_1D_简单测试模型_001.png', show_shapes=True)

BS = 32

# H = y.fit()

# y.fit(dataset, epochs=2, steps_per_epoch=10)
# print(y(np.random.randint(low=0, high=255, size=(96, 640, 480, 3)).astype('float32')))


# y.save('ceshi.h5')
# print('h5')
s=np.array([[[[1,2,100,1],[1,2,100,1],[1,2,100,1],[1,2,100,1]]]],dtype=np.float32)
print('结束')
print(s)
print('结束')

s= np.expand_dims(s,2)

print(s)

list = []





# print(nn.Conv2d(1,1,1,1,1,1,bias=True).requires_grad_(False).weight)
# a=tf.reshape(tf.convert_to_tensor(list,dtype=tf.float32),(1,56,56,3))
# print(a.shape)

# la = layers.Conv2D(1, (3,3), (1,1), use_bias=False,
                    #   kernel_initializer=tf.random_normal_initializer(stddev=0.01),groups=1, kernel_regularizer=L2(4e-5),padding='same'  )
# print(la(a).shape)
print('结束')




net = NeuralNetwork(4, 6)
# print(net.named_parameters)
# for param in net.parameters():
    # print(param)


# a= np.reshape(np.array(list,dtype=np.float32),[3,3,-1])
# a= tf.reshape(a,(3,85))


# print('测试np')
# print(a.size)




# [[[0 1][4 5]]
#  [[2 3][6 7]]]

import os 


dist.init_process_group(backend='nccl', init_method='env://',rank = 1,world_size=1)  # distributed backend
b = torch.tensor(list)
# print(b)
# k = dist.broadcast(b,0)
# print(k)
print('测试')
print(b)
print(b.expand(3,3,5,5))



handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo)