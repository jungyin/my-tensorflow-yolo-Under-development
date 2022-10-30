import torch 
import sys
from torch.cuda.amp.common import amp_definitely_not_available
import numpy as np
import tensorflow as tf
import math
import cv2
from tensorflow.keras import losses


a = tf.convert_to_tensor([[1.0,0.0,0.0,1.0,0.0]])
b = tf.convert_to_tensor([[1.0,0.0,0.0,0.3,0.0]])

c=losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(a,b)


def moveleft(cv,moveValue):
    cv1 = np.zeros_like(cv)
    cv1[::,0:abs(moveValue-cv1.shape[1]),::] = cv[::,moveValue:cv1.shape[1],::]
    # return cv * np.array([[moveValue,0,0],[0,moveValue,0],[0,0,1]])
    return cv1



print(tf.convert_to_tensor([[1,1,1],[2,2,2]],dty))

cc = cv2.imread('F:/widerface/test/images/0--Parade/0_Parade_marchingband_1_9.jpg')
# cv2.imshow('cc3',moveleft(cc,100))
cv2.imshow('cc1',cc)
# 如果是为了修正图片(低通滤波)，算子中的值加起来为1，如果是为了找边缘（高通滤波），如果所有值均为0，则为原图

#修正部分
# 模糊
cv2.imshow('mosic',cv2.filter2D(cc,-1,kernel=np.array([
    [0.0625,0.125,0.0625],
    [0.125,0.25,0.125],
    [0.0625,0.125,0.125],
])))

# 锐化：强调在相邻的像素值的差异，使图片看起来更加的生动
cv2.imshow('more',cv2.filter2D(cc,-1,kernel=np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0],
])))


# 找特征部分
# sobel 横向 常用的，画轮廓用的
cv2.imshow('sobel row',cv2.filter2D(cc,-1,kernel=np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1],
])))

#sobel 纵向
cv2.imshow('sobel col',cv2.filter2D(cc,-1,kernel=np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],
])))

#大纲（outline）：一个轮廓内核（也称为“边缘”的内核）用于突出显示的像素值大的差异。具有接近相同强度的相邻像素旁边的像素在新图像中将显示为黑色，而强烈不同的相邻像素相邻的像素将显示为白色。
cv2.imshow('outline',cv2.filter2D(cc,-1,kernel=np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1],
])))



cv2.waitKey(0)
cv2.destroyAllWindows()





