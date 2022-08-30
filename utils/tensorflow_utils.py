
from contextlib import contextmanager
from copy import deepcopy
import glob
import logging
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

from icecream import ic

import tensorflow.keras.layers

import cv2
import hashlib
import numpy as np
import yaml
import shutil
import tempfile
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
import tensorflow as tf
from tqdm.auto import tqdm

from tensorflow.keras import layers

#该修饰器是在多线程训练时过程中如果主线程有事情，就要所有子线程暂停的方法
#这个方法在pytorch中是为了等数据加载完成接着走，tensorflow后面看看有没有必要使用这个东西
# @contextmanager
# def tensorflow_distributed_zero_first(local_rank:int):
#     # tensorflow无法找到可替代语句
#     if(local_rank not in [-1,0]):
#         tf.distribute
#     yield
#     if local_rank==0:
#         tf.distribute.

#初始化tensorflow随机数种子
def init_tensorflow_seeds (seed):
    tf.random.set_seed(seed)



def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
            
            
def scale_img(img, ratio=1.0, same_shape=False, gs=32): #img(16,3,256,416)
    # scales img(bs,,y,x) by
    if ratio == 1.0:
        return img
    else :
        h,w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))
        img = tf.image.resize(img, s ,  tf.image.ResizeMethod.BILINEAR,False)
        if not same_shape:
            h, w = [math.ceil((x * ratio / gs) * gs for x in (h,w))]
            
        return tf.pad(img,[0, w - s[1] , h - s[0]] , constant_values= 0.447)
        
        
# 计算浮点能力
def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    return flops,params


def intersect_dicts(da,db,exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

# 将传入的bn和conv融合
def fuse_conv_and_bn(conv,bn):
    fusedconv = layers.Conv2D(conv.out_channels,conv.kernel_size,conv.stride,padding='same',groups=conv.groups)
    
    
    w_conv = np.reshape(conv.weight.clone(),[conv.out_channels,-1])
    
    # 获取新的bn?
    # 先计算bn.eps+bn.running_var的平方根
    # 然后用bn.weight除以这个平方根
    #最后取它们的对角线
    w_bn = tf.linalg.diag_part(bn.weight.div(tf.math.sqrt(bn.eps + bn.running_var)))
    fusedconv.weights.copy_(layers)
    
    b_conv = tf.zeros(conv.weight.size(0)) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(tf.math.sqrt(bn.running_var + bn.eps))
    fusedconv.bias=np.reshape(tf.convert_to_tensor(w_bn,np.reshape(b_conv,[-1,1])),[-1])+b_bn

    return fusedconv    

def initiaLize_weights(model):
    for m in model:
        t = type(m)
        if t is layers.Conv2D:
            pass
        elif t is layers.BatchNormalization:
            m.epsilon=1e-3
            m.momentum = 0.03
        # tensorflow 没找到这个类似的操作
        # inplace，当需要处理值(比如数组中出现负数)就会直接把内存中的数组变成0，节约内存
        # elif t in [layers.ReLU,layers.LeakyReLU]:
            # m.inplace=True
           
            
    
 # 打印model的性能，可以先不用打印
def model_info(model,verbose=False,img_size=640):
    ic('先不打印模型性能')
    # # 模型信息，img_size可以是int或者是数组
    # n_p = sum(x.num for x in model.par)# 模型的参数
    # n_g = sum(x.num for x in model.par)# 模型的梯度
    
    
    # if(verbose) :
    #     print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    #     for i, (name, p) in enumerate(model.)
    
    
# def is_parallel(model):
#     return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model)  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        # tensorflow 不应该这么固定
        # for p in self.ema.parameters():
        #     p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        
        self.updates += 1
        d = self.decay(self.updates)

        msd =  model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

    