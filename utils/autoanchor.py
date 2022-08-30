import numpy as np
import tensorflow as tf
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm
from utils.general import colorstr


def check_anchor_order(m):
    # 检查YOLOv5 Detect（）模块m的锚点顺序和步幅顺序，必要时进行纠正
    a = tf.reshape(np.prod(m.anchors_grid,axis=-1),[-1])
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if np.sign(da) != np.sign(ds):
        print('Reversing anchor order')
        m.anchors[:] = np.flip(m.anchors,0)
        m.anchor_grid[:] = np.flip(m.anchor_grid,0) 

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors...', end='')
    m = model.layers[-1] if hasattr(model,'model') else model.layers[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims = True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    
    