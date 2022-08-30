from pathlib import Path
import numpy as np 
import random
import os
import math
import tensorflow as tf
import re
import glob
import yaml 
import logging
from utils.metrics import fitness
from utils.tensorflow_utils import init_tensorflow_seeds,download_url_to_file

from utils.google_utils import gsutil_getsize

# 随机数种子生成器
def init_seeds(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    init_tensorflow_seeds(seed)
    
    

def check_img_size(img_size , s=32):
    # 获取可以被s整除的新大小
    new_size = make_divisible(img_size,int(s))    
    if new_size!=img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload

    
    
# 如果没找到数据，就去下载数据
#这个，将来换tensorflow的 tensorflow-dataset包下载
def check_dataset(dict):
    val,s = dict.get('val'),dict.get('download')
    if val and len(val):
        # print('\nWARNING: Dataset not found, nonexistent paths: %s' % str(val))
        
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists()  for x in val):
            if s and len(s):
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name()
                    download_url_to_file(s,f)
                    #解压，并且把压缩包删除
                    r = os.system('unzip -q %s -d ../ && rm %s')
                else: 
                    r= os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
                
            else :
                raise Exception('Dataset not found')
        
#返回可以被整除的x
def make_divisible(x,divisor):
    return math.ceil(x/divisor) * divisor



#xywh的图片转为xyxy的图
# 注意，转化后的xy为xs和ys，不是xc和yc
def xyxy2xywh(x):
    y = np.copy(x)
    y[:,0] = (x[:,3] + x[:,1]) / 2 # x center
    y[:,1] = (x[:,3] + x[:,1]) / 2 # y center
    y[:,2] = x[:,2] - x[:,0]  # w
    y[:,3] = x[:,3] - x[:,1]  # h 
    return y

#Xywh的图片转为xyxy的图
def xywh2xyxy(x):
    y = np.copy(x)
    y[:,0] =(x[:,0] - x[:,2] // 2)  # top left x
    y[:,1] =(x[:,1] - x[:,3] // 2)  # top left y
    y[:,0] =(x[:,0] + x[:,2] // 2)  # bottom right x
    y[:,1] =(x[:,1] + x[:,3] // 2)  # bottom right y
    return y

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)



def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def increment_path(path, exist_ok=True, sep=''):
    # 如果文件已存在，自动重命名新起一个文件
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    # 从训练的标签中获取权重
    if labels[0] is None:  # no labels loaded
        return np.zeros(0)

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights


#GIou,DIou和CIoU的解释：https://zhuanlan.zhihu.com/p/416550548

#Iou是一个评估俩个矩形框之间距离的指标，这个指标具有distance的一切特性，包括对称性，非负性，同一性，三角不等性
#注意：Iou损失最大的问题是俩个框没有重叠时，Iou会变成0,但giou还是会有数值的，(虽然是负数)
#IoU值是在计算俩个框之间重叠率有多高

# GIoU是吧俩个box丢入一个大box，然后计算大box中阴影部分的面积，来计算俩个box的重叠效率有多高
# DIoU是计算俩个box的中心距离，距离越近效率越高
# CIoU

#(Intersection over Union） iou全名，giou为Ground-Truth，diou 为Distance IoU，ciou 为Complete IoU


#eps是为了防止分母为0而存在的值
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, EIoU = False,MCIoU = False, eps=1e-9):
    # 返回 box2在box1中的iou值，其中 box1为4，box2为nx4（什么意思？）
    
    if x1y1x2y2: # boxes的组合为x1y1x2y2时走这里
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy #如果不是x1y1x2y2组合，就变成是xywh组合
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    
    # Intersection area
    # 获取box1和box2的交叉区域,的面积
    inter = (tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1)) * \
            (tf.minimum(b1_y1, b2_y1) - tf.maximum(b1_y1, b2_y1))
    
    #重新获取box1和box2的宽高
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 
    unio = (w1 * h1) + (w2 * h2) #计算box1和box2的面积之和，奇怪，要干嘛
    
    iou = inter / unio # 计算交叉区域和重叠区域相除，得到的值必定为0到1之间，这样就完成了iou的计算（然而还是不懂iou要用来干嘛）
    
    if GIoU or DIoU or CIoU or EIoU:
        # convex (smallest enclosing box) width 
        # 计算最外面大box的宽高
        c_w = tf.maximum(b1_x2,b2_x2) - tf.minimum(b1_x1,b2_x1)
        c_h = tf.maximum(b1_y2,b2_y2) - tf.minimum(b1_y1,b2_y1)
        
        # 考虑替换为EIOU,eiou教程https://blog.csdn.net/q1552211/article/details/124590458
        if CIoU or DIoU or EIoU: # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1 # 看起来Ciou和dIou似乎是近亲啊？
            c2 = c_w ** 2 + c_h ** 2 + eps # c2 是全名是啥？,这个东西应该计算最大box的斜边，但为啥叫c2？ 他的值是最外面最大box的平方加高的平方，为了尽可能让俩个框重合，最后加个eps防止为0 
            rho2 = (((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 +
                    ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 ) /4 #这里计算出俩个box生成的大box中阴影的占比
            

            #diou，ciou和eiou，必定需要用导rho2/c2，然后用iou减去它

            if DIoU : 
                #DIoU是预测框与真实框中心点的欧式距离
                return iou - (rho2 / c2) #rho2 除以c2应该是为了归一化，但还是不懂为啥要算一下平方，直接裸上不更好吗？
            
            
            elif CIoU : # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                #CIoU是在DIoU的基础上增加了检测框尺度的loss，增加了长和宽的loss
                # 看ciou的图，似乎除了diou的计算中心点距离，还额外计算了俩个box之间最远点的距离
                # 用box2的宽高比正反切 减去 box1的宽高比正反切， 然后
               # ciou的部分看看第七周的课
               
                # CIou应该是用atan特殊处理了宽高比了，淦
                atan = tf.atan(w2 / h2) - tf.atan(w1 / h1) 
                v = (4 / math.pi ** 2) * \
                    (atan ** 2)#这样写也可以  v是用来衡量anchor和目标框一致度的值，奇怪，那为啥要做平方化？为啥要和一个奇怪处理后的pi乘？ #计算俩个box的长款比相似性
                    # tf.pow(atan, np.full_like(atan,2))#?这里。是吧宽高比给平方化了？
                alpha = v / ((1 + eps) - iou + v) # ？奇怪，alpha是什么？用1减去现在算出来的iou,然后加上v？这是干嘛？
                return iou - ((rho2 / c2) + (v * alpha))  # CIoU 

            elif MCIoU: #这个不用百度，是我自己根据EIoU想的CIoU,因为感觉CIoU还是奇奇怪怪的，所以这里用我自己的思路再弄一次

                # 淦，怎么完全感觉就是去pi的版本？
                atan = ((w2 / h2) - (w1 / h1)) ** 2
                                
                alpha = v / ((1 + eps) - iou + atan)

                return iou - ((rho2 / c2) + (v * alpha))
            
            elif EIoU:
                #计算俩个box之间的宽高的欧式距离
                rho_w2  =  (w2 - w1) ** 2 # 算出俩个box的宽度的差值的平方，必定为正数
                rho_h2  =  (h2 - h1) ** 2 # 算出俩个box的高度的差值的平方，必定为正数
                
                cw2 = c_w ** 2 #计算出图片的宽度的平方，这是要干嘛？
                ch2 = c_h ** 2 #计算出图片的高度的平方，不懂为啥这么喜欢平方？
                
                
                return iou - ((rho2 / c2) + (rho_w2 / cw2) + (rho_h2 / ch2))
                
    
        elif GIoU:# GIoU https://arxiv.org/pdf/1902.09630.pdf
            # giou的计算，先用大box的面积减去俩小box的面积,除以大box的面积，再被iou减去,就得到了giou
            # 换句话说，就是要用iou减去大box中俩个box之外空位所占比例
            c_area = c_h * c_w
            return iou - ((c_area - unio) / c_area)
       
        
    else: 
        return iou #IoU
                    
            
        
    
    
    