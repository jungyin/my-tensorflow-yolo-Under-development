
from cProfile import label
import io
from operator import getitem
from pathlib import Path
from random import random,uniform,randint,choices
from cv2 import BORDER_CONSTANT
import tensorflow as tf
import numpy as np
import glob
import os
import cv2
from itertools import repeat
import math
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from tqdm import tqdm
import logging
from PIL import Image, ExifTags
import utils.tfrecord_utils as trutil
from tensorflow.keras.utils import Sequence

from utils.general import xyxy2xywh, xyxy2xywh, clean_str


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break



logger = logging.getLogger(__name__)
# from utils.tensorflow_utils import 
tf.data.Dataset
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes



def img2label_paths(img_paths):
     # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]



def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    # with tensorflow_distributed_zero_first(rank):
    
 
    
    dataset = LoadFaceImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights
                                    )
    
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    

    return dataset
    
    
class LoadFaceImagesAndLabels(Sequence):
    # 这个方法只是将图片信息缓存到内存中的，图片的具体信息。。。也缓存吗？
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.batch_size = batch_size
        
        
        f=[] #图片文件数组
        
        
        for p in path if isinstance(path,list) else [path] :
            p = Path(p)
            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive = True)
            elif p.is_file:
                with open(p,'r') as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.paent) + os. sep
                    # 重新校对一下图片存储的路径
                    f += [x.replace('./',parent) if x.startswith('./') else x for x in t]
            # 既不是路径，也不是图片，那就报错    
            else :
                raise Exception('%s does not exist' % p)
            
        self.img_files = sorted([x.replace('/',os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
        assert self.img_files ,'No images found'
        
         
        
        # 检查标签缓存
        self.label_files = img2label_paths(self.img_files) #labels
        
        # cache_path = Path(self.label_files[0]).parent.with_suffix('.TFRecord')  # cached labels
         
        # if cache_path.is_file():
        #     with open(path,'w') as f:
        #         cache = yaml.dump(path,f,sort_keys=False)
                
        #     if(cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache):
        #         cache = self.cache_labels(cache_path)
        # else:
        # 你缓冲啥我都支持，但缓冲labels，有啥问题啊？反正都要往内存里读，每次运行也不差你这个统计用的时间
        # 似乎内容太多了，加个注释，cache是读取读取出来图片的一些其他统计性信息，img_files是图片们的路径，shapes是图片们的大小，labels是图片们上面有哪些位置有什么标签，至于n，就是有多少个图（有点不太对劲？）
        cache,self.img_files,self.shapes,self.labels, n = self.cache_labels() 
            
        [nf, nm, ne, nc, n] = cache.pop('results')
        
        
        
        desc = f"Scanning  for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        
        tqdm(None, desc=desc, total=n, initial=n)
      
        assert nf > 0 or not augment, f'No labels found {nf} . Can not train without labels.'
        
        self.label_files = img2label_paths(self.img_files)  # update
        
        # 如果只有一个类，那就把x的所有第二维数组变成0？
        if(single_cls):
            for x in self.labels:
                x[:, 0]=0
                
        bi = np.floor(np.arange(n) / batch_size).astype(np.int) # 计算所在batch的下标
        nb = bi[-1] + 1 #batch有多少个
        self.batch = bi # 在图片中，当前batch下标？感觉不太怼
        self.n = n # 当前图片总量
        
        self.nb = nb
        
        
        if self.rect:
            # 奇怪，这个rect是在干嘛？
            wh = self.shapes
            ar = wh[:,1]/wh[:,0] # aspect ratio ,宽高比
            # 内部的rect框
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = [self.shapes[i] for i in irect]
            ar = ar[irect]
            
            shapes = [[1,1]] * nb
            
            # 奇怪？这里为啥要nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi,1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
                    
            bs = np.array(shapes) * img_size / stride + pad  #先计算一次bs的大小，公式是图片数量 乘以图片的体积，再除以步进，然后加上padding 
            
            self.batch_shapes = np.ceil(bs).astype(np.int) * stride #将shapes向上取整，然后乘s(步进)活动新的一次batch_shapes
        
        
        # 在训练前，是否需要将图片缓存丢到内存中，警告，大量图片数据可能把电脑内存用光，造成oom问题
        self.imgs = [None] * n
        if cache_images:
            gb = 0 #当前缓存用了多少gb
            self.img_hw0, self.img_hw = [None] * n , [None] * n #准备一下图片的宽高数组，有几个图片准备几组
            results = ThreadPool(8).imap(lambda x : load_image(*x),zip(repeat(self),range(n))) # 准备线程池,多线程加载图片数据
            pbar=tqdm(enumerate(results),total=n)
            for i,x in pbar:
                self.imgs[i], self.img_hw0[i],self.img_hw[i] = x #将上面加载成功的图片数据（用load_image）写入数组中去
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)
      
        
    def __len__(self):
        #  返回batch_size的个数,number of batch
        # return self.nb
        # 但我似乎更需要的是图片的总数量而不是baatch_size的个数
        return self.n
    
    #pytorch似乎是返回指定条目，但tensorflow就是返回一个bs的标签和数据了，这里注意一下
    def __getitem__(self,index):
        
        
        hyp = self.hyp
        mosaic = self.mosaic and random() < hyp['mosaic']#是否会有随机马赛克，奇怪，tensorflow似乎本身就有这个玩意，随机生成马赛克
        if mosaic :
            img,labels = load_mosaic_face(self,index)
            shape = None
            if random() < hyp['mixup']:#mixup 日后有机会研究， https://arxiv.org/pdf/1710.09412.pdf
                img2,labels2 = load_mosaic_face(self,index)
                r = np.random.beta(8.0,8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
            
                labels = np.concatenate((labels,labels2),0)
        else:
            img, (h0, w0),(h, w) = load_image(self,index)
            
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size# 确认验证框的大小，没有就是图片大小
            
            img, ratio, pad = letterbox(img, shape ,auto=False, scaleup=self.augment)
        
            shapes = (h0,w0), ((h / h0, w/ w0), pad)# 用于图片缩放
            
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # 调整图片由xywh为xyxy
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
                
                # 为labels配种pad
                labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 5] + pad[0]) + (
                    np.array(x[:, 5] > 0, dtype=np.int32) - 1)
                labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 6] + pad[1]) + (
                    np.array(x[:, 6] > 0, dtype=np.int32) - 1)
                labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 7] + pad[0]) + (
                    np.array(x[:, 7] > 0, dtype=np.int32) - 1)
                labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 8] + pad[1]) + (
                    np.array(x[:, 8] > 0, dtype=np.int32) - 1)
                labels[:, 9] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 9] + pad[0]) + (
                    np.array(x[:, 9] > 0, dtype=np.int32) - 1)
                labels[:, 10] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 10] + pad[1]) + (
                    np.array(x[:, 10] > 0, dtype=np.int32) - 1)
                labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 11] + pad[0]) + (
                    np.array(x[:, 11] > 0, dtype=np.int32) - 1)
                labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 12] + pad[1]) + (
                    np.array(x[:, 12] > 0, dtype=np.int32) - 1)
                labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 13] + pad[0]) + (
                    np.array(x[:, 13] > 0, dtype=np.int32) - 1)
                labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 14] + pad[1]) + (
                    np.array(x[:, 14] > 0, dtype=np.int32) - 1)
            
        if(self.augment):
            #如果没作过mosaic操作，那图片会没走图片的一些泛化处理，这里补上
            if not mosaic:
                img, labels = random_perspective(img, labels,
                            degrees=self.hyp['degrees'],
                            translate=self.hyp['translate'],
                            scale=self.hyp['scale'],
                            shear=self.hyp['shear'],
                            perspective=self.hyp['perspective'])  # border to remove
            
            augment_hsv(img,hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'])
            
        nL = len(labels)
        
        if(nL):
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # 把高的所有值都归一化为0-1
            labels[:, [1, 3]] /= img.shape[1]  # 把宽的所有值都归一化为0-1

            labels[:, [5, 7, 9, 11, 13]] /= img.shape[1]  # normalized landmark x 0-1
            labels[:, [5, 7, 9, 11, 13]] = np.where(labels[:, [5, 7, 9, 11, 13]] < 0, -1, labels[:, [5, 7, 9, 11, 13]])
            labels[:, [6, 8, 10, 12, 14]] /= img.shape[0]  # normalized landmark y 0-1
            labels[:, [6, 8, 10, 12, 14]] = np.where(labels[:, [6, 8, 10, 12, 14]] < 0, -1, labels[:, [6, 8, 10, 12, 14]])

        #图片资源繁化
        if(self.augment):
            # 图片上下翻转
            if random() < hyp['flipud']:
                img = np.flipud(img)
                # 如果打过标签，记得把标签内容物也做翻转
                if(nL):
                    labels[:, 2]  = 1 - labels[:, 2]
                    
                    labels[:, 6] = np.where(labels[:,6] < 0, -1, 1 - labels[:, 6])
                    labels[:, 8] = np.where(labels[:, 8] < 0, -1, 1 - labels[:, 8])
                    labels[:, 10] = np.where(labels[:, 10] < 0, -1, 1 - labels[:, 10])
                    labels[:, 12] = np.where(labels[:, 12] < 0, -1, 1 - labels[:, 12])
                    labels[:, 14] = np.where(labels[:, 14] < 0, -1, 1 - labels[:, 14])
            
            if random() < hyp['fliplr']:
                img = np.flipud(img)
                # 如果打过标签，记得把标签内容物也做翻转
                if(nL):
                    labels[:, 1]  = 1 - labels[:, 1]
                    
                    labels[:, 5] = np.where(labels[:,5] < 0, -1, 1 - labels[:, 5])
                    labels[:, 7] = np.where(labels[:, 7] < 0, -1, 1 - labels[:, 7])
                    labels[:, 9] = np.where(labels[:, 9] < 0, -1, 1 - labels[:, 9])
                    labels[:, 11] = np.where(labels[:, 11] < 0, -1, 1 - labels[:, 11])
                    labels[:, 13] = np.where(labels[:, 13] < 0, -1, 1 - labels[:, 13])

                    #左右镜像的时候，左眼、右眼，　左嘴角、右嘴角无法区分, 应该交换位置，便于网络学习

                    eye_left = np.copy(labels[:,[5, 6]])
                    mouth_left = np.copy(labels[:,[11, 12]])
                    labels[:,[5,6]] = labels[:,[7, 8]]
                    labels[:,[7,8]] = eye_left
                    labels[:,[11,12]] = labels[:,[13,14]] 
                    labels[:,[13,14]] = mouth_left
        labels_out = np.zeros((nL,16),dtype=np.float32)

        if nL:
            labels_out[:,1:] = labels
    
        #Convert
        img = tf.transpose(img[:,: , ::-1],(2,0,1))
        
        img = np.ascontiguousarray(img)
        
        return img, labels_out , self.img_files[index]
    
    def collate_fn(self,imgs,labels,paths):
        ll = labels[0]
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
            
        
        
        labels = tf.concat(labels, 0)
        imgs = np.stack(imgs, 0)
        return imgs , labels, paths
    
    
    # 这个b方法想了半天觉得没用就是没用，tnnd存啥不好要存路径？
    def cache_labels(self):
        imgFature={}
        # 按理来说应该没事，毕竟只是存路径，应该不会用太大空间吧？一个labels应该塞得下
        # 这里nm,nf,ne,nc改改规则？
        nm, nf, ne, nc =0, 0, 0, 0
        pbar = tqdm(zip(self.img_files, self.label_files))
        
        
        filepaths = []
        shapes = []
        labels = []
        for i, (im_file, lb_file) in enumerate(pbar):
            
            try:
            # if(True):
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                im.close()
                
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                       
                    if len(l):
                        assert l.shape[1] == 15, 'labels require 15 columns each'
                        assert (l >= -1).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 15), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 15), dtype=np.float32)
                filepaths.append(im_file)
                shapes.append([shape[0],shape[1]])
                
                labels.append(l)
            except Exception as e:
                nc += 1
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"Scanning  for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'WARNING: No labels found in . ')

        # 将图片的关键信息做成tfrecord
        
        imgFature['hash'] = get_hash(self.label_files + self.img_files)
        imgFature['results'] = [nf, nm, ne, nc, i ]
        
        # x = tf.train.Example(tf.train.Features(imgFature))
        #似乎先不保存图片的边框啥的，后面可以考虑改改？
        #保存tfrecodes        
        
        # writer.write(x.SerializeToString())
        # writer.close()
        
        logging.info(f"load image message surcess")
        
        
        
        return imgFature,filepaths,np.array(shapes, dtype=np.float32),labels ,len(shapes)
    def decode_fn(record_byte):
        return tf.io.parse_single_example(record_byte)
    
    
# Ancillary functions --------------------------------------------------------------------------------------------------
    # 图片加载
def load_image(self, index):
    #加载来自dataset中的图片，返回图片，图片原始宽高，重新配置后新的宽高
    img = self.imgs[index]
    if img is None:#如果图片缓存不存在，就需要亲自处理一下
        path = self.img_files[index]
        img = cv2.imread(path) #BGR
        assert img is not None ,'Image Not Found '+path
        h0, w0 = img.shape[:2] #原始宽高
        r = self.img_size / max(h0, w0) #用opencv重新配置图片大小成同正方形图片
        if r != 1: #防止图片被过度压缩，这里用if配置一下，只要r没到1这么小，那就resize图像
            # 选择图片裁剪方法
            interp = cv2.INTER_AREA if r <1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img,(int(w0 * r),int(h0 * r)),interpolation = interp)
        
        return img,(h0,w0),img.shape[:2] # 图片本身，图片的原始宽高，图片的缩放后宽高
    #如果图片缓存存在，直接返回就行了
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
        
#加载图片，但有马赛克            
def load_mosaic_face(self,index):
    
    labels4 = [] 

    s = self.img_size
    #这个地方是-x完全是因为最开始给的就是负数，别想太多了，浪费时间太多了在这里
    myc,mxc = [int(uniform(-x, 2 * s + x)) for x in self.mosaic_border] #随机取马赛克的输量
    
    myc,mxc = [174, 292]
    
    indices = [index] + [ randint(0, self.n - 1 )  for _ in range(3)]  #3个随机的图像附加索引
    indices = [36,1744,913,2656]
    # 我tm一个飞刀，找了半天是这里吧index给改了
    
    for i, index in enumerate(indices):
        #加载图片
        img, _, (h,w) = load_image(self,index)
       
        
        
        #计算新图像的左上，右上，左下，右下位置
        if i == 0 :# 左上
            # ？这玩意在干嘛？随机抽图像？
            # 我的娘类，这玩意是不是随机抽4个图片，然后把四个边砍下来拼凑一个新的图？那这图还能用吗？
        
            img4 = np.full((s*2,s*2,img.shape[2]), 114, np.uint8)#准备一下马赛克处理后的图片数组,奇怪，为啥要把图片的宽高乘2？

            x1a,y1a,x2a,y2a = max (mxc - w, 0), max(myc-h,0),mxc,myc 
            x1b,y1b,x2b,y2b = w - (x2a-x1a), h - (y2a - y1a),w,h 
        elif i==1: # 右上
            x1a,y1a,x2a,y2a = mxc, max(myc-h,0),min(mxc + w, s * 2), myc 
            x1b,y1b,x2b,y2b = 0, h - (y2a - y1a),min(w, x2a - x1a),h 
        elif i==2: # 左下
            x1a,y1a,x2a,y2a = max(mxc - w , 0), myc , mxc , min(s * 2, myc + h)
            x1b,y1b,x2b,y2b = w - (x2a - x1a), 0 , w , min(y2a - y1a, h)
        elif i==3: # 右下
            x1a,y1a,x2a,y2a = mxc, myc,min(mxc + w , s * 2),min(s * 2, myc + h) 
            x1b,y1b,x2b,y2b = 0, 0,min(w,x2a - x1a),min(h,y2a-y1a) 
    
        
    
        img4[y1a:y2a,x1a:x2a] = img[y1b:y2b,x1b:x2b]

        padw = x1a - x1b
        padh = y1a - y1b    
        # Labels
        x = self.labels[index]
        labels = x.copy()
        if len(x)>0: #标准化xywh到像素xyxy格式
            # box ,x1, y1, x2, y2
            labels[:,1] = w * (x[:, 1] - x[:, 3] / 2) +padw
            labels[:,2] = h * (x[:, 2] - x[:, 4] / 2) +padh
            labels[:,3] = w * (x[:, 1] + x[:, 3] / 2) +padw
            labels[:,4] = h * (x[:, 2] + x[:, 4] / 2) +padh
            # 10 landmarks
            #?这个大于0是干嘛的?也没个判断用函数
            labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (w * x[:, 5] + padw) + (np.array(x[:, 5] > 0, dtype=np.int32) - 1)
            labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (h * x[:, 6] + padh) + (np.array(x[:, 6] > 0, dtype=np.int32) - 1)
            labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (w * x[:, 7] + padw) + (np.array(x[:, 7] > 0, dtype=np.int32) - 1)
            labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (h * x[:, 8] + padh) + (np.array(x[:, 8] > 0, dtype=np.int32) - 1)
            labels[:, 9] = np.array(x[:, 9] > 0, dtype=np.int32) * (w * x[:, 9] + padw) + (np.array(x[:, 9] > 0, dtype=np.int32) - 1)
            labels[:, 10] = np.array(x[:, 10] > 0, dtype=np.int32) * (h * x[:, 10] + padh) + (np.array(x[:, 10] > 0, dtype=np.int32) - 1)
            labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (w * x[:, 11] + padw) + (np.array(x[:, 11] > 0, dtype=np.int32) - 1)
            labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (h * x[:, 12] + padh) + (np.array(x[:, 12] > 0, dtype=np.int32) - 1)
            labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (w * x[:, 13] + padw) + (np.array(x[:, 13] > 0, dtype=np.int32) - 1)
            labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (h * x[:, 14] + padh) + (np.array(x[:, 14] > 0, dtype=np.int32) - 1)
        labels4.append(labels)
    
    # concat/修剪标签
    if len(labels4):
        labels4 = np.concatenate(labels4,0)
        np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])  #与随机透视一起使用
        # img4, labels4 = replicate(img4,labels4)
        
        #这个5号位是干嘛的来着？我记得0到3是图像的起点和中央点，5是什么来着？
        labels4[:, 5:] = np.where(labels4[:,5:] < 0, -1, labels4[:, 5:])
        labels4[:, 5:] = np.where(labels4[:,5:] > 2 * s, -1, labels4[:, 5:])
        
        # 他们在干嘛？都没意义啊
        labels4[:, 5] = np.where(labels4[:, 6] == -1, -1, labels4[:, 5])
        labels4[:, 6] = np.where(labels4[:, 5] == -1, -1, labels4[:, 6])

        labels4[:, 7] = np.where(labels4[:, 8] == -1, -1, labels4[:, 7])
        labels4[:, 8] = np.where(labels4[:, 7] == -1, -1, labels4[:, 8])

        labels4[:, 9] = np.where(labels4[:, 10] == -1, -1, labels4[:, 9])
        labels4[:, 10] = np.where(labels4[:, 9] == -1, -1, labels4[:, 10])

        labels4[:, 11] = np.where(labels4[:, 12] == -1, -1, labels4[:, 11])
        labels4[:, 12] = np.where(labels4[:, 11] == -1, -1, labels4[:, 12])

        labels4[:, 13] = np.where(labels4[:, 14] == -1, -1, labels4[:, 13])
        labels4[:, 14] = np.where(labels4[:, 13] == -1, -1, labels4[:, 14])
        
        
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    
    return img4,labels4

# 图片的宽高转置，感觉这个方法写复杂了
def replicate(image,labels):
    h, w, c = image.shape()#读取图片的宽高和channel
    boxes = labels[:,1:].astype(int) #读取标签的数量？
    
    #python 中.t操作为转置
    # a
    # array([[1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]])
    # a.T
    # array([[1, 4, 7],
    #     [2, 5, 8],
    #     [3, 6, 9]])
    x1, y1, x2, y2 = boxes.T #不知道这个转置要干嘛
    s = ((x2 - x1) + (y2 - y1)) / 2 #side length(像素)
    for i in s.argsort()[:round(s.size * 0.5)]: # 最小索引
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = randint(0,h - bh),  randint (0, w - bw)
        x1a, y1a, x2a, y2a = [xc,yc,xc+bw,yc+bh]
        image[y1a:y2a,x1a:x2a] = image[y1b:y2b,x1b:x2b]
        labels = np.append(labels,[[labels[i,0], x1a, y1a, x2a, y2a]],axis=0)
    
    return image,labels



#对图片作一些处理，tensorflow 的image库本身就可以，但这里重写一下我也不介意
def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
  

    # 首先，读取图片宽高，并且将宽高加上border（边界）
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # 获取图片的中心点
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    # 透视图？不太像
    P = np.eye(3)
    P[2, 0] = uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    # 图片旋转和图片缩放
    R = np.eye(3)
    #图片旋转角度
    a = uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    #图片随机裁剪？啥玩意啊
    # Shear
    S = np.eye(3)
    # 计算正切值？奇怪，为啥要裁剪圆形出来？
    S[0, 1] = math.tan(uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    # 图片翻转
    T = np.eye(3)
    T[0, 2] = uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 提前预定了的函数
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        #xy = np.ones((n * 4, 3))
        xy = np.ones((n * 9, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].reshape(n * 9, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            #图片缩放
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 18)  # rescale
        else:  # affine
            # 图片仿射？
            xy = xy[:, :2].reshape(n, 18)

        # 这个box应该是为了吧处理完成的图像的labels继续放进去准备的吧?
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        landmarks = xy[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
        mask = np.array(targets[:, 5:] > 0, dtype=np.int32)
        landmarks = landmarks * mask
        landmarks = landmarks + mask - 1

        landmarks = np.where(landmarks < 0, -1, landmarks)
        landmarks[:, [0, 2, 4, 6, 8]] = np.where(landmarks[:, [0, 2, 4, 6, 8]] > width, -1, landmarks[:, [0, 2, 4, 6, 8]])
        landmarks[:, [1, 3, 5, 7, 9]] = np.where(landmarks[:, [1, 3, 5, 7, 9]] > height, -1,landmarks[:, [1, 3, 5, 7, 9]])

        
        landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
        landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

        landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
        landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

        landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
        landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

        landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
        landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])

        landmarks[:, 8] = np.where(landmarks[:, 9] == -1, -1, landmarks[:, 8])
        landmarks[:, 9] = np.where(landmarks[:, 8] == -1, -1, landmarks[:, 9])

        targets[:,5:] = landmarks

        # if(x.min(1) == 0 and x.max(1) == 0 and y.min(1)==0 and y.max(1) == 0):
        #     xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # else:
        
        xmin = x.min(1)    
        ymin = y.min(1)    
        xmax = x.max(1)    
        ymax = y.max(1)    
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        # 吧target中等宽高的标签拿出来，这些用不了
        box1=targets[:, 1:5].T * s
        box2=xy.T
        i = box_candidates(box1, box2)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
        
    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # box1是图像增强前的box,box2是增强后的box,
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1] #box宽高获取
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # 计算他们的变化比例
    #好了,计算新的边框内容并返回
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

#将图片切割为32像素宽高的图片，很重要！
def letterbox(img,new_shape=(640,640),color=(114,114,114),auto = True,scaleFill=False, scaleup = True):
    shape = img.shape[:2]#读取当前图片宽高
    if isinstance(new_shape,int):
        new_shape = (new_shape,new_shape)
        
    # 计算图片缩放比
    r = min(new_shape[0] / shape[0],new_shape[1]/shape[1])

    #如果图片不允许变大，那就要规定一下，缩放值不能大于1
    if not scaleup:
        r = min (r,1.0)
    
    #计算图片padding
    ratio = r,r #计算图片缩放比例
    # 吧图片内容给扩张一下
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] #计算图片的边界
    if auto: 
        # 自动化吧图片的宽高给处理为64的整除数
        # 那也不对啊，这是吧图片里的数据变成64的整除输，没吧宽高处理了啊？再看看？
        dw, dh = np.mod(dw,64), np.mod(dh,64) 
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1],new_shape[0])
        ratio = new_shape[1] / shape[1] , new_shape[0] / shape[0] #图片缩放比
    
    dw /= 2 # 将填充物填到上下左右边外
    dh /= 2
    
    if shape[::-1] != new_unpad: # 如果图片宽高不对，就要重新配图片大小    
        img = cv2.resize(img,new_unpad,interpolation=cv2.INTER_LINEAR)# 俩个像像素间的平均值？好像是
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT, value= color) # 为图片添加边框
    
    return img, ratio, (dw, dh)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    