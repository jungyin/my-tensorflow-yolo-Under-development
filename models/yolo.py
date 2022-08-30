
from calendar import c
import tensorflow as tf
import tensorflow.keras as keras
import logging
from pathlib import Path

from icecream import ic 

from tensorflow.python.framework.tensor_shape import TensorShape

import numpy as np
import math
from models.common import BottleneckCSP, Conv,StemBlock,C3,SPP,SPPF,Concat,Detect
from utils.general import make_divisible
from utils.tensorflow_utils import scale_img,stats_graph,fuse_conv_and_bn,model_info,initiaLize_weights
from tensorflow.keras.layers import BatchNormalization,UpSampling2D
from tensorflow.keras import layers 
from tensorflow.keras import Sequential
import time
from utils.autoanchor import check_anchor_order


logger = logging.getLogger(__name__)

class Model(keras.models.Model):
    def __init__(self,cfg='/models/yolov5s.yaml',ch=3,nc=None):
        super(Model,self).__init__()
        if isinstance(cfg,dict):
            yaml = cfg
        else :
            import yaml as y
            yaml_file=Path(cfg).name
            with open(cfg) as f:
                yaml = y.load(f,Loader=y.FullLoader)
        
        ch = yaml['ch'] = yaml.get('ch',ch)
        if(nc and nc!= yaml['nc']):
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (yaml['nc'],nc))
        
        self.model , self.save = parse_model(yaml,ch=[ch])
        
        self.names = [str(i) for i in range(yaml['nc'])] 
        
        m = self.model[-1] #获取detect()层
        if isinstance(m,Detect):
            s = 128
        
            call = self.call(tf.zeros([1,ch,s,s]))
        
            
            stridearray = [s / x.shape[-2] for x in call]
            m.stride=tf.convert_to_tensor(stridearray)
            
            m.anchors /= tf.reshape(m.stride,[-1,1,1])
            check_anchor_order(m)
            self.stride = m.stride 
            #tensorflow似乎不需要我来配置weights
            # self._initialize_biases()
        
        # 初始化权重和biases
        initiaLize_weights(self.model)
        self.info()
        logger.info('')


    def call(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1,0.83,0.67]
            f = [None,3,None]
            y = [] #outputs
            
            for si, fi in zip(s,f):
                xi = scale_img(x.filp(fi) if fi else x,si)
                yi = self.forward_once(xi)[0]
                
                yi[...,:4] /= si# 重新调整图片大小
                
                if(fi ==2):
                    # 图片的高裁剪
                    yi[...:, 1] = img_size[0]-yi[...,1]
                elif fi ==3 :
                    # 图片的宽裁剪
                    yi[...:, 0] = img_size[1]-yi[...,0]
                y.append(yi)
            return tf.concat(y,1)
        else :
            return self.forward_once(x,profile)
       
        
    # 首次初始化，这里改一下方法，改走build，不走call，不然会吧build给走了然后就寄了
    def forward_once(self,x,profile=False):
        y, dt=[],[]
        # 这个i用来判断当前的m是否需要被保存下来
        
        i=0
        for m in self.model.layers:
            i+=1
            if(m.f != -1):
                
                x = y[m.f] if isinstance(m.f,int) else [x if j==-1 else y[j] for j in m.f]
                if(x is None):
                    print('错误的x',end='')
                    print(m)
                
            #浮点能力计算有问题，先不算了
            # if profile:
                # with tf.Graph().as_default() as graph:
                    # t = time.time()
                    
                    # 行吧，这里是瞎跑10次，测浮点算力的
                    #？是吗，怎么感觉这里不完全像是测浮点算力的？
                    # for _ in range(10):
                        # _ = m(x)
                    # dt.append((time.time()-t)*100)
                    # o = stats_graph(graph)
                    # print('FLOPS:\t%0.1f\t%0.1f\t%0.1f\t\tms %s\t\t'%(o.flops,m.np,dt[-1],m.type))
            
            # print('展示model名',end=',')
            # print(m,end=',')
            # print(x)
            
            #似乎tensorflow无法使用不同的数组来反复调用同一个layers，会被维度检验卡住。。。不知道有没有好的解决办法
            # if(not isinstance(x,Detect)):
                # input_shape = x.shape
                
            x = m(x)
          
            # 1,64,32,32
            # 1,128,16,16
            y.append(x if m.i in self.save else None)
            
        return x
    def _initialize_biases(self,cf=None):
        # 初始化 detect中的biases ,cf是类出现的频率(class frequency)
        m = self.model.layers[-1]# 一般detect是在yaml中的最后一个
        for mi, s in zip(m.m.layers,m.stride):
            mi.get_weights()
            mi.bias = True
            print(mi.get_weights())
            b = np.reshape(mi.bias,(m.na,-1))
           
            n = float(s.numpy())
            
            b[:, 4] += math.log(8 / (640 / n ) **2)#？？为啥每640个图片才8个待识别对象
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99) if cf is None else np.log(cf/cf.sum))#？这里取对数干嘛？
            mi.bias = b #bias 是一个可以跑的layers?不太i清楚，torch上是Parameter,暂时不太理解这个东西。。
    
    # 输出biases中的信息    
    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = np.reshape(mi.bias.detach(),[-1])
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    #融合模型中的conv和bn
    def fuse(self):
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) in Conv and hasattr(m,'bn'):
                m.conv = fuse_conv_and_bn(m.conv,m.bn)
                delattr(m,'bn') #由于上面已经把bn融合进去了，这里就需要删除bn
                m.call = m.fuseCall
            
                
            
        self.info()
        return self
    
    def info(self,verbose = False ,img_size=640):
        ic('先不打印模型性能')
        # model_info(self,verbose,img_size)
        
                
        
def parse_model(d, ch):  # model_dict, input_channels(3)
    # 被跳过的str，读取yaml中有一些属性是string字符串格式，这些字符串需要被跳过
    jumpStr=['nearest','channels_first']
    
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors,nc,gd,gw = d['anchors'],d['nc'],d['depth_multiple'],d['width_multiple']
    na = (len(anchors[0])//2) if isinstance(anchors,list) else anchors
    no = na * (nc+5)
    
    itemlayers,save,c2=[],[],ch[-1]
    for i, (f,n,m,args) in enumerate(d['backbone'] + d['head']):
        if isinstance(m,str):
           
            m = eval (m)
            
        else:
            m=m
        
        for j,a in enumerate (args):
            
            args[j] = eval(a) if isinstance(a,str) and a not in jumpStr else a
        print('打印参数',end=' ')
        print(m,end=' ')
        print(args,end=' ')
        # 重新计算模型深度
        n = max(round(n*gd),1) if n>1 else n 
        print('模型深度',end='')
        print(n,end='')
        if m in [Conv,StemBlock,C3,SPP,SPPF]:
            
            c1,c2 = ch[f[0] if isinstance(f,list) else f],args[0]
            
            
            c2 = make_divisible(c2 * gw,8) if c2!= no else c2
            
            
            
            args = [c1,c2,*args[1:]]
            if m in [C3,BottleneckCSP]:
                args.insert(2,n)
                n = 1
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is BatchNormalization:
            args = [ch[f[0] if isinstance(f,list) else f]]

        elif m is Concat:
            c2 = sum ([ch[-1 if x == -1 else x +1] for x in f])
        elif m is Detect:
            args.append([ch[x+1] for x in f])
            if isinstance(args[1],int):
                args[1] = [list(range(args[1]*2))] * len(f)
            
        else :
            c2 = ch[f[0] if isinstance(f,list) else f]
        
        print('处理后的参数',end='')
        print(args)
        
        for _ in range(n):
            
            m_=m(*args)
            m_.i,m_.f=i,f
            itemlayers.append(m_)
            
           
            
        # m_= Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        # t = str(m)[8:-2].replace('__main__','')
        # 获取当前模型中所有的元素及计算它们的平均值（这个是干嘛用的？）
        #放弃计算np,tensorflow没有找到pytorch那样的parameters参数
        # np = sum([x.numel() for x in m_.count_params])
        # m_.i,m_.f,m_.type = i,f,t
        t="暂无"
        logger.info('%3s%18s%3s %-40s%-30s' % (i, f, n, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        itemlayers.append(m_)
        ch.append(c2)
        
    model = itemlayers
    # model = Sequential([
    #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(10)
    # ])

    # model.summary()
    # model.summary()
    # model.variables
    return model,sorted(save)

        
        