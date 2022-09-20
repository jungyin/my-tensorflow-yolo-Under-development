
from turtle import forward
from typing import Optional, Tuple, Union
from numpy.lib.function_base import place
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2
from tensorflow import nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
import warnings
import numpy as np
from tensorflow.python.keras.backend import dtype
from torch import channel_shuffle, conv2d, float32
from tensorflow.python.keras import initializers
from tensorflow.python.ops import gen_math_ops

conv2d_data_format='channels_last'
# conv2d_data_format='channels_first'

channels_index = (1 if conv2d_data_format == 'channels_first' else 3)


initializer = tf.random_normal_initializer(stddev=0.01)
def autopadding(k,p=None): # kernel ,padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k,int) else[x // 2 for x in k]
    return p


# 测试用的Denselayer
class MyDenseLayer(layers.Layer):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                    shape=[int(input_shape[-1]), self.num_outputs])

    def __call__(self, input):
        self.build(input.shape) ## build is called here when input shape is known
        return tf.matmul(input, self.kernel)

# 最基础的conv
class MyConv(layers.Layer):
    def __init__(self, ci,filters, k=1, s=1, p=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(MyConv, self).__init__()
        # l2正则
        # 正则
        l2 = L2(4e-5)
        self.filters=filters

        if(p is None):
            p=1

        # 标准2d卷积
        # 卷积作用：将卷积范围内的值相加,1*1范围的卷积多用于增维度与减维，相当于把通道上的值都给加起来，然后需要几层新的通道，就加载几次，例如一次1*1得到的是(x,y,1),俩次得到的是(x,y,2)以此类推
        self.conv = layers.Conv2D(self.filters, k, s, use_bias=False,
                      kernel_initializer=initializer,groups=g, kernel_regularizer=l2,padding='valid' if p==0 else 'same',data_format=conv2d_data_format)
        # 配置批标准化，这里我已经彻底晕了，网上说这个是啥的都有
        self.bn = layers.BatchNormalization(momentum=0.03)
        
        self.k=k
        # 配置激活函数，这里使用silu
        # silu:为x⋅Sigmoid(x)，swish为x⋅Sigmoid(βx)
        self.act = layers.Activation(nn.silu) if act is True else (act if isinstance(act,layers.Activation) else None)
        
        self.kernel_initializers=initializers.get("glorot_normal")
        self.bn = layers.BatchNormalization(momentum=0.03)
        
        
    def build(self,input_shape):
        
        super(MyConv, self).build(input_shape)
        
        # self.kernel  = self.add_weight(
        #     name='kernel ',
        #     shape=output_shape,
        #     initializer= self.kernel_initializers
        # )
        # bias先不考虑用
        # self.bias = self.add_weight(
        #     name='bias',
        #     shape=[self.filters],
        #     initializer="zeros"
        # )
        
        
        
    def call(self,inputs):
        # 一个conv，需要将输入的参数进行bn，conv，如果有需要的话，还要走一次silu操作
        
        
        
        # 这里我也不太懂当时为啥用or，直接==None不行吗?
        # if(self.act==None or self.act == None):
        if(self.act == None):
            outputs = self.bn(self.conv(inputs))
        else :
            outputs = self.act(self.bn(self.conv(inputs))) 
          
        # outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)# 矩阵相乘
    
        return outputs
    
    def fuseCall(self,inputs):
        return self.act(self.conv(inputs))
    
# stem block 结构，将输入图像尺寸缩为原先的4分之一，用于轻量化网络
class StemBlock(layers.Layer):
    def __init__(self, c1,filters, k=3, s=2, p='same', g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(StemBlock, self).__init__()
        self.stem_1=MyConv(c1 , filters , k , s , p , g , act)
        self.stem_2a=MyConv(filters , filters//2 , 1 , 1 , 'valid')
        self.stem_2b=MyConv(filters//2 , filters , 3 , 2 , 1)
        # 注意！torch版本这里是有一个取整方向，我这里没找到tensorflow对应的！
        self.stem_2p=layers.MaxPool2D((2,2),2,data_format=conv2d_data_format)
        self.stem_3=MyConv(filters*2,filters,1 , 1,'valid')
        
    def build(self,input_shape):
        super(StemBlock, self).build(input_shape)
        
    def call(self,x):
        stem_1_out  = self.stem_1(x)
        # stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(self.stem_2a(stem_1_out))
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(tf.concat((stem_2b_out,stem_2p_out),channels_index))
      
        return out


# 沙漏型结构，减少通道，降低计算量，通过1*1的conv降维计算后再升维返回
class Bottleneck(layers.Layer):
    def __init__(self, ci, co, shortcut=True, g=1, e =0.5):   # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(co * e) #hidden channels
        
       
        self.cv1= MyConv(ci,c_,1,1)
        self.cv2= MyConv(c_,co,3,1,g=g)
        self.add = shortcut and ci == co
        
    def build(self,input_shape):
        super(Bottleneck, self).build(input_shape)
        
    def call(self,x):

     
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# 带有csp的沙漏型结构，具体是将输入的数据源拆成俩组，一组走沙漏，一组直接conv，完成后再将俩个通过concat拼接到一起
class BottleneckCSP(layers.Layer):
    
    def __init__(self,ci,co,n=1,shortcut=True,g=1,e=0.5) :  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckCSP,self).__init__()
        c_ = int(co * e) # 隐藏层的深度
        self.cv1 = MyConv(c_,1,1)
        self.cv2 = layers.Conv2D(c_,(1,1),(1,1),use_bias=False,data_format=conv2d_data_format)
        self.cv3 = layers.Conv2D(c_,(1,1),(1,1),use_bias=False,data_format=conv2d_data_format)
        self.cv4 = MyConv(co,1,1,use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation(nn.leaky_relu(0.1))
        self.m = [Bottleneck(c_,c_,shortcut,g,e=1.0) for _ in range(n)]
    
    def build(self,input_shape):
        super(BottleneckCSP, self).build(input_shape)
    
    def call(self, x):
        cv1 = self.cv1(x)
        
        for mm in self.m:
            cv1 = mm(cv1)
        
        return self.cv4(self.act(self.bn(tf.concat([self.cv3(cv1),self.cv2(x)],axis=channels_index))))
    
class C3(layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,input,filters,n=1,shortcut = True ,g=1,e=0.5) : # ch_in, ch_out,number, shortcut, groups, expansion
        super(C3,self).__init__()
        c_ = int(filters * e)
        self .cv1 = MyConv(input,c_,1,1)
        self .cv2 = MyConv(input,c_,1,1)
        self .cv3 = MyConv( 2 * c_, filters ,1)
        self .m = [Bottleneck(c_,c_,shortcut,g,e=1.0) for _ in range(n)]
        
    
    def build(self,input_shape):
        super(C3, self).build(input_shape)
    
    def call(self, x, *args, **kwargs): 
        
        cv1 = self.cv1(x)
        
        for mm in self.m:
            cv1 = mm(cv1)
        
        return self.cv3(tf.concat([cv1, self.cv2(x)], axis=channels_index))


class SPPF(layers.Layer):
    def __init__(self,ci,co,k=5):
        super(SPPF, self).__init__()
        c_ = ci //2 # hidden channels
        self.cv1 = MyConv( ci,c_,1,1)
        self.cv2 = MyConv(c_ * 4 ,co ,1,1)
        self.m = layers.MaxPool2D((k,k),1,data_format=conv2d_data_format)
    
    def build(self,input_shape):
        super(SPPF, self).build(input_shape)
    
    
    def call(self, inputs):
        inputs = self.cv1(inputs)
        # with warnings.catch_warnings():
            # warnings.simplefilter('ignore')
        y1 = self.m(inputs)
        y2 = self.m(y1)
        
        return self.cv2(tf.concat([inputs,y1,y2,self.m(y2)],1))
    
class ShuffleV2Block(layers.Layer):
    def __init__(self, inp,oup,stride=2):
        super(ShuffleV2Block,self).__init__()
        
        if not (1<= stride <=3):
            raise ValueError('illedgal stride value')
        self.stride = stride
        branch_features = oup //2
        assert (self.stride!=1) or (inp ==branch_features << 1)
        
        if(self.stride>1):
            self.branch1 = Sequential([
                self.depthwise_conv(inp,inp,kernel_size=3,stride=self.stride,padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(branch_features,kernel_size=(1,1),strides=(1,1),padding='valid',use_bias=False,data_format=conv2d_data_format),
                layers.BatchNormalization(),
                layers.Activation(nn.silu())
            ])
        else:
            self.branch1 = Sequential()
            
        self.branch2 = Sequential([
            layers.Conv2D(branch_features,(1,1),(1,1),use_bias=False,data_format=conv2d_data_format),
            layers.BatchNormalization(),
            layers.Activation(nn.silu()),
            self.depthwise_conv(branch_features,branch_features,3,self.stride,'same'),
            layers.BatchNormalization(),
            layers.Conv2D(branch_features,(1,1),(1,1),use_bias=False,data_format=conv2d_data_format),
            layers.BatchNormalization(),
            layers.Activation(nn.silu())
        ])
    
    def build(self,input_shape):
        super(ShuffleV2Block, self).build(input_shape)
    
    @staticmethod
    def depthwise_conv(i,o,kernel_size,stride = 1,padding = 'same',bias = False):
        return layers.Conv2D(o,(kernel_size,kernel_size),(stride,stride),use_bias=bias,padding=padding,data_format=conv2d_data_format)
    
    def call(self, inputs):
        if(self.stride == 1):
            x1,x2 = input.chunk(2,dim=1)
            out = tf.concat((x1,self.branch1(x2)),axis=channels_index)
        else:
            out = tf.concat((self.branch1(inputs),self.branch2(inputs)),axis=channels_index)
        out = channel_shuffle(out,2)
        return out
    
class BlazeBlock(layers.Layer):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1,2]
        if stride >1:
            self.use_pool = True
        else:
            self.use_pool = False
            
        self.branch1 = Sequential([
            layers.Conv2D(out_channels,(5,5),(1,1),use_bias=False,padding='same',groups=in_channels,data_format=conv2d_data_format),
            layers.BatchNormalization(),
            layers.Conv2D(out_channels,(1,1),(1,1),data_format=conv2d_data_format),
            layers.BatchNormalization()
        ])
        
        if(self.use_pool):
            self.shortcut = Sequential([
                layers.MaxPool2D((stride),stride,data_format=conv2d_data_format),
                layers.Conv2D(out_channels,(1,1),(1,1),data_format=conv2d_data_format),
                layers.BatchNormalization()
            ])
        self.relu = layers.Activation(nn.relu)
        
    def build(self,input_shape):
        super(BlazeBlock, self).build(input_shape)
    
    
    def call(self, inputs):
        branch1 = self.branch1(inputs)
        out = (branch1+self.shortcut(inputs) if(self.use_pool) else (branch1+inputs))
        
        return self.relu(out)
class SPP(layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,ci,filters , k=[5, 9, 13]):
        super(SPP, self).__init__()
        c_ = ci // 2  # hidden channels
        
        self.k = k
        
        self.filters=filters
        self.ci = MyConv(ci,c_, 1, 1)
        self.co = MyConv(c_ * (len(k) + 1),filters, 1, 1)
        self.m= [layers.MaxPool2D(pool_size=(m,m),strides=(1,1),padding='same',data_format=conv2d_data_format) for m in k]
    
    def build(self,input_shape):
        super(SPP, self).build(input_shape)
    
    
    def call(self, x):
        x = self.ci(x)
        c=self.co(tf.concat([x] + [m(x) for m in self.m], channels_index))
        return c

class Focus(layers.Layer):
    def __init__(self, gain = 2):
        self.gain = gain
    
    
    def build(self,input_shape):
        super(Focus, self).build(input_shape)
    
    
    def call(self,x):
        N,C,H,W = x.size() # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        
        x = x.view(N,C,H//s,s,W//s,s) #x(1,64,40,2,40,2)
        x = x.permute(0,3,5,1,2,4).contiguous() # x(1,2,2,64,40,40)
        return x.view(N,C *s * s , H // s ,W // s) #x(1,256,40,40)

class Concat(layers.Layer):
    def __init__(self,dimension=1):
        super(Concat,self).__init__()
        self.d=dimension
    
    def build(self,input_shape):
        super(Concat, self).build(input_shape)
    
    
    def call(self,x):
        return tf.concat(x,self.d)

#什么也不做的layer
class NoneLayer(layers.Layer):
    def __init__(self) :
        super(NoneLayer,self).__init__()
        
        
    def build(self,input_shape):
        super(NoneLayer, self).build(input_shape)
    
    def call(self,x):
        return x
    
class Detect(layers.Layer):
    stride = None
    export_cat = False#是否要用onnx导出
    def __init__(self, nc = 1,anchors=[],ch=[3]):
        #yolos 为例 nc=1,anchors 为一个3*6数组，ch为一个1*3数组
        super(Detect,self).__init__()
        self.training=True
        self.nc=nc #有多少个类
        self.no = nc + 5 + 10# 每个anchor 的输出数 默认16
        
        self.nl  = len(anchors) # 监测层数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        
        self.grid = [tf.zeros(1)] * self.nl #init grid
        
        self.anchors = tf.convert_to_tensor(np.reshape(anchors,(self.nl,-1,2) ),dtype=tf.float32)
        self.anchors_grid = tf.convert_to_tensor(np.reshape(self.anchors,(self.nl,1,-1,1,1,2)),dtype=tf.float32)
        self.m = [layers.Conv2D(self.no * self.na,1,data_format=conv2d_data_format,use_bias=True,bias_initializer='zeros') for x in ch] 
        # self.conv = layers.Conv2D(self.no * self.na,1,data_format=conv2d_data_format,use_bias=True,bias_initializer='zeros')
    def build(self,input_shape):
        super(Detect, self).build(input_shape)    

    # def call(self,x1,x2,x3):
    def call(self,x):
        x=x.copy()
        # x=[x1,x2,x3]
        z = []
        # print(x)
        if self.export_cat:
            for i in range(self.nl):
            # if(True):
                # ?奇怪，这个是在干嘛？
                x[i] = self.m[i](x[i]) #conv
                
                if channels_index == 3: 
                    bs,ny,nx,_ = x[i].shape # x(bs,255,20,20) to x(bs,3,20,20,85)
                else:
                    bs,_,ny,nx = x[i].shape # x(bs,255,20,20) to x(bs,3,20,20,85)
                    
                
                    
                x[i] = tf.transpose(tf.reshape(x[i],[bs,self.na,self.no,ny,nx]),perm=[0,1,3,4,2])
                
                if(self.grid[i].shape[2:4] != x[i].shape[2:4]):
                    self.grid[i],self.anchors_grid[i] = self._make_grid_new(nx,ny,i)
                # 将conv的
                y = np.full_like(x[i],0)
            
                y = y +tf.concat((tf.math.sigmoid(x[i][:,:,:,:,0:5]),tf.concat((x[i][:,:,:,:,5:15],tf.math.sigmoid(x[i][:,:,:,:,:,15:15+self.nc])),4)),4)
                box_xy = (y[:,:,:,:,0:2]*2. - 0.5 + self.grid[i]) * self.stride[i]
                box_wh = (y[:,:,:,:,2:4] * 2) ** 2 *self.anchors_grid[i]
                
                lambda1 = y[:,:,:,:,5:7] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                lambda2 = y[:,:,:,:,7:9] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                lambda3 = y[:,:,:,:,9:11] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                lambda4 = y[:,:,:,:,11:13] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                lambda5 = y[:,:,:,:,13:15] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                
                nclamb = y[:,:,:,:,15:15+self.nc] * self.anchors_grid[i] + self.grid[i] * self.stride[i]
                
                y = tf.concat([box_xy,box_wh,lambda1,lambda2,lambda3,lambda4,lambda5,nclamb],-1)
                
                z.append(tf.reshape(y,bs,-1,self.no))
        
            return tf.concat(z,1)
        for i in range(self.nl):
        # if(True):
            x[i] = self.m[i](x[i])
            
            if channels_index == 3: 
                bs,ny,nx,_ = x[i].shape # x(bs,255,20,20) to x(bs,3,20,20,85)
            else:
                bs,_,ny,nx = x[i].shape # x(bs,255,20,20) to x(bs,3,20,20,85)
                
            
            # 将x[i]重新编辑成 anchor输，输出条目类型和宽高
            # 用transposse将模型输出由bs,先验框数量，nc数量，宽高，变成bs,先验框数量，宽高，nc数量
            # if(channels_index==1):
            # print(x[i])
            x[i] = tf.reshape(x[i],(-1,self.na, self.no, ny, nx))
            x[i] =  tf.transpose(x[i],perm=[0,1,3,4,2])
           

            # tensorflow没有明确的训练和校对方法，但torch有，所以这里不用管先
            # 如果不是已经进入训练中，那就要先吧传入的值归零一下
            # if not self.training:
            #     # 先看看宽高是否正常，不正常重新计算宽高
            #     if self.grid[i].shape[2:4] !=x[i].shape[2:4]:
            #         self.grid[i] = self._make_grid(nx,ny)
                
            #     y = np.full_like(x[i],0)
            #     class_range = list(range(5)) + list(range(15,15+self.nc))
            #     y[...,class_range] = tf.sigmoid(x[i][...,class_range])
            #     y[..., 5:15] = x[i][...,5:15]
                
            #     # 这个应该是还原图片本身全部的宽高处理？或者是，获取图片的中心点位置？
            #     y[..., 0:2] = (y[..., 0:2] * 2 -  0.5 +self.grid[i]) * self.stride[i] #xy
            #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchors_grid[i] #wh
                
            #     y[..., 5:7] = y[..., 5:7] * self.anchors_grid[i] * self.grid[i] * self.stride[i]     #landmark x1 y1
            #     y[..., 7:9] = y[..., 7:9] * self.anchors_grid[i] * self.grid[i] * self.stride[i]     #landmark x2 y2
            #     y[..., 9:11] = y[..., 9:11] * self.anchors_grid[i] * self.grid[i] * self.stride[i]   #landmark x3 y3
            #     y[..., 11:13] = y[..., 11:13] * self.anchors_grid[i] * self.grid[i] * self.stride[i] #landmark x4 y4
            #     y[..., 13:15] = y[..., 13:15] * self.anchors_grid[i] * self.grid[i] * self.stride[i] #landmark x5 y5

            #     z.append(tf.reshape(y,(bs,-1,self.no)))
        return x #if self.training else (tf.concat(z,1),x)
                
                
        
    @staticmethod
    def _make_grid(self,nx=20,ny=20):
        yv,xv=tf.meshgrid(np.array([np.arange(ny,dtype=tf.float32),np.arange(nx,dtype=tf.float32)]),indexing='ij')
        return tf.reshape(tf.stack([xv,yv]),(1,1,ny,nx,2))
    
    def _make_grid_new(self,nx=20,ny=20,i=0):
        # d=self.anchors[i].device
        # 网格网络生成
        # 这个地方必须传ij，不然生成的和预要的不一样
        yv,xv = tf.meshgrid(np.array([np.arange(ny,dtype=tf.float32),np.arange(nx,dtype=tf.float32)]),indexing='ij')
        # 拼接数组，与cat不同的是，cat会把多个一维数组拼成一个长一维数组，stack是把n个一维数组拼成n维数组
        grid = tf.stack((xv,yv),2)
        #将拼接后的数组再次reshape成？？？
        grid = tf.reshape(grid,[1,self.na,ny,nx,2],dtype=float32)
        
        anchor_grid = (self.anchors[i].clone() * self.stride[i])
        anchor_grid = tf.reshape(anchor_grid,[1,self.na,1,1,2])
        anchor_grid = tf.tile(anchor_grid,(1,self.na,ny,nx,2))
        
        

