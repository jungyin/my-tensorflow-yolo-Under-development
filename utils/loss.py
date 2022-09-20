import tensorflow as tf
import numpy as np
from utils. general import bbox_iou
from tensorflow import keras
from tensorflow.keras import losses

#返回正、负标签平滑BCE目标
#？还是有点不太懂，这个是干嘛的
def smooth_BCE( eps = 0.1):
    return 1.0 - 0.5 * eps, 0.5*eps


# 编译目标
def build_targets(preds, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #计算传入的图片中的目标所在位置，并计算损失？
    det = model.model.layers[-1] if is_parallel(model) else model.model[-1]
    tcls, tbox, indices, anch, landmarks, lmks_mask  = [] , [] , [] , [] , [] , []
    na, nt = det.na, len(targets)  # number of anchors, targets # 获取anchors先验框和标签的数量
    
    gain = np.ones(17)
    
    nreshape = np.reshape( np.arange(na,dtype=np.float32),(na, 1))
    
    ai = np.repeat(nreshape,(nt),axis=1) # same as .repeat_interleave(nt) ？这是在计算啥的重复次数？
    
    tshape = [1]+targets.shape
    
    targets = tf.concat(( np.repeat(np.reshape(targets,tshape),na,axis=0), ai[:,:,None]), 2) # append anchor indices 

    g = 0.5 #bias
    off = np.array([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1]],dtype=np.float32) * g 
    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = np.array(preds[i].shape)[[3,2,3,2]] #xyxy gain
        
        gain[6:16] = np.array(preds[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]
        
        
        t = targets * gain
        
        if nt : # 在有标签时才会走这里
            # Matches
            r = t[:, :, 4:6] / anchors[:,None] #wh ratio 获取这个区域的概率是多少
            
            j = np.max(tf.maximum(r, 1. / r),2) < model.hyp['anchor_t'] #用于判断这个targets是不是一个合格的targets
          
            
            
            t = t[j] #filter 这里的j应该还是一个单独的boolean值，不是一个boolean数组，那这里应该取的下表位不是0，就是1咯

            #Offsets？在计算什么的补偿？
            gxy = t[:, 2: 4] #grid xy
            gxi = gain[[2,3]] - gxy #inverse ?奇怪，为啥要配grid xy的相反值
            
            
            j,k = tf.transpose(((gxy %1. <g) & (gxy  > 1.)))
            l,m = tf.transpose(((gxi %1. <g) & (gxi  > 1.)))
          
            
            j = np.stack((np.ones_like(j), j, k, l, m))
            t=np.resize(t,[1]+t.shape)
        
            t= np.repeat(t,5,axis=0)
            t= t[j]
            offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]
        else : #如果没有，取anchor 第一个索引？不太像啊
            t = targets[0].numpy()
            offsets = 0
        
        #Define
        
        
        ic= tf.cast(tf.transpose(t[:, :2]),tf.int64)
        
        img, c = ic[0],ic[1] # image,class 图片，类型
        gxy = t[:, 2:4] # grid xy 识别框的xy
        gwh = t[:, 4:6] # grid wh 识别框的wh
        gij = gxy - offsets #补偿值纠正（吧起始点做个纠偏）
        gxi, gyi = tf.transpose(gij) # grid xy indices # 识别框的标？
        
        # append
        a = t[:, 16]  # anchor indices 获取所有先验框的下标
        indices.append((tf.cast(img,tf.int64), a.astype(np.int64), np.clip(gyi, 0, gain[3] - 1).astype(np.int64), np.clip(gxi, 0, gain[2] - 1).astype(np.int64)))  # image, anchor, grid indices 将图片，先验框和标签下表存起来
        tbox.append(tf.concat((gxy - gij, gwh), 1))  # 标记框存入数组
        
        tcls.append(c)  # class
        anch.append(anchors[0])  # anchors 
        #landmarks
        lks = t[:,6:16]
        lks_mask = tf.where(lks < 0 , np.full_like(lks, 0.), np.full_like(lks, 1.0))


        #？这是我写的吗？
        #应该是关键点的坐标除以anch的宽高才对，便于模型学习。使用gwh会导致不同关键点的编码不同，没有统一的参考标准
        
        # 吧现有标签的宽高都做补偿值纠正1一下
        lks [: , [0, 1]] = (lks[:, [0, 1]] - gij)
        lks [: , [2, 3]] = (lks[:, [2, 3]] - gij)
        lks [: , [4, 5]] = (lks[:, [4, 5]] - gij)
        lks [: , [5, 6]] = (lks[:, [5, 6]] - gij)
    
        lks_mask_new = lks_mask
        lmks_mask.append(lks_mask_new)
        landmarks.append(lks)
    return tcls,tbox,indices,anch,landmarks,lmks_mask
        
    

def compute_loss(preds, targets, model):
    lcls, lbox, loss_obj, lmark = np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32)
    tcls, tbox, indices, anchors, tlandmarks, lmks_mask = build_targets(preds, targets, model)
    h = model.hyp
    
    BCEcls = losses.BinaryCrossentropy(from_logits = True) 
    BCEobj = losses.BinaryCrossentropy(from_logits = True)
    
    
    clsWeights = np.array([h['cls_pw']])
    objWeights = np.array([h['obj_pw']])
    
    landmarks_loss = LandmarksLoss(1.0)
    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps = 0.0)
    
    #Focal loss 
    g = h['fl_gamma'] # focal loss gamma?gamma是什么
    
    if(g > 0):
        BCEcls,BCEobj = FocalLoss(BCEcls,g),FocalLoss(BCEobj,g)
    
    #Losses 计算1
    nt = 0 #标签数量，number targets
    no = len(preds) # 输出数量,number output
    balance = [4.0,1.0,0.4] if no ==3 else [4.0, 1.0, 0.4, 0.1] #P3-5 或者 P3-6,什么意思？均衡了什么？

    for i, pred in enumerate(preds): #layer index,layer predictions  layer的下表和预猜测结果
        img, a, gy, gx =indices[i] #image,anchor, grid y, grid x
        
       
        # pred=tf.cast(pred,tf.int32)
        
        t_obj = np.zeros_like(pred[..., 0]) #target obj
        
        for_nt = img.shape[0] #number of targets 在for循环中的number targets  targets不是标签，是对象，别搞错了，然后又绕圈圈
        
        if for_nt :
            nt += for_nt 
            pred_set = pred.numpy()[img,a,gy,gx].astype(np.float32) # prediction subset corresponding to targets
            
            pred_xy = tf.sigmoid(pred_set[:, :2]) * 2 - 0.5
            pred_wh = (tf.sigmoid(pred_set[:, :2]) * 2) ** 2 * anchors[i]
            
            pred_box = tf.concat((pred_xy,pred_wh),1) #将宽高合并为一个box
            
            iou = bbox_iou(tf.transpose(pred_box),tf.transpose(tbox[i]),x1y1x2y2=False,CIoU=True)
            
            niou = iou.numpy()
            
            lbox += np.mean(1.0 - iou)# mean是取均值，别tm百度了 ,另外因为每个iou都必定减一，所以在外面减，不在里面减了
            
            #Objectness 反对？反对啥？ 
            # 还是不懂pytorch的detach，这个是相当于clone?
            #这个，应该就是所谓的惩罚了把？
            #?奇怪，这玩意咋赋值的？
            t_obj[img, a, gy, gx] = (1.0 - model.gr) + (model.gr * tf.clip_by_value(iou,0,tf.reduce_max(iou,-1)))
      
            
            # Classification 分类
            if model.nc > 1: #cls loss (only if mutiple classes) 计算class的loss（只有在多个分类的情况下才会使用）
                t = np.full_like(pred_set[:, 15:],cn) # targets
                t[range(for_nt), tcls[i]] = cp  
                lcls += BCEcls(pred_set[:,15:],t,clsWeights) # bce
              
            
            plandmarks = pred_set[:,5:15]
            # ?这是把marks都乘以anchors了？要干嘛？
            plandmarks[:, 0:2] = plandmarks[:, 0:2] * anchors[i]
            plandmarks[:, 2:4] = plandmarks[:, 2:4] * anchors[i]
            plandmarks[:, 4:6] = plandmarks[:, 4:6] * anchors[i]
            plandmarks[:, 6:8] = plandmarks[:, 6:8] * anchors[i]
            plandmarks[:, 8:10] = plandmarks[:, 8:10] * anchors[i] 
          
            
            lmark += landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])
     
        loss_obj += BCEobj(pred[..., 4],t_obj,objWeights) * balance[i]
        
    s = 3 / no
    
    lbox *= h['box'] * s
    loss_obj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    lmark *= h['landmark'] * s
    
    bs = t_obj.shape[0] #batch size
    
    # 计算本次loss
    
    loss = lbox + loss_obj + lcls + lmark
    
    return loss * bs , np.array((lbox[0],loss_obj[0],lcls,lmark,loss[0]),dtype=np.float32)
    
        
def is_parallel(model):
    # return type(model) in (keras.Model)
    isst = isinstance(model,keras.Model) 
    return isst

class FocalLoss(keras.Model):
    #这是一种侧重于处理样本分类不均衡的损失函数（大部分都不均衡吧？），让模型在训练过程中吧注意力更多放在稀少的目标上，此外，fc还可以吧容易分类的（识别精度较高/低，比如0.8，0.9或者0.1，0.2这种）权重降低，吧难分类的(比如0.5，0.6，0.4)的权重调高
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # 必须是 二分类交叉熵损失 binary cross entropy loss
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def call(self, pred, true):
        #这部分后面得再回来看看，毕竟函数还是不太懂
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        
        pred_prob = tf.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob) #获取标签置信度
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha) #获取α因子
        modulating_factor = (1.0 - p_t) ** self.gamma #获取调试因子
        loss *= alpha_factor * modulating_factor# 新的loss就是自己乘以α因子乘以调试因子

        #完成后按着reduction再对loss做一次处理
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        else:  # 'none'
            return loss


class QFocalLoss(keras.Model):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def call(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = tf.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = np.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        else:  # 'none'
            return loss

class WingLoss(keras.Model):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def call(self, x, t, sigma=1):
        weight = np.ones_like(t)
        weight[np.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff =tf.abs(diff)
        flag = tf.cast((abs_diff < self.w),tf.float32)
        y = flag * self.w * np.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return np.sum(y)

class LandmarksLoss(keras.Model):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()#nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def call(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (np.sum(mask) + 10e-14)