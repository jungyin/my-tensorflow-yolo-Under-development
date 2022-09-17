import argparse
from email.policy import strict
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn
import numpy as np
from tqdm import tqdm
import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from utils.autoanchor import check_anchor_order


from utils.loss import compute_loss
from utils.plots import plot_evolution
from utils.tensorflow_utils import intersect_dicts

from tensorflow.keras  import models
import pynvml
import threading

from utils.autoanchor import check_anchors
from utils.plots import plot_labels,plot_images
from utils.face_datasets import create_dataloader
from models.yolo import Model
from models.common import NoneLayer
from utils.general import init_seeds,check_dataset,check_img_size,print_mutation,set_logging,check_file,increment_path,labels_to_class_weights
from utils.google_utils import attempt_download
from utils.metrics import fitness
from utils.tensorflow_utils import ModelEMA

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as mvn2
import gc


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print('gpu可用',end='')
print(tf.test.is_gpu_available)

logger = logging.getLogger(__name__)

try:
    # import wandb
    wandb = None
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def buildFreeze(model,freeze):
    
    for k in model.layers:
        # v.trainable=True
        if(k is tf.keras.Sequential):
            buildFreeze(k,freeze)
        if any (x in k for x in freeze):
            k.trainable=False

def train(hyp,opt,wandb=None):
    
    # 初始化gpu存储空间余量展示器
    pynvml.nvmlInit()
    useGpu=True
    # useGpu= tf.test.is_gpu_available() #当前gpu是否可用
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
    
    logger.info(f'Hyperparameters{hyp}')
    save_dir,epochs,batch_size,total_batch_size,weights,rank = \
        Path(opt.save_dir),opt.epochs,opt.batch_size,opt.total_batch_size,opt.weights,opt.global_rank
        
    # wdir = save_dir/'weights'
    # run\train\exp就是在这里被创建的
    # wdir.mkdir(parents=True,exist_ok=True)
    # last = wdir/ 'last.h5'
    # best = wdir/ 'best.h5'
    # 模型跑不起来，先注释这部分
    
    results_file = save_dir / 'results.txt'
    
    # with open(save_dir/'hyp.yaml','w') as f:
        # yaml.dump(hyp,f,sort_keys=False)
    # with open(save_dir/'opt.yaml','w') as f:
        # yaml.dump(vars(opt),f,sort_keys=False)
    
    plots = not opt.evolve
    # cuda = device.typ != 'cpu'
    init_seeds(2 + rank)
    with open (opt.data) as f:
        data_dict=yaml.load(f,Loader=yaml.FullLoader)
    # with tensorflow_distributed_zero_first(rand):
    check_dataset(data_dict)
    train_path = data_dict['train']
    test_path = data_dict['test']
    #这次要训练的类别数量
    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    
    
    pretrained = weights.endswith('.ckpt')
    if(pretrained):
        attempt_download(weights)
        ckpt = models.load_model(weights)
        if hyp.get('anchors'):
            # 如果在配置表里有anchors，就更换掉model中已有的anchors
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])
        # 初始化model
        model = Model(opt.cfg or ckpt['model'].yaml,ch=3,nc=nc,format=opt.format)
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else[]
        state_dict = ckpt['model'].float().state_dict# 将配置表中配的参修改为float类型
        state_dict = intersect_dicts(state_dict,strict=False)
        # model.load_state_dict(state_dict,strict=False)
        
    else:
        model = Model(opt.cfg,ch=3,batch_size=opt.batch_size,img_size=opt.img_size,nc=nc,format=opt.format)
    
  
    
    # freeze  = [] # 要作冻结的模块
    
    # #循环遍历模型中所有的模块,并为其配置冻结
    # buildFreeze(model,freeze)
    # model.name_scope
    
    nbs = 64 # nominal batch size
    
    accumulate = max(round(nbs),1) #累计损失，在优化前的
    
    hyp['weight_decay'] *= total_batch_size * accumulate /nbs # weights_decay 缩放
        
    
    #优化器
    optimizer = SGD(hyp['lr0'],momentum=hyp['momentum'],nesterov=True)
    if opt.adam:
        optimizer = Adam(hyp['lr0'],beta_1=hyp['momentum'],beta_2=0.999)
    
        
    
    # 配置学习曲线
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf'] #cosine
    # scheduler = lr_
    
    if wandb and wandb.run is None:
        opt.hyp = hyp
        wandb_run = wandb.init(config=opt, resume="allow",
                            project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                            name=save_dir.stem,
                            id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    loggers = {'wandb':wandb}
    
    start_epoch,best_fiteness = 0 , 0.0
    #有点蛋疼，以前保存的模型都是h5，这里要ckpt的，等我有了模型再来试试吧
    #判断h5模型有没有成功加载
    # if pretrained:
        #先加载 优化器
        # if ckpt['optimizer'] is not None:
            # optimizer.load_state_dict(ckpt['optimizer'])
    
    
    # 步进，正常应该是在里面算，不过，这里先丢出来吧
    # gs = int(max(model.stride))
    gs = 32
    imgsz, imgsz_test = [check_img_size(x,gs) for x in opt.img_size]
    

 
    model.build([1,128,128,3])
    
    
    model.summary()
    # model.load_weights()
    
    # EMA
    # ema = ModelEMA(model) if rank in [-1, 0] else None

        
    # Trainloader
    dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)
    
    
    mlc = np.concatenate(dataset.labels,0)[:, 0].max() #确认可能性最大的class
    # ema=None
    
    dl = len(dataset) #  data len 当前图片数据的总个数
    nb = dl // batch_size # number of batches 当前数据按照batchs分了后需要轮读几次才能全部读取完，跑完一个epochs
    if (len(dataset)%batch_size!=0) :
        nb+=1
    
    
    
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    
    if rank in [-1,0]:
        # ema = tf.train.ExponentialMovingAverage(decay=0.9999,num_updates=start_epoch * nb // accumulate)
        # testdataset = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
        #                                         hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
        #                                         world_size=opt.world_size, workers=opt.workers,
        #                                         pad = 0.5)
        
        if not opt.resume:
            labels = np.concatenate(dataset.labels,0)
            
            if plots :
                pass
                # plot_labels(labels, save_dir, loggers)
                # if tb_writer:
                #     tb_writer.add_histogram('classes', labels[:, 0], 0)
    
            if not opt.noautoanchor:
                check_anchors(dataset, model = model , thr = hyp['anchor_t'],imgsz= imgsz)

    
  # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc) * nc  # attach class weights
    model.names = names
    # 训练开始
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    
    # scheduler.last_epoch = start_epoch
    # scaler = amp
    
    # scheduler.last_epoch = start_epoch - 1  # do not move
    logger.info('Image sizes %g train, %g test\n'
            'Logging results to %s\n'
            'Starting training for %g epochs...' % (imgsz, imgsz_test,  save_dir, epochs))
   
    # model.save("./mask_detector")
   
    for epoch in range(start_epoch,epochs):
        # 更新图片权重
        if opt.image_weights:
            # 生成索引
            if rank in [-1, 0]:
                cw = model.class_widths * (1 - maps) ** 2 /nc # 类权重(class weights)
                iw = model.class_widths * (1 - maps) ** 2 /nc # 图片权重(image weights)
                dataset.indices = random.choice(range(dataset.n),weights=id, k = dataset.n)
            #如果是ddp模式(ddp大概是分布式训练？)，就需要配置广播
            # if rank != -1 :
                # indices = np.array[dataset.indices] if rank == 0 else np.zeros(dataset.n,dtype=np.int8)
        
        mloss = np.zeros(5)    
        logger.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'landmark', 'total', 'targets', 'img_size'))
        # dataset len
      
        pbar = range(nb)
        if(rank in [-1,0]):
            pbar = tqdm(pbar,total=nb)
        
        
        for i  in pbar:
            # number integrated batches (since train start)
            # 这个参大概意思是本次训练累计使用了多少图片了
            ni = i + nb * epoch 
            
            # 通过批次数读取一次epochs所需要使用到的数据
            (imgs, targets, paths)=[],[],[]
            
            batch_index = 0
            
            for imgi in range(i * batch_size , i * batch_size + batch_size):
                
                
                # 下表从0开始，删一个
                if(imgi < dl ):
                    img,target,path = dataset.__getitem__(imgi)
                    # 在getitem里改图片的shape有点困难，还是在外面改吧
                    if(opt.format=='NHWC'):
                        img = tf.transpose(img,perm=[1,2,0]).numpy()
                    
                    imgs.append(img)
                    targets.append(target)
                    paths.append(path)
                    

            (imgs, targets, paths) =  dataset.collate_fn(imgs,targets,paths)
            
            
            
            imgs = np.array(imgs,dtype=np.float32) / 255.0 #将图片从uint8的0-255转化为float32的0到1
            
            #预处理了图像一下
            if ni <= nw:
                xi = [0,nw]
                
                # ？奇怪，为啥要获取这个？
                # 这个东西大概意思是求nbs/tbs 的第ni个线性变化值，最小必须是1，然后会做取整操作
                # 累计？累计了什么？
                accumulate = max(1,np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                # 这是在干嘛？
                # for j, x in enumerate(opt):
                #通过_set_hyper调整lr和mom
                
                # optimizer._set_hyper("learning_rate", np.interp(ni, xi, 0.0, hyp['lr0'] * lf(epoch)))
                optimizer._set_hyper("momentum", np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']]))

            # 是否要做缩放
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 * gs) // gs * gs 
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]# new shape (stretched to gs-multiple)
                    imgs = tf.image.resize(imgs, ns ,  tf.image.ResizeMethod.BILINEAR,False)
                
            #Forward
            with tf.GradientTape() as gt:
             
                pred = model(imgs)
                
                
                loss, loss_items = compute_loss(pred,targets,model)
               
                print(loss_items,end='')
                print(loss)
                if rank != -1 :
                # gradient averaged between devices in DDP mode
                # ddp模式下需要配置设备间的梯度平均值
                    loss *= opt.world_size 
                
                grads = gt.gradient(loss,model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads,model.trainable_variables) if grad is not None) # 优化函数应用梯度进行优化
                # optimizer.apply_gradients(zip(grads,model.trainable_variables)) # 不要使用trainable_variables

            
            #?什么玩意
            # tensorflow的混合精度学习。。后面再琢磨，现在不想
            
            # scaler.scale(loss).backward()
            
            
            # Optimize
            # if ni % accumulate == 0 :
                # scaler.step(optimizer)
                # scaler.update()
               
                # optimizer.zero_grad()
                # if ema :
                #     ema.update(model)
            
            
            # Print 
            if rank in [-1,0]:
                mloss = (mloss * i + loss_items) / (i+1) # 更新当前平均loss
                
                mem = '%.3G' % ( pynvml.nvmlDeviceGetMemoryInfo(handle).used  / 1E9 if useGpu else 0)
                s = ('%10s' * 2 + '%10.4g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1] if opt.format == 'NCHW' else imgs.shape[1])
                pbar.set_description(s)
            # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    f = ''
                    Thread(target=plot_images, args=(imgs, targets, paths, f, opt.format), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})
                    
            # 单次训练完成，清理掉img，paths和 targets
            del imgs, targets, paths
            # # 跑完一次epoch记得gc叫出来一下，做个深度清理
            gc.collect()
        
    model.save("mask_detector")


if __name__ == '__main__':
    
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/widerface.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=40, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[256, 256], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=False, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    
    parser.add_argument('--cache-images', action='store_true',default=False, help='cache images for faster training')
    # 是否使用先前已有的权重文件进行训练，默认是使用
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # 用那个设备做训练，torch在我的这种垃圾显卡上会提示性能不够不给用，所以这里用cpu
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # （可能）训练过程中图片的缩放范围
    parser.add_argument('--multi-scale', action='store_true', default=False, help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # 是否使用adam
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # 是否使用SyncBatchNorm，这是torch特有的一个多线程优化用的同步的bn
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--format', default='NHWC', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    
    
    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
        # check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-2]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    # device = select_device(opt.device, batch_size=opt.batch_size)
    # if opt.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        train(hyp, opt,  wandb)
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt,  wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
