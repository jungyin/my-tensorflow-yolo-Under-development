import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small as MNV3S
import argparse
import os
import numpy as np
from utils.face_datasets import create_dataloader
import yaml
from warnings import warn
from utils.general import check_img_size,check_dataset,loadOpt

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset",
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="my_mask_detector_mv3.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())


INIT_LR = 1e-5
EPOCHS = 15
BS = 32


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
opt = loadOpt()
    
with open(opt.hyp) as f:
	hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
	if 'box' not in hyp:
		warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
				(opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
		hyp['box'] = hyp.pop('giou')
gs = 32
with open (opt.data) as f:
	data_dict=yaml.load(f,Loader=yaml.FullLoader)
	# with tensorflow_distributed_zero_first(rand):
	check_dataset(data_dict)
	train_path = data_dict['train']
	test_path = data_dict['test']
imgsz, imgsz_test = [check_img_size(x,gs) for x in opt.img_size]

rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)

kk = 0


