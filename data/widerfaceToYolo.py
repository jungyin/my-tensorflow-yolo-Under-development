from os import makedirs
import os.path
import sys
from tkinter import Y
import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path,type, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(os.path.join(f'{txt_path}/wider_face_split', f'wider_face_{type}_bbx_gt.txt'), 'r')
        lines = f.readlines()
        

        
        for line in lines:
            line = line.rstrip()
            # print(line)
            
            #来个非空判断
            if(line==''):
                continue
            
            
            
            # 如果计数器大于0，那么这一行就是box框的信息
            splits= line.split(' ')
            if(len(splits)>3):
                self.words[-1].append([float(x) for x in splits])
                # print(label)
                # labels.append(label)

            #如果这一行的结尾是.jpg，那说明这一行是图片的信息，那么下一行就是接下来几行是图片信息了
            elif line.endswith('.jpg'):

                path = txt_path +'/'+type+'/images/' + line
                # print(line)
                self.imgs_path.append(path)
                self.words.append([])
                
            #如果都不是，就跳过
            else:
                continue
        # self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape
        print('widerfaceToYolo这里？')
        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), targets


# widerface专用yolo转换工具
if __name__ == '__main__':
    
    root_path = 'F:/widerface'
    type = 'train'

    # 稍微晚些把labels数据存储的位置
    save_path = f'{root_path}/{type}/labels'
    if not os.path.isfile(os.path.join(root_path, type, 'label.txt')):
        print('Missing label.txt file.')
        exit(1)

    # 将bbox_gt中的东西读取出来
    
    
    
    '''
    bbox_gt中的每一个字段的作用
    
    0--Parade/0_Parade_marchingband_1_849.jpg (文件名)
    1(人脸的数量)
    449(人脸框所在位置的x轴,注意这个是起始点,不是中心点) 330(人脸框所在位置的y轴,注意这个是中心点,不是起始点) 122(图片宽) 149(图片高) 0 0 0 0 0 0 
    
    File name
    Number of bounding box
    x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
    
    长宽高外的属性注释：
    blur:是模糊度,分三档:0,清晰,1:一般般;2:人鬼难分
    express:表情
    illumination:曝光，分正常和过曝
    illumination：曝光，分正常和过曝
    occlusion：遮挡，分三档。0，无遮挡；1，小遮挡；2，大遮挡；
    invalid：是否可用（没弄明白），应该是由于遮挡过度导致人眼也很难识别出来
    pose：（疑似姿态？分典型和非典型姿态）
    
    
    '''
    
    aa = WiderFaceDetection(root_path,type)
    count = 0
    
    print(len(aa.imgs_path))
    print(len(aa.words))
    
    wordslen=len(aa.words)
    imgslen=len(aa.words)
    
    
    for i in range(len(aa.imgs_path)):
        # print(i, aa.imgs_path[i])
        
        img = cv2.imread(aa.imgs_path[i])
        base_img = os.path.basename(aa.imgs_path[i])
        # base_txt = os.path.basename(aa.imgs_path[i])[:-4] + ".txt"
        save_img_path = os.path.join(save_path, base_img)
        pa=aa.imgs_path[i].split('/')
        
        cardtype=pa[-2]
        name=pa[-1].replace('.jpg','.txt')
        
        save_txt_path = save_path+'/'+cardtype
        
        # print(pa[-1])
        
        if not os.path.exists(save_txt_path):
            # print('创建了吗')
            makedirs(save_txt_path)
        save_txt_path=save_txt_path+'/'+name
        
        height, width, _ = img.shape
        labels = aa.words[i]
        annotations = np.zeros((0, 14))
        if len(labels) == 0:
            continue
        
       
       
        str_label = ''
        #不行了，label用不了，不知道训练时候咋读分类这里就写不了了
        for idx, label in enumerate(labels):
            # bbox
            
            label[0] = max(0, label[0])
            label[1] = max(0, label[1])
            label[2] = min(width - 1, label[2])
            label[3] = min(height - 1, label[3])
            
            
            
            
            # 注意：这个获取的是中心点，不是x和y的起始坐标！
            label[0] = (label[0] + label[2] / 2) / width  # cx
            label[1] = (label[1] + label[3] / 2) / height  # cy
            label[2] = label[2] / width  # w
            label[3] = label[3] / height  # h
        
            
            str_label = str_label + str('0')
            
            for i in range(len(label)):
                str_label = str_label + ' '+ str(label[i])
            str_label = str_label.replace('[', ' ').replace(']', ' ')
            str_label = str_label.replace(',', ' ') + '\n'
            
        
        if(len(str_label.split('\n'))>1):
            with open(save_txt_path, "w") as f:
                f.write(str_label)
                
                strs = str_label.split('\n')
                
                # print(str_label)
                # for items in strs:
                    # item = items.split(' ')
                    # if(len(item)>1):
                      
                    #     xc = int(float(item[1])*width)
                    #     yc = int(float(item[2])*height)
                    #     w = int(float(item[3])/2*width)
                    #     h = int(float(item[4])/2*height)
                        
                        # cv2.rectangle(img,(xc-w,yc-h),(xc+w+w,yc+h+h),(255,255,255),2)
                # cv2.imwrite(save_img_path, img)
                # count+=1
                # if(count>1):
                #     break
        # if(save):
        #     cv2.imwrite(save_img_path, img)
        # break
        
    print('结束')