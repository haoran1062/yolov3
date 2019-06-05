# encoding:utf-8
import os, numpy as np, random, cv2, logging, json
import torch

from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from modules.backbones.OriginResNet import resnet18, resnet50
from modules.FPN import FPN
from modules.YOLOv3 import YOLO
from configs.resnet50_yolo_style_fpn_yolov3 import Config

def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))

    return device

def compute_iou_matrix(bbox1, bbox2):
    '''
        input:
            tensor bbox1: [N, 4] (x1, y1, x2, y2)
            tensor bbox2: [M, 4]
        
        process:
            1. get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point
            2. get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point
            3. expand left-top/right-bottom list to [N, M] matrix
            4. Intersection W_H = right-bottom - left-top
            5. clip witch W_H < 0 = 0
            6. Intersection area = W * H
            7. IoU = I / (bbox1's area + bbox2's area - I)

        output:
            IoU matrix:
                        [N, M]

    '''
    if not isinstance(bbox1, torch.Tensor) or not isinstance(bbox2, torch.Tensor):
        print('compute iou input must be Tensor !!!')
        exit()
    N = bbox1.size(0)
    M = bbox2.size(0)
    b1_left_top = bbox1[:, :2].unsqueeze(1).expand(N, M, 2) # [N,2] -> [N,1,2] -> [N,M,2]
    b2_left_top = bbox2[:, :2].unsqueeze(0).expand(N, M, 2) # [M,2] -> [1,M,2] -> [N,M,2]

    left_top = torch.max(b1_left_top, b2_left_top)  # get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point

    b1_right_bottom = bbox1[:, 2:].unsqueeze(1).expand(N, M, 2)
    b2_right_bottom = bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)

    right_bottom = torch.min(b1_right_bottom, b2_right_bottom)  # get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point

    w_h = right_bottom - left_top   # get Intersection W and H
    w_h[w_h < 0] = 0    # clip -x to 0

    I = w_h[:, :, 0] * w_h[: ,: ,1] # get intersection area

    b1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    b1_area = b1_area.unsqueeze(1).expand_as(I) # [N, M]
    b2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    b2_area = b2_area.unsqueeze(0).expand_as(I) # [N, M]

    IoU = I / (b1_area + b2_area - I)   # [N, M] 

    return IoU


def nms(bboxes,scores,threshold=0.25):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def from_img_path_get_label_list(img_path, img_size=(448, 448)):
    '''
        return [ [label, x0, y0, x1, y1] ]
    '''
    label_path = img_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
    label_list = []
    with open(label_path, 'r') as f:
        for line in f:
            ll = line.strip().split(' ')
            label = int(ll[0])
            x = float(ll[1])
            y = float(ll[2])
            w = float(ll[3])
            h = float(ll[4])
            x0 = int( (x - 0.5*w) * img_size[0] )
            y0 = int( (y - 0.5*h) * img_size[1] )
            x1 = int( (x + 0.5*w) * img_size[0] )
            y1 = int( (y + 0.5*h) * img_size[1] )
            label_list.append([label, x0, y0, x1, y1])
    return label_list

def bbox_un_norm(bboxes, img_size=(448, 448)):
    (w, h) = img_size
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * w)
        bbox[1] = int(bbox[1] * h)
        bbox[2] = int(bbox[2] * w)
        bbox[3] = int(bbox[3] * h)
    return bboxes


mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def create_logger(base_path, log_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger

def warmming_up_policy(now_iter, now_lr, stop_down_iter=1000):
    if now_iter <= stop_down_iter:
        now_lr += 0.000001
    return now_lr

def learning_rate_policy(now_iter, now_epoch, now_lr, lr_adjust_map, stop_down_iter=1000):
    now_lr = warmming_up_policy(now_iter, now_lr, stop_down_iter)
    if now_iter >= stop_down_iter and now_epoch in lr_adjust_map.keys():
        now_lr = lr_adjust_map[now_epoch]

    return now_lr

def get_config_map(cfg):
    
    return cfg.train_config

def init_model(config_map, backbone_type_list=['resnet18', 'resnet50'], device='cuda:0'):
    assert config_map.backbone in backbone_type_list, 'backbone not supported!!!'
    if config_map.backbone == backbone_type_list[1]:
        backbone_net = resnet50()
        resnet = models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = backbone_net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        backbone_net.load_state_dict(dd)

    if config_map.backbone == backbone_type_list[0]:
        backbone_net = resnet18()
        resnet = models.resnet18(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = backbone_net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        backbone_net.load_state_dict(dd)
    
    fpn = FPN(backbone_net, config_map.fpn_config)
    yolo = YOLO(fpn, config_map)

    return yolo

def init_lr(cfg):
    config_map = cfg.train_config
    learning_rate = 0.0
    if config_map['resume_epoch'] > 0:
        for k, v in config_map['lr_adjust_map'].items():
            if k <= config_map['resume_epoch']:
                learning_rate = v 
    return learning_rate

def addImage(img, img1): 
    
    h, w, _ = img1.shape 
    # 函数要求两张图必须是同一个size 
    img2 = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) #print img1.shape, img2.shape #alpha，beta，gamma可调 
    alpha = 0.5
    beta = 1-alpha 
    gamma = 0 
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add


def draw_classify_confidence_map(img, pred_tensor, S, color_list, B=2):
    if isinstance(img, torch.Tensor):
        img = img.mul(255).byte()
        img = img.cpu().numpy()
    pred_tensor = pred_tensor.data
    pred_tensor = pred_tensor.squeeze(0) 
    h, w, c = img.shape 
    empty_img = np.zeros((h, w, c), np.uint8)
    empty_img.fill(255)
    for i in range(S):
        for j in range(S):
            cv2.line(img, (0, int(j * h/S)), (w, int(j * h/S)), (0, 0, 0), 3)
            cv2.line(img, (int(i * w/S), 0), (int(i * w/S), h), (0, 0, 0), 3)
            if i < S and j < S:
                # color_index = torch.max(pred_tensor[i,j,5*B:],0)
                max_prob, cls_index = torch.max(pred_tensor[i,j,5*B:],0)
                # print(cls_index)
                color_index = cls_index.item()
                empty_img[int(i * h/S):int((i+1) * h/S), int(j * w/S):int((j+1) * w/S)] = np.array(color_list[color_index], np.uint8)
    img = addImage(img, empty_img)
    return img

def get_class_color_img():
    img = np.zeros((750, 300, 3), np.uint8)
    h, w, c = img.shape
    img.fill(255)
    color_img = np.zeros(img.shape, np.uint8)
    clsn = 20
    cross = int(h / clsn)
    for i in range(clsn):
        color_img[i*cross:(i+1)*cross] =  np.array(Color[i], np.uint8)
        cv2.putText(img, '%s'%(VOC_CLASSES[i]), (30, int(i * cross) + int(cross/1.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, 30)
    img = addImage(img, color_img)
    return img

if __name__ == "__main__":
    from YOLODataLoader import yoloDataset
    
    b1 = [
        [10, 20, 100, 123],
        [200, 300, 300, 350]
    ]

    b2 = [
        [50, 60, 150, 120],
        [0, 10, 123, 150],
        [170, 190, 310, 400]
    ]

    nb1 = np.array(b1, np.float32)
    nb2 = np.array(b2, np.float32)

    tb1 = torch.from_numpy(nb1)
    tb2 = torch.from_numpy(nb2)

    iou = compute_iou_matrix(tb1, tb2)
    print(iou)
