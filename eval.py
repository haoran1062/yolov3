# encoding:utf-8
import os, cv2, logging, numpy as np, time, json, argparse
import torch

import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from modules.backbones.OriginResNet import resnet50, resnet18
from torchsummary import summary
from modules.YOLODatasets import yoloDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from modules.Loss import *
import multiprocessing as mp
from torchsummary import summary
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from utils.visual import Visual
from configs.resnet50_yolo_style_fpn_yolov3 import Config
from utils.train_utils import *
from modules.compute_utils import *
from utils.eval_utils import *

if __name__ == "__main__":
    
    config_map = Config()
    eval_config = config_map.eval_config
    device = select_device()

    yolo = init_model(config_map)
    yolo_p = nn.DataParallel(yolo.to(device), device_ids=eval_config['gpu_ids'])
    yolo_p.load_state_dict(torch.load(eval_config['load_from_path']))
    yolo_p.eval()

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = yoloDataset(list_file=eval_config['test_txt_path'], train=False, transform = transform, test_mode=False, device='cuda:0')
    test_loader = DataLoader(test_dataset, batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['worker_num'], collate_fn=test_dataset.collate_fn)

    test_iter = iter(test_loader)


    
    [layer.eval() for layer in yolo_p.module.yolo_layer_list]
    yolo_p.module.fpn.backbone.eval()
    yolo_p.module.fpn.eval()
    with torch.no_grad():
        for i in range(200):
            # yolo_p.eval()
            
            img, label_bboxes = next(test_iter)
            img, label_bboxes = img.to(device), label_bboxes.to(device)
            # print(img.is_cuda)
            print(img.shape)
            pred = yolo_p(img)

            output = simplify_eval_output(pred)
            print(output)

            # cls_id, gt_xy, gt_wh, index_list = yolo.yolo_layer_list[-1].encoder(label_bboxes)
            # print(cls_id, gt_xy, gt_wh, index_list)
            # print(img.shape, label_bboxes.shape)
            # print(label_bboxes)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            img = un_normal_trans(img.squeeze(0)).permute(1, 2, 0)
            # print(img.shape)
            if isinstance(img, torch.Tensor):
                img = img.mul(255).byte()
                img = img.cpu().numpy()
            # temp_label_bboxes = label_bboxes[label_bboxes[:, 0] == 1.0]
            # print(temp_label_bboxes, temp_label_bboxes.shape)
            # img = draw_debug_rect(img.permute(1, 2 ,0), temp_label_bboxes[..., 2:], temp_label_bboxes[..., 1])
            cv2.imshow('img', img)
            
            if cv2.waitKey(12000)&0xFF == ord('q'):
                break


