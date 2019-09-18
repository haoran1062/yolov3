# encoding:utf-8
import os, sys, time
sys.path.insert(0, sys.path[0]+'/modules')
from backbones.OriginResNet import resnet18, resnet34, resnet50
# print(sys.path)
import torch, torch.nn as nn
import torch.nn.functional as F
from FPN import FPN
# from modules.resnet50_yolo_style_fpn_yolov3 import Config
from compute_utils import *
from configs.resnet18_yolo_style_fpn_yolov3 import Config

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
bce_log_loss = nn.BCEWithLogitsLoss()

class YOLOLayer(nn.Module):
    def __init__(self, cfg, lbd_cfg, logger=None, vis=None, img_size=416):
        super(YOLOLayer, self).__init__()
        self.obj_lbd = cfg['obj_lbd']
        self.noobj_lbd = cfg['noobj_lbd']
        self.device = cfg['device']
        self.lbd_cfg = lbd_cfg
        self.class_num = cfg['class_num']
        self.anchors = torch.Tensor(cfg['anchor_list']).to(self.device)
        self.anchor_num = len(self.anchors)
        self.stride = cfg['stride']
        self.img_size = img_size
        self.logger=logger
        self.vis=vis
        self.iou_thresh=cfg['iou_thresh']
        self.grid_num = img_size // self.stride
        self.batch_size = 0
        self.create_grid(img_size // self.stride)

    def create_grid(self, grid_num, cuda=True):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.grid_num = grid_num
        nx, ny = self.grid_num, self.grid_num
        y_it, x_it = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((x_it, y_it), 2).view(1, 1, ny, nx, 2).type(FloatTensor)
        self.anchor_wh = self.anchors / self.stride
        self.anchor_wh = self.anchor_wh.type(FloatTensor)
        self.anchor_decoder_wh = self.anchor_wh.view(1, len(self.anchors), 1, 1, 2).type(FloatTensor)

    def decoder(self, eval_tensor):
        # batch_size = eval_tensor.shape[0]


        last_dim_size = eval_tensor.shape[-1]
        output_tensor = eval_tensor.clone()
        output_tensor[..., 0] = output_tensor[..., 0].sigmoid()
        output_tensor[..., 1:3] = output_tensor[..., 1:3].sigmoid() 
        output_tensor[..., 1:3] += self.grid_xy
        output_tensor[..., 3:5] = output_tensor[..., 3:5].exp() * self.anchor_decoder_wh

        output_tensor[..., 1:5] *= self.stride
        output_tensor[..., 5:] = output_tensor[..., 5:].sigmoid()

        return output_tensor.view(self.batch_size, -1, last_dim_size)

    def forward(self, x):
        self.batch_size = x.shape[0]
        grid_size = x.shape[2]
        if grid_size != self.grid_num:
            print('-'*20)
            print(grid_size, self.grid_num)
            self.create_grid(grid_size, x.is_cuda)
            print(grid_size, self.grid_num)
            print('-'*20)
        x = x.view(self.batch_size, self.anchor_num, -1, self.grid_num, self.grid_num).permute(0, 1, 3, 4, 2).contiguous()        

        if self.training:
            return x
        else:
            return self.decoder(x)



class YOLO(nn.Module):

    def __init__(self, cfg, logger=None, vis=None, device='cuda:0'):
        super(YOLO, self).__init__()
        self.cfg=cfg
        self.device = device
        self.logger=logger
        self.vis=vis
        self.lbd_cfg = cfg.train_config['lbd_map']
        self.model_list = nn.ModuleList()
        self.model_names = []

        self.build_models()

    
    def init_yolo_layers(self, yolo_layer_config_list, logger=None, vis=None):
        for now_map in yolo_layer_config_list:
            self.model_list.append(YOLOLayer(now_map, self.lbd_cfg, logger=logger, vis=vis))
    
    def build_models(self):
        
        self.load_backbone(self.cfg.backbone_type, self.cfg.backbone_config)
        self.model_names.append('backbone')
        self.load_fpn(self.cfg.fpn_config)
        self.model_names.append('fpn')
        self.init_yolo_layers(self.cfg.yolo_layer_config_list, logger=self.logger, vis=self.vis)
        for i in range(len(self.cfg.yolo_layer_config_list)):
            self.model_names.append('yolo_layer_%d'%(i))

    def load_backbone(self, backbone_type, cfg):
        if backbone_type == 'resnet18':
            backbone = resnet18(True, **cfg)

        else:
            print(backbone_type, ' not support!!!')
        self.model_list.append(backbone)
        # return backbone
    
    def load_fpn(self, cfg):
        fpn = FPN(cfg)
        self.model_list.append(fpn)

        # return fpn


    def forward(self, x, cuda=True):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # print('start forward!')        
        pred_list = []
        
        for it, (nn_name, nn_layer) in enumerate(zip(self.model_names, self.model_list)):
            if nn_name == 'backbone':
                feature_map_list = nn_layer(x)
            elif nn_name == 'fpn':
                pyramid_output_list = nn_layer(feature_map_list)
            elif nn_name[:len('yolo')] == 'yolo':
                layer_id = int(nn_name.split('_')[-1])
                now_scale_pred = nn_layer(pyramid_output_list[layer_id])
                pred_list.append(now_scale_pred)

        if self.training:
            return pred_list
        else:
            return torch.cat(pred_list, 1)
