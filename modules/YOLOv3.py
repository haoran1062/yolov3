# encoding:utf-8
import os, sys, time
sys.path.insert(0, '/data/projects/my_yolov3/modules')
from backbones.OriginResNet import resnet18, resnet34, resnet50
# print(sys.path)
import torch, torch.nn as nn
import torch.nn.functional as F
from FPN import FPN
# from modules.resnet50_yolo_style_fpn_yolov3 import Config
from compute_utils import *
from configs.resnet18_yolo_style_fpn_yolov3 import Config

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
        # self.grid_num = torch.Tensor(cfg['grid_num']).to(self.device)
        self.stride = cfg['stride']
        self.img_size = img_size
        self.logger=logger
        self.vis=vis
        # self.create_grid()
        self.iou_thresh=cfg['iou_thresh']
        self.grid_num = 0
        self.batch_size = 0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_log_loss = nn.BCEWithLogitsLoss()

    def create_grid(self, grid_num, cuda=True):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.grid_num = grid_num
        nx, ny = self.grid_num, self.grid_num
        x_it, y_it = torch.meshgrid([torch.arange(nx), torch.arange(ny)])
        # print(x_it.shape, y_it.shape)
        self.grid_xy = torch.stack((x_it, y_it), 2).view(1, 1, nx, ny, 2).type(FloatTensor)
        # print(self.grid_xy.shape)
        self.anchor_wh = self.anchors / self.stride
        self.anchor_wh = self.anchor_wh.type(FloatTensor)
        self.anchor_decoder_wh = self.anchor_wh.view(1, len(self.anchors), 1, 1, 2).type(FloatTensor)


    def encoder(self, target, cuda=True):
        '''
            input:
                target: [image_id, cls_id, Cx, Cy, w, h]

            output:

        '''


        ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        obj_mask = ByteTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(0)
        noobj_mask = ByteTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(1)
        tx = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(0)
        ty = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(0)
        tw = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(0)
        th = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num).fill_(0)
        tcls = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num, self.class_num).fill_(0)
        # tlabel = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num, 1).fill_(0)

        # target_tensor = FloatTensor(self.batch_size, self.anchor_num, self.grid_num, self.grid_num, self.class_num + 5)
        
        img_id, label = target[:, :2].long().t()
        # print(label)
        # convert normilzed 0~1 coord to grid normilzed coord
        target_bboxes = target[:, 2:] * self.grid_num
        gt_xy = target_bboxes[:, :2]
        gt_wh = target_bboxes[:, 2:]
        # gt contain obj index
        gt_index_i, gt_index_j = gt_xy.long().t()
        # print(gt_index_i[gt_index_i > self.grid_num], gt_index_i[gt_index_i < 0], gt_index_j[gt_index_j > self.grid_num], gt_index_j[gt_index_j < 0])
        

        gt_anchor_ious = torch.stack([anchor_iou(self.anchor_wh[it], gt_wh) for it in range(len(self.anchor_wh))])
        best_ious, best_index = gt_anchor_ious.max(0)
        
        obj_mask[img_id, best_index, gt_index_i, gt_index_j] = 1
        noobj_mask[img_id, best_index, gt_index_i, gt_index_j] = 0

        # for it, now_ious in enumerate(gt_anchor_ious.t()):
        #     noobj_mask[img_id[it], now_ious > self.iou_thresh, gt_index_i[it], gt_index_j[it]] = 0
        
        # obj_mask = 1 - noobj_mask
        tx[img_id, best_index, gt_index_i, gt_index_j] = gt_xy[:, 0] - gt_xy[:, 0].floor()
        ty[img_id, best_index, gt_index_i, gt_index_j] = gt_xy[:, 1] - gt_xy[:, 1].floor()
        tw[img_id, best_index, gt_index_i, gt_index_j] = torch.log(gt_wh[:, 0] / self.anchor_wh[best_index][:, 0] + 1e-16)
        th[img_id, best_index, gt_index_i, gt_index_j] = torch.log(gt_wh[:, 1] / self.anchor_wh[best_index][:, 1] + 1e-16)
        tcls[img_id, best_index, gt_index_i, gt_index_j, label] = 1
        

        tconf = obj_mask.float()

        return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

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

    def compute_loss(self, pred_tensor, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, cuda=True):
        ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        k = self.lbd_cfg['k'] * self.batch_size
        xy_lbd = k * self.lbd_cfg['xy']
        wh_lbd = k * self.lbd_cfg['wh']
        cls_lbd = k * self.lbd_cfg['cls']
        conf_lbd = k * self.lbd_cfg['conf']

        pconf = pred_tensor[..., 0].sigmoid()
        px = pred_tensor[..., 1].sigmoid()
        py = pred_tensor[..., 2].sigmoid()
        pw = pred_tensor[..., 3]
        ph = pred_tensor[..., 4]
        pcls = pred_tensor[..., 5:]# .sigmoid()

        obj_conf_loss = self.obj_lbd * self.bce_loss(pconf[obj_mask], tconf[obj_mask])
        noobj_conf_loss = self.noobj_lbd * self.bce_loss(pconf[noobj_mask], tconf[noobj_mask])
        conf_loss = conf_lbd * (obj_conf_loss + noobj_conf_loss)
        x_loss = xy_lbd * self.mse_loss(px[obj_mask], tx[obj_mask])
        y_loss = xy_lbd * self.mse_loss(py[obj_mask], ty[obj_mask])
        w_loss = wh_lbd * self.mse_loss(pw[obj_mask], tw[obj_mask])
        h_loss = wh_lbd * self.mse_loss(ph[obj_mask], th[obj_mask])

        # cls_loss = cls_lbd * self.bce_loss(pcls[obj_mask], tcls[obj_mask])
        cls_loss = cls_lbd * self.bce_log_loss(pcls[obj_mask], tcls[obj_mask])
        # _, top_cls = tcls[obj_mask].max(-1)
        # print(top_cls)
        # print(obj_mask)
        # print(obj_mask.shape)
        # print(top_cls[obj_mask].shape)
        # cls_loss = cls_lbd * self.ce_loss(pcls[obj_mask], top_cls[obj_mask])

        total_loss = conf_loss + x_loss + y_loss + w_loss + h_loss + cls_loss

        # self.logger.info('stride %d yolo layer\t Loss: %.4f, confidence_loss: %.4f, xy_loss: %.4f, wh_loss: %.4f, classify_loss: %.4f, ' %(self.stride, total_loss.item() / self.batch_size, conf_loss.item(), xy_loss.item(), wh_loss.item(), cls_loss.item()) )
        self.vis.plot('stride %d detect layer : confidence loss'%(self.stride), conf_loss.item())
        self.vis.plot('stride %d detect layer : obj loss'%(self.stride), obj_conf_loss.item())
        self.vis.plot('stride %d detect layer : noobj loss'%(self.stride), noobj_conf_loss.item())
        self.vis.plot('stride %d detect layer : x loss'%(self.stride), x_loss.item())
        self.vis.plot('stride %d detect layer : y loss'%(self.stride), y_loss.item())
        self.vis.plot('stride %d detect layer : w loss'%(self.stride), w_loss.item())
        self.vis.plot('stride %d detect layer : h loss'%(self.stride), h_loss.item())
        self.vis.plot('stride %d detect layer : classify loss'%(self.stride), cls_loss.item())
        # self.vis.plot('total loss', total_loss.item() / self.batch_size)

        return total_loss



    def forward(self, x, target_tensor=None):
        self.batch_size = x.shape[0]
        grid_size = x.shape[2]
        # print(x.shape)
        if grid_size != self.grid_num:
            self.create_grid(grid_size, x.is_cuda)

        x = x.view(self.batch_size, self.anchor_num, -1, self.grid_num, self.grid_num).permute(0, 1, 3, 4, 2).contiguous()
        # print(x.shape)
        

        if self.training and target_tensor is not None:
            obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.encoder(target_tensor)
            loss = self.compute_loss(x, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf)
            return x, loss
        else:
            return self.decoder(x), 0



class YOLO(nn.Module):

    def __init__(self, cfg, logger=None, vis=None, device='cuda:0'):
        super(YOLO, self).__init__()
        self.cfg=cfg
        # self.backbone = self.load_backbone(cfg.backbone_type[0], cfg.backbone_config[0]).to(device)
        # self.fpn = self.load_fpn(cfg.fpn_config).to(device)
        self.device = device
        self.logger=logger
        self.vis=vis
        self.lbd_cfg = cfg.train_config['lbd_map']
        # self.yolo_layer_list = nn.ModuleList(self.init_yolo_layers(cfg.yolo_layer_config_list, logger=logger, vis=vis))
        self.model_list = nn.ModuleList()
        self.model_names = []

        self.build_models()

    
    def init_yolo_layers(self, yolo_layer_config_list, logger=None, vis=None):
        yolo_layer_list = []
        for now_map in yolo_layer_config_list:
            self.model_list.append(YOLOLayer(now_map, self.lbd_cfg, logger=logger, vis=vis))
            # yolo_layer_list.append(YOLOLayer(now_map, self.lbd_cfg, logger=logger, vis=vis).to(self.device))
        # return yolo_layer_list
    
    def build_models(self):
        
        # self.model_list.append(self.load_backbone(self.cfg.backbone_type[0], self.cfg.backbone_config[0]))
        # self.model_list.append(self.load_fpn(self.cfg.fpn_config))
        # self.model_list.append([nn.Sequential(i) for i in self.init_yolo_layers(self.cfg.yolo_layer_config_list, logger=self.logger, vis=self.vis)])
        self.load_backbone(self.cfg.backbone_type[0], self.cfg.backbone_config[0])
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


    def forward(self, x, target_tensor=None, cuda=True):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        pred_list = []
        total_loss = FloatTensor([0])
        
        for it, (nn_name, nn_layer) in enumerate(zip(self.model_names, self.model_list)):
            if nn_name == 'backbone':
                feature_map_list = nn_layer(x)
            elif nn_name == 'fpn':
                pyramid_output_list = nn_layer(feature_map_list)
            elif nn_name[:len('yolo')] == 'yolo':
                layer_id = int(nn_name.split('_')[-1])
                now_scale_pred, layer_loss = nn_layer(pyramid_output_list[layer_id], target_tensor)
                pred_list.append(now_scale_pred)
                total_loss += layer_loss

        # feature_map_list = self.backbone(x)
        # pyramid_output_list = self.fpn(feature_map_list)
        
        # for i in range(len(pyramid_output_list)):
        #     now_scale_pred, layer_loss = self.yolo_layer_list[i](pyramid_output_list[i], target_tensor)
        #     pred_list.append(now_scale_pred)
        #     total_loss += layer_loss
        if self.training and target_tensor is not None:
            return pred_list, total_loss
        else:
            return torch.cat(pred_list, 1), total_loss
            # return torch.cat(pred_list, 1), total_loss

if __name__ == "__main__":
    from torchsummary import summary
    from YOLODatasets import yoloDataset
    from compute_utils import cv_resize, draw_debug_rect
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from Loss import *

    temp_config = Config()
    device = 'cuda:0'
    # backbone = resnet18()
    # fpn = FPN(backbone, temp_config.fpn_config)
    # yolo = YOLO(fpn, temp_config).to(device)
    yolo = YOLO(temp_config)
    # summary(yolo, (3, 416, 416))
    print(yolo)

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = yoloDataset(list_file='datasets/2007_train.txt',train=False, transform = transform, test_mode=True, device='cuda:0')
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
    train_iter = iter(train_loader)

    for i in range(200):
        # temp = next(train_iter)
        # print(temp)
        img, label_bboxes = next(train_iter)
        img, label_bboxes = img.to(device), label_bboxes.to(device)
        # print(img.is_cuda)
        pred = yolo(img)
        now_loss = compute_loss(pred, label_bboxes, yolo.yolo_layer_list)
        print(now_loss)
        # cls_id, gt_xy, gt_wh, index_list = yolo.yolo_layer_list[-1].encoder(label_bboxes)
        # print(cls_id, gt_xy, gt_wh, index_list)
        # print(img.shape, label_bboxes.shape)
        # print(label_bboxes)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img = un_normal_trans(img[1].squeeze(0))
        temp_label_bboxes = label_bboxes[label_bboxes[:, 0] == 1.0]
        # print(temp_label_bboxes, temp_label_bboxes.shape)
        img = draw_debug_rect(img.permute(1, 2 ,0), temp_label_bboxes[..., 2:], temp_label_bboxes[..., 1])
        cv2.imshow('img', img)
        
        if cv2.waitKey(12000)&0xFF == ord('q'):
            break

