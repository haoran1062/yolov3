# encoding:utf-8
import numpy as np, torch 
import torch.nn as nn

from modules.compute_utils import *

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
bce_log_loss = nn.BCEWithLogitsLoss()

def encoder(yolo_layer, target, cuda=True):
    '''
        input:
            target: [image_id, cls_id, Cx, Cy, w, h]

        output:

    '''

    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # target = target.to(yolo_layer.anchor_wh.device)
    img_id, label = target[:, :2].long().t()
    # convert normilzed 0~1 coord to grid normilzed coord
    target_bboxes = target[:, 2:] * yolo_layer.grid_num
    gt_xy = target_bboxes[:, :2]
    gt_wh = target_bboxes[:, 2:]
    # gt contain obj index
    gt_index_j, gt_index_i = gt_xy.long().t()

    gt_anchor_ious = torch.stack([anchor_iou(yolo_layer.anchor_wh[it], gt_wh) for it in range(len(yolo_layer.anchor_wh))])
    best_ious, best_index = gt_anchor_ious.max(0)

    obj_index = (img_id, best_index, gt_index_i, gt_index_j)

    txy = gt_xy - gt_xy.floor()
    # print(gt_wh, self.anchor_wh)
    twh = torch.log(gt_wh / yolo_layer.anchor_wh[best_index])
    tcls = label

    return txy, twh, tcls, obj_index

def compute_loss(yolo_layer, pred_tensor, target, cuda=True, vis=None, logger=None):
    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # now_device = pred_tensor.device
    # now_device = target.device
    # target = target.to(now_device)

    k = yolo_layer.lbd_cfg['k'] * pred_tensor.shape[0]
    xy_lbd = k * yolo_layer.lbd_cfg['xy']
    wh_lbd = k * yolo_layer.lbd_cfg['wh']
    cls_lbd = k * yolo_layer.lbd_cfg['cls']
    conf_lbd = k * yolo_layer.lbd_cfg['conf']

    xy_loss, wh_loss, cls_loss, conf_loss = FloatTensor([0]), FloatTensor([0]), FloatTensor([0]), FloatTensor([0])
    txy, twh, tcls, obj_indexs = encoder(yolo_layer, target)
    img_id, best_index, gx_index, gy_index = obj_indexs
    # print('target device: ', target.device, ' pred device: ', pred_tensor.device)
    # now_device = txy.device
    # pred_tensor = pred_tensor.to(now_device)
    pconf = torch.sigmoid(pred_tensor[..., 0])
    tconf = torch.zeros_like(pconf)
    
    if len(img_id):
        obj_pred = pred_tensor[obj_indexs]# .to(now_device)
        # print(obj_pred.shape, obj_pred.device)
        # print(obj_pred.to('cuda:0'))
        tconf[obj_indexs] = 1. 
        pxy = obj_pred[..., 1:3].sigmoid() #.to(now_device)

        pwh = obj_pred[..., 3:5] #.to(now_device)
        # print('*'*30)
        # print('now device: ', now_device)
        # print(pxy.device, txy.device)
        # print(txy)
        # print(pxy)
        # print(pxy.shape, txy.shape)
        # print('*'*30)
        xy_loss += xy_lbd * mse_loss(pxy, txy)
        wh_loss += wh_lbd * mse_loss(pwh, twh)
        cls_loss += cls_lbd * ce_loss(obj_pred[..., 5:], tcls)
    
    conf_loss += conf_lbd * bce_loss(pconf, tconf)
    all_loss = conf_loss + xy_loss + wh_loss + cls_loss

    # self.logger.info('stride %d yolo layer\t Loss: %.4f, confidence_loss: %.4f, xy_loss: %.4f, wh_loss: %.4f, classify_loss: %.4f, ' %(self.stride, total_loss.item() / self.batch_size, conf_loss.item(), xy_loss.item(), wh_loss.item(), cls_loss.item()) )
    vis.plot('stride %d detect layer : confidence loss'%(yolo_layer.stride), conf_loss.item())
    vis.plot('stride %d detect layer : xy loss'%(yolo_layer.stride), xy_loss.item())
    vis.plot('stride %d detect layer : wh loss'%(yolo_layer.stride), wh_loss.item())
    vis.plot('stride %d detect layer : classify loss'%(yolo_layer.stride), cls_loss.item())
    vis.plot('stride %d total loss'%(yolo_layer.stride), all_loss.item())

    return all_loss
