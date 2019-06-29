# encoding:utf-8
import numpy as np, torch 
import torch.nn as nn



mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
bce_log_loss = nn.BCEWithLogitsLoss()

def compute_loss(self, pred_tensor, target, cuda=True):
    
    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    k = self.lbd_cfg['k'] * self.batch_size
    xy_lbd = k * self.lbd_cfg['xy']
    wh_lbd = k * self.lbd_cfg['wh']
    cls_lbd = k * self.lbd_cfg['cls']
    conf_lbd = k * self.lbd_cfg['conf']

    xy_loss, wh_loss, cls_loss, conf_loss = FloatTensor([0]), FloatTensor([0]), FloatTensor([0]), FloatTensor([0])
    txy, twh, tcls, obj_indexs = self.encoder(target)
    img_id, best_index, gx_index, gy_index = obj_indexs

    now_device = txy.device
    pred_tensor = pred_tensor.to(now_device)
    pconf = torch.sigmoid(pred_tensor[..., 0])
    tconf = torch.zeros_like(pconf)
    
    if len(img_id):
        obj_pred = pred_tensor[obj_indexs].to(now_device)
        tconf[obj_indexs] = 1. 
        pxy = obj_pred[..., 1:3].sigmoid().to(now_device)

        pwh = obj_pred[..., 3:5].to(now_device)
        xy_loss += xy_lbd * mse_loss(pxy, txy)
        wh_loss += wh_lbd * mse_loss(pwh, twh)
        cls_loss += cls_lbd * ce_loss(obj_pred[..., 5:], tcls)
    
    conf_loss += conf_lbd * bce_loss(pconf, tconf)
    all_loss = conf_loss + xy_loss + wh_loss + cls_loss

    # self.logger.info('stride %d yolo layer\t Loss: %.4f, confidence_loss: %.4f, xy_loss: %.4f, wh_loss: %.4f, classify_loss: %.4f, ' %(self.stride, total_loss.item() / self.batch_size, conf_loss.item(), xy_loss.item(), wh_loss.item(), cls_loss.item()) )
    self.vis.plot('stride %d detect layer : confidence loss'%(self.stride), conf_loss.item())
    self.vis.plot('stride %d detect layer : xy loss'%(self.stride), xy_loss.item())
    self.vis.plot('stride %d detect layer : wh loss'%(self.stride), wh_loss.item())
    self.vis.plot('stride %d detect layer : classify loss'%(self.stride), cls_loss.item())
    self.vis.plot('stride %d total loss'%(self.stride), all_loss.item())

    return all_loss