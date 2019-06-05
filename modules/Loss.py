# encoding:utf-8
import numpy as np, torch 
import torch.nn as nn


MSELoss = nn.MSELoss()
CELoss = nn.CrossEntropyLoss()
BCELoss = nn.BCEWithLogitsLoss()


def compute_loss(pred, target, yolo_layer_list, cfg, logger, vis, device='cuda:0'):
    
    batch_size = len(pred[0])
    k = cfg['k'] * batch_size  # loss gain
    xy_lbd = k * cfg['xy']
    wh_lbd = k * cfg['wh']
    cls_lbd = k * cfg['cls']
    conf_lbd = k * cfg['conf']

    cls_loss, xy_loss, wh_loss, confidence_loss = torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device)
    
    for it in range(len(yolo_layer_list)):
        now_pred = pred[it]

        pred_confidence = now_pred[..., 0].sigmoid()

        gt_confidence = torch.zeros_like(now_pred[..., 0])

        gt_cls, gt_xy, gt_wh, index_list = yolo_layer_list[it].encoder(target)

        img_id, best_index, grid_x, grid_y = index_list
        
        now_pred = now_pred[img_id, best_index, grid_x, grid_y]

        
        gt_confidence[img_id, best_index, grid_x, grid_y] = 1

        
        pred_xy = now_pred[..., 1:3].sigmoid()
        pred_wh = now_pred[..., 3:5]
        pred_cls = now_pred[..., 5:] # .sigmoid()# (dim=-1)

        confidence_loss += conf_lbd * BCELoss(pred_confidence, gt_confidence)
        if len(img_id):
            xy_loss += xy_lbd * MSELoss(pred_xy, gt_xy)
            wh_loss += wh_lbd * MSELoss(pred_wh, gt_wh)
            cls_loss += cls_lbd * CELoss(pred_cls, gt_cls)

    total_loss = confidence_loss + xy_loss + wh_loss + cls_loss

    logger.info('total Loss: %.4f, confidence_loss: %.4f, xy_loss: %.4f, wh_loss: %.4f, classify_loss: %.4f, ' %(total_loss.item() / batch_size, confidence_loss.item(), xy_loss.item(), wh_loss.item(), cls_loss.item()) )
    vis.plot('confidence loss', confidence_loss.item())
    vis.plot('xy loss', xy_loss.item())


    vis.plot('wh loss', wh_loss.item())
    vis.plot('classify loss', cls_loss.item())
    vis.plot('total loss', total_loss.item() / batch_size)
    return total_loss



                