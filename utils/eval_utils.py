# encoding:utf-8
import os, numpy as np, random, cv2, logging, json
import torch

from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou

def simplify_eval_output(pred, conf_thresh=0.0023, nms_thresh=0.45, min_wh=2, device='cuda:0'):
    '''
        pred: list [every_layer_pred_tensor ... ] 
                every_layer_pred_tensor [batch_size, -1, [confidence, Cx, Cy, w, h, classify...]]
    
    '''
    # print(torch.__vision__)
    # output_tensor = torch.FloatTensor([]).to(device)
    output = [None] * len(pred)
    for it, now_pred_tensor in enumerate(pred):
        cls_conf, cls_pred = now_pred_tensor[:, 5:].max(1)
        confid_tensor = now_pred_tensor[:, 0]
        confid_tensor *= cls_conf
        now_pred_tensor[:, 0] *= cls_conf

        filter_index = (now_pred_tensor[:, 0] > conf_thresh) & (now_pred_tensor[:, 3:5] > min_wh).all(1) # & torch.isfinite(now_pred_tensor).all(1)
        now_pred_tensor = now_pred_tensor[filter_index]
        if len(now_pred_tensor) == 0:
            continue
        
        cls_conf = cls_conf[filter_index]
        cls_pred = cls_pred[filter_index].unsqueeze(1).float()

        now_pred_tensor[:, 1:5] = xywh2xyxy(now_pred_tensor[:, 1:5])

        now_pred_tensor = torch.cat((now_pred_tensor[:, :5], cls_conf.unsqueeze(1), cls_pred), 1)

        # Get detections sorted by decreasing confidence scores
        now_pred_tensor = now_pred_tensor[(-now_pred_tensor[:, 0]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in now_pred_tensor[:, -1].unique():
            dc = now_pred_tensor[now_pred_tensor[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thresh]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thresh]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thresh  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[it] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output
