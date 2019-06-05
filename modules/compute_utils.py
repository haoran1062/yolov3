# encoding:utf-8
import os, numpy as np, random, cv2, logging, json
import torch

from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

def anchor_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou

def cv_resize(img, resize=416):
    return cv2.resize(img, (resize, resize))

def bbox_un_norm(bboxes, im_size=(416, 416)):
    t_boxes = []
    for i in range(len(bboxes)):
        [x, y, w, h] = bboxes[i]
        x1 = int( (x - 0.5*w) * im_size[0] )
        y1 = int( (y - 0.5*h) * im_size[1] )
        x2 = int( (x + 0.5*w) * im_size[0] )
        y2 = int( (y + 0.5*h) * im_size[1] )
        t_boxes.append([x1, y1, x2, y2])
    return t_boxes

def clip_bbox(bbox, img_size=416):
    for i in range(len(bbox)):
        bbox[i] = min(bbox[i], 416)
        bbox[i] = max(bbox[i], 0)
    return bbox

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

def draw_debug_rect(img, bboxes, clss, confs=None, color=(0, 255, 0), show_time=10000):

    if isinstance(img, torch.Tensor):
        img = img.mul(255).byte()
        img = img.cpu().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.tolist()

    if bboxes[0][2] < 1:
        bboxes = bbox_un_norm(bboxes)
        print('un_norm bbox')
    # print(len(bboxes))
    for i, box in enumerate(bboxes):
        box = clip_bbox(box)
        # print(box)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cls_i = int(clss[i].item())
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color,thickness=2)
        # cv2.putText(img, '%s %.2f'%(VOC_CLASSES[cls_i], confs[i]), (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Color[cls_i], 1, 10)
    # cv2.imshow('debug draw bboxes', img)
    # cv2.waitKey(show_time)
    return img

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # print(prediction[0])
    prediction[..., 1:5] = xywh2xyxy(prediction[..., 1:5])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 0], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    # print(output[0].shape)
    return output


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))