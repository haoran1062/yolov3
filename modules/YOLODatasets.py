# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
# from utils import *
ia.seed(random.randint(1, 10000))


class yoloDataset(Dataset):
    image_size = 416
    def __init__(self, list_file, train, transform, device, little_train=False, with_file_path=False, C = 20, test_mode=False):
        print('data init')
        
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.resize = 416
        self.C = C
        self.device = device
        self._test = test_mode
        self.with_file_path = with_file_path
        self.img_augsometimes = lambda aug: iaa.Sometimes(0.25, aug)
        self.bbox_augsometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.augmentation = iaa.Sequential(
            [
                # augment without change bboxes 
                self.img_augsometimes(
                    iaa.SomeOf((1, 3), [
                        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
                        iaa.Sharpen((0.1, .8)),       # sharpen the image
                        # iaa.GaussianBlur(sigma=(2., 3.5)),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(2., 3.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.BilateralBlur(d=(7, 12), sigma_color=(10, 250), sigma_space=(10, 250)),
                            iaa.MedianBlur(k=(3, 7)),
                        ]),
                        

                        iaa.AddElementwise((-50, 50)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                        iaa.JpegCompression(compression=(80, 95)),

                        iaa.Multiply((0.5, 1.5)),
                        iaa.MultiplyElementwise((0.5, 1.5)),
                        iaa.ReplaceElementwise(0.05, [0, 255]),
                        # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                        #                 children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        iaa.OneOf([
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(1, iaa.Add((-10, 50)))),
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        ]),

                    ], random_order=True)
                ),

                iaa.Fliplr(.5),
                iaa.Flipud(.125),
                # # augment changing bboxes 
                self.bbox_augsometimes(
                    iaa.Affine(
                        # translate_px={"x": 40, "y": 60},
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-5, 5),
                    )
                )
            ],
            random_order=True
        )

        # torch.manual_seed(23)
        with open(list_file) as f:
            lines  = f.readlines()
        
        if little_train:
            lines = lines[:little_train]

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            
        self.num_samples = len(self.fnames)
    
    def get_boxes_labels(self, in_path):
        bboxes = []
        with open(in_path.replace('JPEGImages', 'labels').replace('jpg', 'txt'), 'r') as f:
            for line in f:
                ll = line.strip().split(' ')
                c = int(ll[0])
                x = float(ll[1])
                y = float(ll[2])
                w = float(ll[3])
                h = float(ll[4])
                bboxes.append([0, c, x, y, w, h])
        return torch.Tensor(bboxes)

    
    def convertXYWH2XYXY(self, bboxes, im_size=(image_size, image_size)):
        '''
            input: tensor [Cx, Cy, w, h] normalized bboxes
            output: python list [x1, y1, x2, y2] abs bboxes

        '''
        t_boxes = []
        for i in range(bboxes.size()[0]):
            [x, y, w, h] = bboxes[i].tolist()
            x1 = int( (x - 0.5*w) * im_size[0] )
            y1 = int( (y - 0.5*h) * im_size[1] )
            x2 = int( (x + 0.5*w) * im_size[0] )
            y2 = int( (y + 0.5*h) * im_size[1] )
            t_boxes.append([x1, y1, x2, y2])
        return t_boxes
    
    def convert2augbbox(self, img, bboxes):
        '''
            input: tensor list [x1, y1, x2, y2] abs bboxes
            output: python imgaug bboxes object

        '''
        l = []
        for [x1, y1, x2, y2] in bboxes:
            l.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return ia.BoundingBoxesOnImage(l, shape=img.shape)
    
    def convertaugbbox2X1Y1X2Y2(self, bboxes):
        t_boxes = []
        for i in range(len(bboxes.bounding_boxes)):
            t_boxes.append([bboxes.bounding_boxes[i].x1, bboxes.bounding_boxes[i].y1, bboxes.bounding_boxes[i].x2, bboxes.bounding_boxes[i].y2])
        return t_boxes

    def convertAugbbox2XYWH(self, bboxes, im_size=(image_size, image_size)):
        t_boxes = []
        bboxes = self.convertaugbbox2X1Y1X2Y2(bboxes)
        for [x1, y1, x2, y2] in bboxes:
            cx = (x2 + x1)/2. - 1
            cy = (y2 + y1)/2. - 1
            w = x2 - x1
            h = y2 - y1
            cx /= im_size[0]
            cy /= im_size[1]
            w /= im_size[0]
            h /= im_size[1]
            t_boxes.append([cx, cy, w, h])
        return t_boxes

    @staticmethod
    def collate_fn(batch):
        img, label = list(zip(*batch))  # transposed
        # print('fuck')
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        # print(label.shape)    
        return torch.stack(img, 0), torch.cat(label, 0) 
        
        # return img, label

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        if self._test:
            print(fname)
        img = cv2.imread(fname)
        label_boxes = self.get_boxes_labels(fname)
        # print(label_boxes.shape)
        if self.train:
            # TODO
            # add data augument
            # print('before: ')
            # print(boxes)
            boxes = self.convert2augbbox(cv2.resize(img, (self.resize, self.resize)), self.convertXYWH2XYXY(label_boxes[..., 2:]))
            seq_det = self.augmentation.to_deterministic()
            aug_img = seq_det.augment_images([img])[0]
            boxes = seq_det.augment_bounding_boxes([boxes])[0].remove_out_of_image().clip_out_of_image()
            
            # img = boxes.draw_on_image(cv2.resize(img, (self.resize, self.resize)), thickness=2, color=[0, 0, 255])
            
            boxes = self.convertAugbbox2XYWH(boxes)
            boxes = torch.Tensor(boxes)
            if label_boxes.shape[0] == boxes.shape[0]:
                label_boxes[..., 2:] = boxes[:]
                img = aug_img
            else:
                # print('bbox shfit failed use origin label')
                pass

        img = self.transform(img)
        # print(fname)
        # print(type(img))
        # if self.with_file_path:
        #     print(fname)
        # print(label_boxes.shape)
        # print(img.shape, label_boxes.shape)
        return img, label_boxes
        # return img.to(self.device), target.to(self.device)

    def __len__(self):
        return self.num_samples

    

    
if __name__ == "__main__":

    from compute_utils import cv_resize, draw_debug_rect

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = yoloDataset(list_file='datasets/2007_train.txt',train=True,transform = transform, test_mode=True, device='cuda:0')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
    train_iter = iter(train_loader)
    # print(next(train_iter))
    for i in range(200):
        # temp = next(train_iter)
        # print(temp)
        img, label_bboxes = next(train_iter)
        print(img.shape, label_bboxes.shape)
        # print(label_bboxes)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img = un_normal_trans(img[1].squeeze(0))
        temp_label_bboxes = label_bboxes[label_bboxes[:, 0] == 1.0]
        print(temp_label_bboxes, temp_label_bboxes.shape)
        # temp_label_bboxes = temp_label_bboxes[]
        img = draw_debug_rect(img.permute(1, 2 ,0), temp_label_bboxes[..., 2:], temp_label_bboxes[..., 1])
        # img = draw_classify_confidence_map(img, target, S, Color)
        cv2.imshow('img', img)
        # print(mask_label[0, 10:100, 10:100])
        
        if cv2.waitKey(12000)&0xFF == ord('q'):
            break

