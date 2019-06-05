# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
from utils import *
ia.seed(random.randint(1, 10000))


class yoloDataset(data.Dataset):
    image_size = 448
    def __init__(self,list_file,train,transform, device, little_train=False, with_file_path=False, with_mask=False, S=7, B = 2, C = 20, test_mode=False):
        print('data init')
        
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.resize = 448
        self.with_mask = with_mask
        self.S = S
        self.B = B
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

                # iaa.Fliplr(.5),
                # iaa.Flipud(.125),
                # # augment changing bboxes 
                # self.bbox_augsometimes(
                #     iaa.Affine(
                #         # translate_px={"x": 40, "y": 60},
                #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #         rotate=(-5, 5),
                #     )
                # )
            ],
            random_order=True
        )

        # torch.manual_seed(23)
        with open(list_file) as f:
            lines  = f.readlines()
        
        if little_train:
            lines = lines[:64*8]

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            
        self.num_samples = len(self.fnames)
    
    def get_boxes_labels(self, in_path):
        bboxes = []
        labels = []
        with open(in_path.replace('JPEGImages', 'labels').replace('jpg', 'txt'), 'r') as f:
            for line in f:
                ll = line.strip().split(' ')
                labels.append(int(ll[0]))
                x = float(ll[1])
                y = float(ll[2])
                w = float(ll[3])
                h = float(ll[4])
                bboxes.append([x, y, w, h])
        return torch.Tensor(bboxes), torch.LongTensor(labels)
    
    def get_bit_mask(self, mask_img, cls_n=20):
        '''
            mask_img : cv2 numpy mat BGR
            return [NxHxW] numpy
        '''
        b = mask_img
        
        h, w = b.shape
        bit_mask = np.zeros((cls_n, h, w), np.uint8)
        for i in range(cls_n):
            bit_mask[i, :, :] = np.where(b==i, 1, 0)

        return bit_mask
    
    def load_mask_label(self, file_path, resize=448):
        """
            Load label image as 1 x height x width integer array of label indices.
            The leading singleton dimension is required by the loss.
        """
        im = Image.open(file_path.replace('JPEGImages', 'SegmentationClass').replace('jpg', 'png'))
        im = im.resize((resize, resize))
        label = np.array(im, dtype=np.uint8)
        label = np.where(label==255, 0, label)
        # label = label[np.newaxis, ...]
        return torch.from_numpy(self.get_bit_mask(label)).float() #.to(self.device)# .byte()
    


    
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

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        if self._test:
            print(fname)
        img = cv2.imread(fname)
        boxes, labels = self.get_boxes_labels(fname)
        mask_label = self.load_mask_label(fname)
        


        if self.train:
            # TODO
            # add data augument
            # print('before: ')
            # print(boxes)
            boxes = self.convert2augbbox(cv2.resize(img, (self.resize, self.resize)), self.convertXYWH2XYXY(boxes))
            seq_det = self.augmentation.to_deterministic()
            img = seq_det.augment_images([img])[0]
            boxes = seq_det.augment_bounding_boxes([boxes])[0].remove_out_of_image().clip_out_of_image()
            
            # img = boxes.draw_on_image(cv2.resize(img, (self.resize, self.resize)), thickness=2, color=[0, 0, 255])
            
            boxes = self.convertAugbbox2XYWH(boxes)
            boxes = torch.Tensor(boxes)
            labels = labels[:boxes.size()[0]]
            # print('after: ')
            # print(boxes)

            

            # print(img.shape)
            # pass
        
        target = self.encoder(boxes,labels)# 7x7x30
        
        img = self.transform(img)
        # print(fname)
        # print(type(img))
        if self.with_file_path:
            if self.with_mask:
                return img, target, mask_label, fname
            return img, target, fname
        if self.with_mask:
            # print('ok')
            # print('mask label, ', mask_label.shape, mask_label.dtype)
            return img, target, mask_label
        return img, target
        # return img.to(self.device), target.to(self.device)

    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[cx,cy,w,h],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        target = torch.zeros((self.S, self.S, self.B*5 + self.C))
        cell_size = 1./self.S
        # wh = boxes[:,2:]-boxes[:,:2]
        # cxcy = (boxes[:,2:]+boxes[:,:2])/2
        if boxes.size()[0] == 0:
            return target
        try:
            wh = boxes[:, 2:]
            cxcy = boxes[:, :2]
        except:
            print(boxes)
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]), :] = 0
            target[int(ij[1]),int(ij[0]),:self.B] = 1
            target[int(ij[1]),int(ij[0]),self.B*5 + int(labels[i])] = 1
            xy = ij*cell_size 
            delta_xy = (cxcy_sample -xy)/cell_size
            for b in range(self.B):
                target[int(ij[1]),int(ij[0]),self.B+b*4 : self.B+b*4+2] = delta_xy
                target[int(ij[1]),int(ij[0]),self.B+b*4+2 : self.B+b*4+4] = wh[i]
                
            
        return target

def draw_classify_confidence_map(img, cls_tensor, S, color_map):
    h, w, c = img.shape 
    for i in range(S):
        for j in range(S):
            cv2.line(img, (0, j * h/S), (w, j * h/S), (0, 0, 0), 3)
            cv2.line(img, (i * w/S, 0), (i * w/S, h), (0, 0, 0), 3)
    return img
    
if __name__ == "__main__":

    from utils import decoder, draw_debug_rect, cv_resize, draw_classify_confidence_map, Color

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    S = 7
    train_dataset = yoloDataset(list_file='datasets/2012_seg.txt',train=False,transform = transform, test_mode=True, with_mask=True, S=S, device='cuda:0')
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    # print(next(train_iter))
    for i in range(200):
        img, target, mask_label = next(train_iter)
        # mask_img = mask_img.squeeze(0).cpu().numpy()
        # print('mask shape is :', mask_img.shape)
        # print(img.shape, target.shape)
        boxes, clss, confs = decoder(target, grid_num=S, gt=True)
        # print(boxes, clss, confs)
        print('~'*50 + '\n\n\n')

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img = un_normal_trans(img.squeeze(0))
        img = draw_debug_rect(img.permute(1, 2 ,0), boxes, clss, confs)
        img = draw_classify_confidence_map(img, target, S, Color)
        cv2.imshow('img', img)
        # print(mask_label[0, 10:100, 10:100])
        
        mask_img = mask_label_2_img(mask_label)
        print(mask_label.shape, mask_label.dtype)
        pred_mask = pred_mask_label_2_img(mask_label[0])
        cv2.imshow('mask', mask_img)
        cv2.imshow('pred', pred_mask)
        if cv2.waitKey(12000)&0xFF == ord('q'):
            break

    # for i in range(7):
    #     for j in range(7):
    #         print(target[:, i:i+1, j:j+1, :])
