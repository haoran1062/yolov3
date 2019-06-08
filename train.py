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
from modules.configs.resnet50_yolo_style_fpn_yolov3 import Config
from utils.train_utils import *
from modules.compute_utils import *
import torch.optim as optim

parser = argparse.ArgumentParser(
    description='YOLO V1 Training params')
parser.add_argument('--config', default='configs/resnet50_yolo_style_fpn_yolov3.py')
args = parser.parse_args()

config_map = Config()
train_config = config_map.train_config
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# learning_rate = init_lr(config_map)
if not os.path.exists(train_config['base_save_path']):
    os.makedirs(train_config['base_save_path'])

logger = create_logger(train_config['base_save_path'], train_config['log_name'])

my_vis = Visual(train_config['base_save_path'], log_to_file=train_config['vis_log_path'])

# yolo = init_model(config_map)
yolo = YOLO(config_map, logger=logger, vis=my_vis)
yolo_p = nn.DataParallel(yolo.to(device), device_ids=train_config['gpu_ids'])
if train_config['resume_from_path']:
    yolo_p.load_state_dict(torch.load(train_config['resume_from_path']))

print(yolo_p)
# summary(yolo_p, (3, 416, 416), batch_size=train_config['batch_size'])
# exit()
lr0 = train_config['lbd_map']['lr0']
lrf = train_config['lbd_map']['lrf']
momentum = train_config['lbd_map']['momentum']
weight_decay = train_config['lbd_map']['weight_decay']
resume_epoch = train_config['resume_epoch']
epochs = train_config['epoch_num']


learning_rate = 0.
optimizer = optim.SGD(yolo_p.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # , weight_decay=5e-4)
# optimizer = optim.Adam(yolo_p.parameters())

# lf = lambda x: 1 - 10 ** (lrf * (1 - x / epochs))  # inverse exp ramp
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch= resume_epoch - 1)



# yolo_p.load_state_dict(torch.load('densenet_sgd_S7_yolo.pth'))

yolo_p.train()
print(yolo_p)
# exit()
transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


train_dataset = yoloDataset(list_file=train_config['train_txt_path'], train=False, transform = transform, little_train=8, test_mode=False, device='cuda:0')
train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['worker_num'], collate_fn=train_dataset.collate_fn)

test_dataset = yoloDataset(list_file=train_config['test_txt_path'], train=False, transform = transform, test_mode=False, device='cuda:0')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=train_config['worker_num'], collate_fn=train_dataset.collate_fn)
test_iter = iter(test_loader)

data_len = int(len(test_dataset) / train_config['batch_size'])
logger.info('the dataset has %d images' % (len(train_dataset)))
logger.info('the batch_size is %d' % (train_config['batch_size']))

num_iter = 0
best_mAP = 0.0
train_len = len(train_dataset)
train_iter = train_config['resume_epoch'] * len(train_loader)
last_little_mAP = 0.0

# my_vis.img('label colors', get_class_color_img())
n_burnin = min(round(len(train_dataset) / 5 + 1), 1000) 
n_burnin = 100

name_list = get_names(config_map.name_path)
make_color_list(config_map.class_num)


for epoch in range(train_config['resume_epoch'], train_config['epoch_num']):
    
    # scheduler.step()

    yolo_p.train()

    logger.info('\n\nStarting epoch %d / %d' % (epoch + 1, train_config['epoch_num']))
    logger.info('Learning Rate for this epoch: {}'.format(optimizer.param_groups[0]['lr']))

    epoch_start_time = time.clock()
    
    total_loss = 0.
    avg_loss = 0.
    
    for i,(img, label_bboxes) in enumerate(train_loader):
        # print('mask label : ', mask_label.shape, mask_label.dtype)
        # if epoch == 0 and i <= n_burnin:
        #     lr = lr0 * (i / n_burnin) ** 4
        #     for x in optimizer.param_groups:
        #         x['lr'] = lr

        it_st_time = time.clock()
        train_iter += 1
        learning_rate = learning_rate_policy(train_iter, epoch, learning_rate, train_config['lr_adjust_map'], train_config['stop_down_iter'], train_config['add_lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        my_vis.plot('now learning rate', optimizer.param_groups[0]['lr'])
        img, label_bboxes = img.to(device), label_bboxes.to(device)

        pred, now_loss = yolo_p(img, label_bboxes)
        # now_loss = compute_loss(pred, label_bboxes, yolo_p.module.yolo_layer_list, train_config['lbd_map'], logger, my_vis)
        my_vis.plot('total loss', now_loss.item() / img.shape[0])
        total_loss += now_loss.data.item()
        # print(p_mask.shape, mask_label.shape)
        # exit()
        if my_vis and i % train_config['show_img_iter_during_train'] == 0:
            yolo_p.eval()
            # img, label_bboxes = next(test_iter)
            pred, _ = yolo_p(img[0].unsqueeze(0))
            detect_tensor = non_max_suppression(pred.to('cpu'), conf_thres=0.5, nms_thres=0.1)
            detect_tensor = detect_tensor[0]
            show_img = unorm(img[0])
            if detect_tensor is not None:
                # print(show_img.shape)
                show_img = draw_debug_rect(show_img.permute(1, 2 ,0), detect_tensor[..., 1:5], detect_tensor[..., -1], detect_tensor[..., -2], name_list)
            my_vis.img('detect bboxes show', show_img)

            yolo_p.train()
            

        optimizer.zero_grad()
        now_loss.backward()
        optimizer.step()
        it_ed_time = time.clock()
        it_cost_time = it_ed_time - it_st_time
        if (i+1) % 5 == 0:
            avg_loss = total_loss / (i+1)
            logger.info('Epoch [%d/%d], Iter [%d/%d] expect end in %.2f min. Loss: %.4f, average_loss: %.4f' %(epoch+1, train_config['epoch_num'], i+1, len(train_loader), it_cost_time * (len(train_loader) - i+1) // 60 , now_loss.item() / train_config['batch_size'], total_loss / (i+1) / train_config['batch_size']))
            num_iter += 1
        
    epoch_end_time = time.clock()
    epoch_cost_time = epoch_end_time - epoch_start_time
    now_epoch_train_loss = total_loss / (i+1)
    my_vis.plot('train loss', now_epoch_train_loss / train_config['batch_size'])
    logger.info('Epoch {} / {} finished, cost time {:.2f} min. expect {} min finish train.'.format(epoch, train_config['epoch_num'], epoch_cost_time / 60, (epoch_cost_time / 60) * (train_config['epoch_num'] - epoch + 1)))

    #validation
    yolo_p.eval()
    now_little_mAP = 0.0
    test_mAP = 0.0

    
    # run full mAP cost much time, so when little mAP > thresh then run full test data's mAP 

    # my_vis.plot('little mAP', now_little_mAP)
    # my_vis.plot('mAP', test_mAP)
    # last_little_mAP = now_little_mAP
    
    # if test_mAP > best_mAP:
    #     best_mAP = test_mAP
    #     logger.info('get best test mAP %.5f' % best_mAP)
    #     torch.save(yolo_p.state_dict(),'%s/%s_S%d_best.pth'%(config_map['base_save_path'], config_map['backbone'], config_map['S']))
   
    torch.save(yolo_p.state_dict(),'%s/%s_last.pth'%(train_config['base_save_path'], config_map.backbone_type[0]))