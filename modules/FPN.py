# encoding:utf-8
import torch, numpy as np
import torch.nn as nn

import torch.nn.functional as F
from modules.backbones.OriginResNet import resnet50, resnet18
from modules.ConvModules import ConvModule

class FPN(nn.Module):

    def __init__(self, cfg, device='cuda:0'):
        super(FPN, self).__init__()
        self.cls_num = cfg['classes_num']
        self.in_channels = cfg['in_channels']
        self.fpn_in_channels = self.get_fpn_input_channel_list(cfg['single_fpn_in_channels'])
        self.last_out_channel = (5 + self.cls_num) * cfg['anchor_num']
        self.fpn_out_channels = cfg['single_fpn_out_channels'] + [self.last_out_channel]
        self.fpn_kernel_sizes = cfg['kernel_sizes']
        self.middle_channel = cfg['middle_channel']
        self.upsampling_scale = cfg['upsample_scale']
        self.upsampling_mode = cfg['upsample_mode']
        self.device = device

        self.align_conv_layer_list = nn.ModuleList()
        self.fpn_layer_list = nn.ModuleList()

        for it, in_channel in enumerate(self.in_channels):
            self.align_conv_layer_list.append(ConvModule(in_channel, in_channel // 2, kernel_size=1, padding=0, bias=False))
            self.fpn_layer_list.append(self.build_fpn_layer(self.fpn_in_channels[it], self.fpn_out_channels, self.fpn_kernel_sizes))
    # 

    def get_fpn_input_channel_list(self, fpn_in_channels):
        channel_list = []
        channel_list.append([self.in_channels[-1] // 2] + fpn_in_channels)
        for it in range(len(self.in_channels)-1, 0, -1):
            channel_list.append([self.in_channels[it-1]//2 + self.in_channels[it]//2] + fpn_in_channels)
        return channel_list


    def upsampling(self, x, scale=2, mode='nearest'):
        return F.interpolate(x, scale_factor=scale, mode=mode)

    def build_fpn_layer(self, in_channels, out_channels, kernel_sizes):
        fpn_layer = nn.Sequential()
        # print(in_channels)
        # print(out_channels)
        # print(kernel_sizes)
        for it in range(len(in_channels)):
            # print(in_channels[it], out_channels[it], kernel_sizes[it])
            fpn_layer.add_module('Conv_%d'%(it), ConvModule(in_channels[it], out_channels[it], kernel_size=kernel_sizes[it], padding=(kernel_sizes[it]-1) // 2, bias=False))


        return fpn_layer


    def forward(self, x):
        feature_map_list = x
        aligned_pyramid_list = [self.align_conv_layer_list[it](feature_map) for it, feature_map in enumerate(feature_map_list)]
        pyramid_list = [aligned_pyramid_list[-1]]
        for it in range(len(aligned_pyramid_list)-1, 0, -1):
            # print(aligned_pyramid_list[it-1].shape, aligned_pyramid_list[it].shape)
            pyramid_list.append(torch.cat([aligned_pyramid_list[it-1], self.upsampling(aligned_pyramid_list[it], self.upsampling_scale, self.upsampling_mode)], 1))
            # print(pyramid_list[-1].shape)
        # for i in pyramid_list:
        #     print('concat shape ', i.shape)
        for it, fpn_layer in enumerate(self.fpn_layer_list):
            pyramid_list[it] = fpn_layer(pyramid_list[it])
            

        

        return pyramid_list

if __name__ == "__main__":
    from torchsummary import summary
    from torchvision import transforms, utils
    import os, sys, time
    

    sys.path.insert(0, '/home/ubuntu/project/simply_yolo_v3/configs')
    from configs.resnet50_yolo_style_fpn_yolov3 import Config

    tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = 'cuda:0'

    in_img = np.zeros((416, 416, 3), np.uint8)
    t_img = transforms.ToTensor()(in_img)
    t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_img)
    t_img.unsqueeze_(0)
    t_img = t_img.to(device)

    temp_config = Config()
    backbone_cfg = temp_config.backbone_config[0]
    # print(backbone_cfg)
    # print(backbone_cfg['layer_st'])

    backbone = resnet18(False, **backbone_cfg).to(device)
    feature_maps = backbone(t_img)
    feature_maps = feature_maps
    print(len(feature_maps))
    for i in feature_maps:
        print(i.shape)
    # exit()
    # summary(backbone, (3, 416, 416))
    fpn = FPN(temp_config.fpn_config)
    fpn = fpn.to(device)
    print(fpn)
    out = fpn(feature_maps)

    for i in out:
        print(i.shape)

    
    # summary(fpn.to(device), (3, 416, 416))
