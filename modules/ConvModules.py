import warnings

import torch.nn as nn
# from mmcv.cnn import kaiming_init, constant_init

def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 inplace=True,):
        super(ConvModule, self).__init__()
        self.with_bias = bias


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.output_padding = self.conv.output_padding
        self.BN = nn.BatchNorm2d(self.out_channels)
        self.add_module('BN', self.BN)
        self.activate = nn.ReLU(inplace=inplace)

        # Default using msra init
        # self.init_weights()


    def init_weights(self):
        nonlinearity = 'relu' 
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        constant_init(self.BN, 1, bias=0)
        

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.BN(x)
        x = self.activate(x)
        
        return x
