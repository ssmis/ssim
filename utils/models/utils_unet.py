#####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif 'Linear' in classname:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.xavier_normal_(m.weight.data, gain=1)
    elif 'Linear' in classname:
        init.xavier_normal_(m.weight.data, gain=1)
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'Linear' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.orthogonal_(m.weight.data, gain=1)
    elif 'Linear' in classname:
        init.orthogonal_(m.weight.data, gain=1)
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True)
                )
                setattr(self, f'conv{i}', conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.ReLU(inplace=True)
                )
                setattr(self, f'conv{i}', conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            x = getattr(self, f'conv{i}')(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1) if is_deconv else nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.children():
            if isinstance(m, unetConv2): continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = [offset // 2] * 4
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

