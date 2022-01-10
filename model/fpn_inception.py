import torch
import torch.nn as nn
from pretrainedmodels import inceptionresnetv2
import torch.nn.functional as F
from model.fpn_mobilenet_v2 import FPNHead

import functools
from torchsummary import summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        )
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        )
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0) # 256
        enc2 = self.enc2(enc1) # 512
        enc3 = self.enc3(enc2) # 1024
        enc4 = self.enc4(enc3) # 2048

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(F.pad(lateral2, pad, 'reflect') + nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return F.pad(lateral0, pad1, 'reflect'), map1, map2, map3, map4


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode='nearest')

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min = -1, max = 1)


if __name__ == '__main__':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    G = FPNInception(norm_layer=norm_layer)
    G.to(device=device)
    summary(G, (3, 256, 256))
