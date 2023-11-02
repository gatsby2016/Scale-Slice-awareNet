# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint

from net_pspnet_resnet import resnet50

# from seg_opr.seg_oprs import ConvBnRelu
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class FCN(nn.Module):
    def __init__(self, out_planes, inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(FCN, self).__init__()
        self.backbone = resnet50(pretrained_model, inplace=inplace,
                                  norm_layer=norm_layer,
                                  bn_eps=1e-5,
                                  bn_momentum=0.1,
                                  deep_stem=True, stem_width=64)

        self.business_layer = []
        self.head = _FCNHead(2048, out_planes, inplace, norm_layer=norm_layer)
        self.aux_head = _FCNHead(1024, out_planes, inplace,
                                 norm_layer=norm_layer)

        self.business_layer.append(self.head)
        self.business_layer.append(self.aux_head)


    def forward(self, data, label=None):
        blocks = self.backbone(data)
        fm = self.head(blocks[-1])
        pred = F.interpolate(fm, scale_factor=32, mode='bilinear',
                             align_corners=True)

        aux_fm = self.aux_head(blocks[-2])
        aux_pred = F.interpolate(aux_fm, scale_factor=16, mode='bilinear',
                                 align_corners=True)
        # return pred, aux_pred
        return pred


class _FCNHead(nn.Module):
    def __init__(self, in_planes, out_planes, inplace=True,
                 norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_planes = in_planes // 4
        self.cbr = ConvBnRelu(in_planes, inter_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(inter_planes, out_planes, kernel_size=1,
                                 stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.conv1x1(x)
        return x


if __name__ == "__main__":
    model = FCN(21, None)
    print(model)