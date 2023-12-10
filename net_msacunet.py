# import _init_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
from net_unet_layers import unetConv2, unetUp, init_weights, count_param


class _ASPPConv(nn.Module): # size not change.
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,  atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()

        rate0, rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        self.b0 = _ASPPConv(in_channels, out_channels, rate0, norm_layer, norm_kwargs)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        # self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.b4 = _ASPPConv(in_channels, out_channels, rate4, norm_layer, norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x) # 1*1 standard conv. keep the same size
        feat2 = self.b1(x) # 3*3 atrous conv with rate1. keep the same size
        feat3 = self.b2(x) # 3*3 atrous conv with rate2. keep the same size
        feat4 = self.b3(x) # 3*3 atrous conv with rate3. keep the same size
        # feat5 = self.b4(x) # GAP to 1*1 feature maps then interpolate to keep the same size
        feat5 = self.b4(x) # 3*3 atrous conv with rate4. keep the same size
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

    # self.aspp = _ASPP(2048, 256, [3, 6, 12], norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs)



class msacUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=1, is_deconv=True, is_batchnorm=True):
        super(msacUNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.aspp = _ASPP(filters[3], filters[3], [1, 2, 3, 5, 8], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2]*3, filters[3], self.is_batchnorm)
        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # upsampling
        # self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, n_concat=2)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, n_concat=2)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, n_concat=2)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1_1 = self.conv1(inputs[:,1:2,:,:])  #output: 64*512*512
        maxpool1_1 = self.maxpool(conv1_1)  # 64*256*256
        conv2_1 = self.conv2(maxpool1_1)  # 128*256*256
        maxpool2_1 = self.maxpool(conv2_1)  # 128*128*128
        conv3_1 = self.conv3(maxpool2_1)  # 256*128*128
        maxpool3_1 = self.maxpool(conv3_1)  # 256*64*64

        conv1_0 = self.conv1(inputs[:, 0:1, :, :])  # output: 64*512*512
        maxpool1_0 = self.maxpool(conv1_0)  # 64*256*256
        conv2_0 = self.conv2(maxpool1_0)  # 128*256*256
        maxpool2_0 = self.maxpool(conv2_0)  # 128*128*128
        conv3_0 = self.conv3(maxpool2_0)  # 256*128*128
        maxpool3_0 = self.maxpool(conv3_0)  # 256*64*64

        conv1_2 = self.conv1(inputs[:, 2:, :, :])  # output: 64*512*512
        maxpool1_2 = self.maxpool(conv1_2)  # 64*256*256
        conv2_2 = self.conv2(maxpool1_2)  # 128*256*256
        maxpool2_2 = self.maxpool(conv2_2)  # 128*128*128
        conv3_2 = self.conv3(maxpool2_2)  # 256*128*128
        maxpool3_2 = self.maxpool(conv3_2)  # 256*64*64

        maxpool_cat = torch.cat([maxpool3_1, maxpool3_0, maxpool3_2], 1)
        conv4 = self.conv4(maxpool_cat)  # 512*64*64

        up4 = self.aspp(conv4)
        # up3 = self.up_concat3(up4, torch.cat([conv3_1, conv3_0, conv3_2], 1))  # 256*128*128
        # up2 = self.up_concat2(up3, torch.cat([conv2_1, conv2_0, conv2_2], 1))  # 128*256*256
        # up1 = self.up_concat1(up2, torch.cat([conv1_1, conv1_0, conv1_2], 1))  # 64*512*512

        up3 = self.up_concat3(up4, conv3_1)  # 256*128*128
        up2 = self.up_concat2(up3, conv2_1)  # 128*256*256
        up1 = self.up_concat1(up2, conv1_1)  # 64*512*512

        final = self.final(up1)
        return final


if __name__ == '__main__':
    pass
