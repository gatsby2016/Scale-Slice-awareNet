# import _init_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
from net_unet_layers import unetConv2, unetUp, init_weights, count_param


class _ASPPConv(nn.Module):  # size not change.
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
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        # out_channels = 256
        # self.b0 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #     norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        #     nn.ReLU(True))
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
        feat1 = self.b0(x)  # 1*1 standard conv. keep the same size
        feat2 = self.b1(x)  # 3*3 atrous conv with rate1. keep the same size
        feat3 = self.b2(x)  # 3*3 atrous conv with rate2. keep the same size
        feat4 = self.b3(x)  # 3*3 atrous conv with rate3. keep the same size
        # feat5 = self.b4(x) # GAP to 1*1 feature maps then interpolate to keep the same size
        feat5 = self.b4(x)  # 3*3 atrous conv with rate4. keep the same size
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

    # self.aspp = _ASPP(2048, 256, [3, 6, 12], norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs)


class A2UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=1, is_deconv=True, is_batchnorm=True):
        super(A2UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.aspp = _ASPP(filters[1], filters[1], [1, 2, 3, 5, 8], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
         # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # upsampling
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)  #output: 64*512*512
        maxpool1 = self.maxpool(conv1)  # 64*256*256

        conv2 = self.conv2(maxpool1)  # 128*256*256
        x = self.aspp(conv2) # 128*256*256
        maxpool2 = self.maxpool(x)  # 128*128*128

        conv3 = self.conv3(maxpool2)  # 256*128*128
        maxpool3 = self.maxpool(conv3)  # 256*64*64

        conv4 = self.conv4(maxpool3)  # 512*64*64

        up3 = self.up_concat3(conv4, conv3)  # 256*128*128
        up2 = self.up_concat2(up3, x)  # 128*256*256
        up1 = self.up_concat1(up2, conv1)  # 64*512*512

        final = self.final(up1)
        return final


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(2, 1, 64, 64)).cuda()
    model = A2UNet().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('AUNet totoal parameters: %.2fM (%d)' % (param / 1e6, param))
