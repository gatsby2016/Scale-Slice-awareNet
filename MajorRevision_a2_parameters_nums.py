import torch
from net_unet import UNet
from net_unet_atrous_center import AcUNet
from net_msacunet import msacUNet

# from net_fcn import FCN
# from net_pspnet import PSPNet

from net_unet_atrous import AUNet
from net_unet_atrous2 import A2UNet
# from net_msacunet_noshare import msacUNet
from MajorRevision_net_msacunet import msacUNet
import sys
sys.path.append(".")
from Strainet_comparison_0525.ganComponents import *


# net = UNet(n_classes=55).cuda()
nums_class = 55
# net = msacUNet(in_channels=1, n_classes=nums_class, nslices=7, feature_scale=1).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model2D/unet_2879.pkl'))
net = ResSegNet(n_classes=nums_class, isRandomConnection=True, isSmallDilation=True,
                isSpatialDropOut=False, dropoutRate=0.25)
netD = Discriminator()
# net.load_state_dict(torch.load('../model2D/minorGAN_2883.pkl'))

num_params = sum(param.numel() for param in net.parameters())
num_params2 = sum(param.numel() for param in netD.parameters())

print(num_params+num_params2)