import os
import time
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from _myUtils import *
from net_unet import UNet
from net_pspnet import PSPNet
from net_msacunet import msacUNet

from net_unet_layers import count_param

######################################### validation progress
def validation_ensmble(network1, network2, loader, class_nums, sample_nums, meanIOU = True, network3=None):
    if meanIOU:
        mIOUs = 0.0
        mdices = 0.0
    else:
        mIOUs = np.zeros((class_nums))
        mdices = np.zeros((class_nums))

    network1.eval()
    network2.eval()
    if network3:
        network3.eval()
    with torch.no_grad():
        for i, (img, target, _) in enumerate(loader):
            print(i)
            img = img.cuda()
            target = target.cuda().long()

            outputs1 = F.softmax(network1(img), 1, _stacklevel=5)
            outputs2 = F.softmax(network2(img), 1, _stacklevel=5)
            if network3:
                outputs3 = F.softmax(network3(img), 1, _stacklevel=5)
                # torch.cuda.empty_cache()
                prediction = torch.argmax((outputs1+outputs2+outputs3), dim=1)
            else:
                # torch.cuda.empty_cache()
                prediction = torch.argmax((outputs1 + outputs2), dim=1)

            m, d = cal_iou(prediction, target, class_nums, sample_nums, mean_iou =meanIOU)
            mIOUs += m
            mdices += d
    return mIOUs, mdices



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(3)
np.random.seed(3)
torch.cuda.manual_seed(3)

################ hyper-parameter
savepath = '/home/cyyan/projects/ISICDM2019/results/'
nums_class = 55

train_loader, val_loader, train_sample_nums, val_sample_nums = reach_data('/home/cyyan/projects/ISICDM2019/data/25D/', batch_size=6)
net1 = msacUNet(in_channels=1, n_classes=nums_class, feature_scale=1)
net1.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model2D/macunet_w_3006.pkl'))
net1.cuda()

# net2 = UNet(in_channels=1, n_classes=nums_class, feature_scale=2)
# net2.load_state_dict(torch.load('../model/unet2_3754.pkl'))
# net2.cuda()
#
# net3 = UNet(in_channels=1, n_classes=nums_class, feature_scale=4)
# net3.load_state_dict(torch.load('../model/unet4_3448.pkl'))
# net3.cuda()

# print(count_param(net3))

val_mIOU, val_mdice = validation(net1, train_loader, nums_class, train_sample_nums, meanIOU=False)
# val_mIOU, val_mdice = validation_ensmble(net1, net2, val_loader, nums_class,
#                                          val_sample_nums, meanIOU=True, network3=net3)
# print('Validation mIoU:{:.6f} mDICE:{:.6f}'.format(val_mIOU, val_mdice))
#
# print(val_mIOU)
print(torch.tensor(val_mdice).cuda())
# sorted_ = np.argsort(val_mIOU)[::-1]
# print(sorted_)
# weightes = np.argsort(sorted_)/28 + 0.0001
# w = torch.tensor(weightes).cuda()
# print(w)
