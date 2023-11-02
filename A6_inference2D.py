# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import numpy as np
from glob import glob
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
from _myUtils import *
from net_unet import UNet
from net_pspnet import PSPNet
from net_fcn import FCN
from net_unet_atrous2 import A2UNet
from net_unet_atrous_center import AcUNet
from net_msacunet import msacUNet
import sys
sys.path.append(".")
from Strainet_comparison_0525.ganComponents import *


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data, data.shape


# normalization
def normalize_maxmin(img):
    max_ = np.max(img)
    min_ = np.min(img)
    x = (img-min_)/(max_ - min_)*255
    return x.astype(np.uint8)


def transforms_data(image):
    norm_mean = [0.196]
    norm_std = [0.205]
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=norm_mean, std=norm_std)])
    return img_transform(image)


def main(path, savepath, network):
    filename = glob(path + '*label.nii')

    for name in filename:
        print(name)
        imgdata, imgshape = read_img(name[:-9] + '.nii')
        # imgdata = read_img(name[:-9] + '.nii')

        network.eval()
        with torch.no_grad():
            for ind in tqdm(range(len(imgdata))):
                # normalization and resize to 512*512
                img_nor = normalize_maxmin(imgdata[ind])
                img = Image.fromarray(img_nor).resize((512, 512))  # , resample=NEAREST
                # img = Image.fromarray(img_nor).resize((256, 256))  # only for minor revision strainet GAN

                # label = Image.fromarray(labeldata[ind]).resize((512, 512))

                img_tensor = transforms_data(img).unsqueeze(dim=0).cuda()

                outputs = network(img_tensor)
                prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)
                pred = prediction.squeeze().cpu().numpy().astype('int16')
                pred = np.array(Image.fromarray(pred).resize((imgshape[2], imgshape[1]), Image.NEAREST), dtype='int16')
                if not ind:
                    whlpre = pred[np.newaxis, :]
                else:
                    whlpre = np.concatenate((whlpre, pred[np.newaxis, :]), axis=0)

        nii = sitk.GetImageFromArray(whlpre)
        sitk.WriteImage(nii, savepath + name.split('/')[-1][:-9] + 'pred.nii')



# def main_ensmble(path, network1, network2, network3):
#     filename = glob(path + '*label.nii')
#
#     for name in filename:
#         print(name)
#         imgdata, imgshape = read_img(name[:-9] + '.nii')
#         # imgdata = read_img(name[:-9] + '.nii')
#         network1.eval()
#         network2.eval()
#         network3.eval()
#         with torch.no_grad():
#             for ind in tqdm(range(len(imgdata))):
#                 # normalization and resize to 512*512
#                 img_nor = normalize_maxmin(imgdata[ind])
#                 img = Image.fromarray(img_nor).resize((512, 512))  # , resample=NEAREST
#                 # label = Image.fromarray(labeldata[ind]).resize((512, 512))
#
#                 img_tensor = transforms_data(img).unsqueeze(dim=0).cuda()
#
#                 outputs1 = F.softmax(network1(img_tensor), 1, _stacklevel=5)
#                 outputs2 = F.softmax(network2(img_tensor), 1, _stacklevel=5)
#                 outputs3 = F.softmax(network3(img_tensor), 1, _stacklevel=5)
#
#                 prediction = torch.argmax((outputs1+outputs2+outputs3), dim=1)
#                 pred = prediction.squeeze().cpu().numpy().astype('int16')
#                 pred = np.array(Image.fromarray(pred).resize((imgshape[2], imgshape[1])), dtype='int16')
#
#                 if not ind:
#                     whlpre = pred[np.newaxis, :]
#                 else:
#                     whlpre = np.concatenate((whlpre, pred[np.newaxis, :]), axis=0)
#
#         nii = sitk.GetImageFromArray(whlpre)
#         sitk.WriteImage(nii, name[:-4]+'label.nii')
#


# noting: when Image.fromarray(np objects) is used, H*W will be converted to W*H
# also, when np.array(PIL.objects) is used, W*H wil be converted to H*W !!!!!!!
# also, torch.tensor will converted for PIL.objects.
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = '../data/3D/test/'
    nums_class = 55
    savpth = '../results/minorRevision_strainetGAN/'
    if os.path.exists(savpth):
        pass
    else:
        os.mkdir(savpth)

    # net = A2UNet(in_channels=1, n_classes=nums_class, feature_scale=1)
    net = ResSegNet(n_classes=nums_class, isRandomConnection=True, isSmallDilation=True,
                    isSpatialDropOut=False, dropoutRate=0.25)
    net.load_state_dict(torch.load('../model2D/minorGAN_2882.pkl'))
    # net = AUNet(in_channels=1, n_classes=nums_class, feature_scale=1)
    # net.load_state_dict(torch.load('../model2D/aunet_2903.pkl'))
    # net = UNet(in_channels=1, n_classes=nums_class, feature_scale=1)
    # net.load_state_dict(torch.load('../model2D/unet_2879.pkl'))
    # net = PSPNet(out_planes=nums_class).cuda()
    # net.load_state_dict(torch.load('../model2D/pspnet_2878.pkl'))
    # net = FCN(out_planes=nums_class).cuda()
    # net.load_state_dict(torch.load('../model2D/fcn_2291.pkl'))
    net.cuda()

    main(path, savpth, net)
