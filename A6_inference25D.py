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
# from net_msacunet_noshare import msacUNet
from net_munet import mUNet


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data, data.shape


# normalization
def normalize_maxmin(img):
    for channel in range(len(img)):
        max_ = np.max(img[channel])
        min_ = np.min(img[channel])
        img[channel] = (img[channel]-min_)/(max_ - min_)*255
    return img.astype(np.uint8)


def transforms_data(image):
    norm_mean = [0.196,0.196,0.196]
    norm_std = [0.205,0.205,0.205]
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=norm_mean, std=norm_std)])
    return img_transform(image)


def main(path, savepath, network):
    filename = glob(path + '*label.nii')

    for name in filename:
        print(name)
        # imgdata, imgshape = read_img(name)
        imgdata, imgshape = read_img(name[:-9] + '.nii')
        imgdata = np.concatenate((imgdata[0][np.newaxis, :], imgdata, imgdata[-1][np.newaxis, :]), axis=0)

        network.eval()
        with torch.no_grad():
            for ind in tqdm(range(1,len(imgdata)-1)):
                # normalization and resize to 512*512
                img_nor = normalize_maxmin(imgdata[ind-1:ind+2])
                img_nor_axis = np.swapaxes(np.swapaxes(img_nor, 0, 1), 1, 2)  # C*H*W --> H*C*W --> H*W*C
                img = Image.fromarray(img_nor_axis).resize((512, 512))  # , resample=NEAREST
                img = np.array(img)
                # label = Image.fromarray(labeldata[ind]).resize((512, 512))

                img_tensor = transforms_data(img).unsqueeze(dim=0).cuda()

                outputs = network(img_tensor)
                prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)
                pred = prediction.squeeze().cpu().numpy().astype('int16')
                pred = np.array(Image.fromarray(pred).resize((imgshape[2], imgshape[1])), dtype='int16')
                if not ind-1:
                    whlpre = pred[np.newaxis, :]
                else:
                    whlpre = np.concatenate((whlpre, pred[np.newaxis, :]), axis=0)

        nii = sitk.GetImageFromArray(whlpre)
        sitk.WriteImage(nii, savepath + name.split('/')[-1][:-9] + 'pred.nii')
        # break


# noting: when Image.fromarray(np objects) is used, H*W will be converted to W*H
# also, when np.array(PIL.objects) is used, W*H wil be converted to H*W !!!!!!!
# also, torch.tensor will converted for PIL.objects.
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = '../data/3D/test/'
    nums_class = 55
    savpth = '../results/3Dtest_munet/'
    if os.path.exists(savpth):
        pass
    else:
        os.mkdir(savpth)
    # net = UNet(in_channels=3, n_classes=nums_class, feature_scale=1)
    # net.load_state_dict(torch.load('../model/unet_2840.pkl'))
    # net = PSPNet(out_planes=nums_class).cuda()
    # net.load_state_dict(torch.load('../model/pspnet_2835.pkl'))
    # net = FCN(out_planes=nums_class).cuda()
    # net.load_state_dict(torch.load('../model/fcn_2283.pkl'))
    net = mUNet(in_channels=1, n_classes=nums_class, feature_scale=1)
    net.load_state_dict(torch.load('../model2D/mUNet_2896.pkl'))
    net.cuda()

    main(path, savpth, net)
    # main_ensmble(path, net1, net2, net3)