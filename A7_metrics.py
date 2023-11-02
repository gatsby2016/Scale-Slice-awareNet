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


def main(path, network):
    filename = glob(path + '*label.nii')

    for name in filename:
        print(name)
        labeldata, _ = read_img(name)
        imgdata, imgshape = read_img(name[:-9] + '.nii')
        # normalization and resize to 512*512
        img_nor = normalize_maxmin(imgdata)  # C*H*W
        img_nor_axis = np.swapaxes(np.swapaxes(img_nor, 0, 1), 1, 2)  # C*H*W --> H*C*W --> H*W*C
        img = Image.fromarray(img_nor_axis).resize((512, 512))  # , resample=NEAREST
        img = np.array(img) # np img is H*W*C

        img_transforms = transforms_data(img) # C*H*W in range(0,1)
        img_transforms0 = torch.cat((img_transforms[:0], img_transforms[0:-1]), dim=0).unsqueeze(dim=1)
        img_transforms2 = torch.cat((img_transforms[1:], img_transforms[-1:]), dim=0).unsqueeze(dim=1)

        input_img = torch.cat((img_transforms0, img_transforms.unsqueeze(dim=1), img_transforms2), dim=1).cuda()

        network.eval()
        with torch.no_grad():
            outputs = network(input_img)
            prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)
            pred = prediction.squeeze().cpu().numpy().astype('int16')
            pred = np.array(Image.fromarray(pred).resize((imgshape[2], imgshape[1])), dtype='int16')

        # nii = sitk.GetImageFromArray(pred)
        # sitk.WriteImage(nii, name[:-4]+'label.nii')


# noting: when Image.fromarray(np objects) is used, H*W will be converted to W*H
# also, when np.array(PIL.objects) is used, W*H wil be converted to H*W !!!!!!!
# also, torch.tensor will converted for PIL.objects.
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = '../data/3D/test/'
    nums_class = 55

    net = UNet(in_channels=3, n_classes=nums_class, feature_scale=1)
    net.load_state_dict(torch.load('../model/unet_2840.pkl'))
    net.cuda()


    # main(path, net2)
    main(path, net)