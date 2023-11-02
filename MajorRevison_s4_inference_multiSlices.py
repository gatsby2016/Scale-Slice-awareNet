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
from MajorRevision_net_msacunet import msacUNet
import scipy.ndimage as ndimage


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data, data.shape


def normalize_maxmin(img):
    for channel in range(len(img)):
        max_ = np.max(img[channel])
        min_ = np.min(img[channel])
        img[channel] = (img[channel]-min_)/(max_ - min_)*255
    return img.astype(np.uint8)


def transforms_data(image):
    norm_mean = [0.196]
    norm_std = [0.205]
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=norm_mean, std=norm_std)])
    return img_transform(image)


def main(path, savepath, network, num_slices):
    start = num_slices // 2

    filename = glob(path + '*label.nii')

    for name in filename:
        print(name)
        imgdata, imgshape = read_img(name.replace('label.nii', '.nii'))
        fuser = [imgdata[0][None, :]]*start + [imgdata] + [imgdata[-1][None, :]]*start
        imgdata = np.concatenate(fuser, axis=0)
        # imgdata = np.concatenate((imgdata[0][np.newaxis, :], imgdata, imgdata[-1][np.newaxis, :]), axis=0)

        network.eval()
        with torch.no_grad():
            for ind in tqdm(range(start, len(imgdata)-start)):
                img_nor = normalize_maxmin(imgdata[ind-start:ind+start+1])
                img_nor_axis = np.swapaxes(np.swapaxes(img_nor, 0, 1), 1, 2)  # C*H*W --> H*C*W --> H*W*C

                img = ndimage.zoom(img_nor_axis, (512 / img_nor_axis.shape[0], 512 / img_nor_axis.shape[1], 1),
                                   mode='nearest', order=0)

                img_tensor = transforms_data(img).unsqueeze(dim=0).cuda()

                outputs = network(img_tensor)
                prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)
                pred = prediction.squeeze().cpu().numpy().astype('int16')
                pred = np.array(Image.fromarray(pred).resize((imgshape[2], imgshape[1])), dtype='int16')
                if not ind-start:
                    whlpre = pred[np.newaxis, :]
                else:
                    whlpre = np.concatenate((whlpre, pred[np.newaxis, :]), axis=0)

        nii = sitk.GetImageFromArray(whlpre)
        sitk.WriteImage(nii, os.path.join(savepath, name.split('/')[-1].replace('label.nii', 'pred.nii')))
        # break


# noting: when Image.fromarray(np objects) is used, H*W will be converted to W*H
# also, when np.array(PIL.objects) is used, W*H wil be converted to H*W !!!!!!!
# also, torch.tensor will converted for PIL.objects.
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    modelpth = '../model2D/5_slices_msacunet_2914.pkl'
    # modelpth = '../model2D/7_slices_msacunet_2847.pkl'
    numslices = 5

    nums_class = 55
    path = '../data/3D/test/'
    savpth = '../results/majorRevision_3Dtest_' + str(numslices) + 'slices_macunet/'
    if not os.path.exists(savpth):
        os.mkdir(savpth)
    net = msacUNet(in_channels=1, n_classes=nums_class, feature_scale=1, nslices=numslices).cuda()
    net.load_state_dict(torch.load(modelpth))
    print(net)

    main(path, savpth, net, numslices)