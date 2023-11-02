import SimpleITK as sitk
# import skimage.io as io
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import os
import scipy.ndimage as ndimage


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 单张显示
def show_img_single(ori_img, channel):
    plt.imshow(ori_img[channel], cmap='gray')
    plt.show()

# normalization
def normalize_maxmin(img):
    for channel in range(len(img)):
        max_ = np.max(img[channel])
        min_ = np.min(img[channel])
        img[channel] = (img[channel]-min_)/(max_ - min_)*255
    return img.astype(np.uint8)


if __name__ == "__main__":
    path='/home/cyyan/projects/ISICDM2019/data/3D/val/'
    filename = glob(path+'*label.nii')

    n = 5
    savepath = '/home/cyyan/projects/ISICDM2019/data/' + str(n)+ 'slices/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    savepath = '/home/cyyan/projects/ISICDM2019/data/' + str(n)+ 'slices/val/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    start_pos = 2
    for name in filename:
        print(name)
        labeldata = read_img(name)
        imgdata = read_img(name.replace('label.nii', '.nii'))
        imgdata = np.concatenate((imgdata[0][np.newaxis, :], imgdata[0][np.newaxis, :], imgdata,
                                  imgdata[-1][np.newaxis, :], imgdata[-1][np.newaxis, :]), axis=0)
        for ind in range(start_pos, len(imgdata)-start_pos):
            # normalization and resize to 512*512
            img_nor = normalize_maxmin(imgdata[ind-start_pos:ind+start_pos+1]) # C*H*W
            img_nor_axis = np.swapaxes(np.swapaxes(img_nor,0,1), 1, 2) #C*H*W --> H*C*W --> H*W*C

            img = ndimage.zoom(img_nor_axis, (512/img_nor_axis.shape[0], 512/img_nor_axis.shape[1], 1), mode='nearest', order=0)
            label = ndimage.zoom(labeldata[ind-start_pos], (512/labeldata.shape[1], 512/labeldata.shape[2]), mode='nearest', order=0)

            np.save(savepath + name.split('/')[-1].replace('label.nii', '') + '_' + str(ind) + '_gt.npy', label)
            np.save(savepath + name.split('/')[-1].replace('label.nii', '') +'_'+str(ind)+'.npy', img)
            # label.save(savepath + name.split('/')[-1].replace('label.nii', '') +'_'+str(ind)+'_gt.png')
            # img.save(savepath + name.split('/')[-1].replace('label.nii', '') + '_' + str(ind) + '.png')