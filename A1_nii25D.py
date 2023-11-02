import SimpleITK as sitk
# import skimage.io as io
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np


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



#######################################
savepath = '/home/cyyan/projects/ISICDM2019/data/25D/train/'
path='/home/cyyan/projects/ISICDM2019/data/3D/train/'
filename = glob(path+'*label.nii')

for name in filename:
    print(name)
    labeldata = read_img(name)
    imgdata = read_img(name[:-9]+'.nii')
    imgdata = np.concatenate((imgdata[0][np.newaxis, :], imgdata, imgdata[-1][np.newaxis, :]), axis=0)
    for ind in range(1,len(imgdata)-1):
        # normalization and resize to 512*512
        img_nor = normalize_maxmin(imgdata[ind-1:ind+2]) # C*H*W
        img_nor_axis = np.swapaxes(np.swapaxes(img_nor,0,1), 1, 2) #C*H*W --> H*C*W --> H*W*C

        img = Image.fromarray(img_nor_axis).resize((512,512)) # , resample=NEAREST
        label = Image.fromarray(labeldata[ind-1]).resize((512,512))
        img.save(savepath + name.split('/')[-1][:-9] + '_' + str(ind) + '.png')
        label.save(savepath + name.split('/')[-1][:-9] +'_'+str(ind)+'_gt.png')
    # break
        # show_img_single(labeldata, channel=ind)
        # show_img_single(imgdata, channel=ind)

# img = Image.open(path+'10_0.png')
# im = np.array(img)
# gt = Image.open(path+'10_0_gt.png')
# gt = np.array(gt)
# print(im)
