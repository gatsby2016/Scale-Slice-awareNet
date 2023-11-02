import SimpleITK as sitk
# import skimage.io as io
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import torch
import cv2



def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

def non_zero_mean(np_array):
    non_zero_nums = np.sum((np_array != 0),axis=1)
    return np.sum(np_array, axis=1)/ non_zero_nums


def statistics(labeldata, K):
    target = torch.as_tensor(labeldata, dtype=torch.int32, device=torch.device('cuda')).view(-1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    return area_target.cpu().numpy()


if __name__ == "__main__":
    # gtpath = '/home/cyyan/projects/ISICDM2019/data/3D/val/'
    gtpath = '/mnt/WorkStation/Students/博士生/闫朝阳/DataSetPelvic/split/'
    filename = glob(gtpath + '*label.nii')
    predpath = '/home/cyyan/projects/ISICDM2019/results/3Dtest_macunet_w/'

    nums_class = 55
    MEAN = False

    setratio = np.zeros([55])
    for gtname in filename:
        print(gtname)
        label = read_img(gtname)

        setratio = setratio + statistics(label, nums_class)

    for ratio in (setratio[1:]/ np.sum(setratio[1:])):
        print(ratio)
    # print('DSC: ', sum(setratio) / len(setratio))
