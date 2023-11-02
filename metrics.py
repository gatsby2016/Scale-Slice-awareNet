import SimpleITK as sitk
# import skimage.io as io
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import torch


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

def non_zero_mean(np_array):
    non_zero_nums = np.sum((np_array != 0),axis=1)
    return np.sum(np_array, axis=1)/ non_zero_nums


def evaluation(labeldata, preddata, K, MEAN = True):
    target = torch.as_tensor(labeldata, dtype=torch.int32, device=torch.device('cuda')).view(-1)
    output = torch.as_tensor(preddata, dtype=torch.int32, device=torch.device('cuda')).view(-1)

    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)  # get n_{ii} in confuse matrix
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_sum = area_output + area_target
    area_union = area_sum - area_intersection

    iou_class = area_intersection.float() / (area_union.float() + 1e-10)
    dice_class = 2 * area_intersection.float() / (area_sum.float() + 1e-10)

    if MEAN:
        non_zero_nums = K - torch.sum((area_target == 0)).cpu().numpy()
        mIoU = torch.sum(iou_class)/non_zero_nums
        mdice = torch.sum(dice_class)/non_zero_nums
        return mIoU.cpu().numpy(), mdice.cpu().numpy()
    else:
        return iou_class.cpu().numpy(), dice_class.cpu().numpy()



if __name__ == "__main__":
    gtpath = '/home/cyyan/projects/ISICDM2019/data/3D/test/'
    filename = glob(gtpath + '*label.nii')
    predpath = '/home/cyyan/projects/ISICDM2019/results/3Dtest_msacunet05/'

    nums_class = 55
    MEAN = True

    if MEAN:
        setDSC = []
        setIoU = []
        for gtname in filename:
            print(gtname)
            label = read_img(gtname)

            predname = predpath + gtname.split('/')[-1][:-9] + 'pred.nii'
            pred = read_img(predname)

            IoU, DSC = evaluation(label, pred, nums_class, MEAN=True)
            print('IoU:', IoU, 'Dice: ', DSC)
            setDSC.append(DSC)
            setIoU.append(IoU)

        print('IoU: ', sum(setIoU) / len(setIoU))
        print('DSC: ', sum(setDSC) / len(setDSC))

    else:
        setIoU = np.zeros((55, 1))
        setDSC = np.zeros((55, 1))
        for gtname in filename:
            print(gtname)
            label = read_img(gtname)

            predname = predpath + gtname.split('/')[-1][:-9] + 'pred.nii'
            pred = read_img(predname)

            IoU, DSC = evaluation(label, pred, nums_class, MEAN=False)
            print('IoU:', IoU, 'Dice: ', DSC)
            setIoU = np.concatenate((setIoU, IoU[:, np.newaxis]), axis=1)
            setDSC = np.concatenate((setDSC, DSC[:, np.newaxis]), axis=1)

        class_DSC = non_zero_mean(setDSC)
        for dice in class_DSC:
            print(dice)
        # print('DSC:', non_zero_mean(setDSC) )
        # print('IoU:', non_zero_mean(setIoU))