# coding: utf-8
# scripts for computing mean and std
# from __future__ import print_function
import numpy as np
import random
import os
import cv2
from PIL import Image
import argparse
import glob


#%% define mean and std compution function
def ComputeMeanStd(imagename):

    img = Image.open(imagename)
    # img = img[:,:, (2,1,0)] # BGR to RGB
    arr = np.array(img)/255.
    mean_vals = np.mean(arr)
    std_vals = np.std(arr) # Compute the mean  and std along the specified axis 0,1.
    return mean_vals, std_vals

# define compution function for filelist
def ComputeMeanStd_List(List):
    Mean_val = []
    Std_val = []
    for name in List:
        # filename = os.path.join(path, name)
        M, S = ComputeMeanStd(name)
        Mean_val.append(M)
        Std_val.append(S)
    datamean = np.mean(Mean_val, 0)
    datastd = np.mean(Std_val, 0)
    return datamean, datastd


#### get args
def GetArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--part', type=int, default=1, choices=[1, 10, 100, 1000,10000],
                        help='1/Part of len(files), have 1, 10, 100, 1000, 10000 choices')

    parser.add_argument('-P', '--path', type=str,
                        # default=['/home/cyyan/Challenge/data/train/停车场',
                        #          '/home/cyyan/Challenge/data/train/停车场'], help='Img path1')
                        default= '/home/cyyan/projects/ISICDM2019/data/2D/train/', help='Img path1')

    parser.add_argument('-S', '--savepath', type=str, default='2DMeanStd.npz',
                        help='Path to save the MeanStd value')
    args = parser.parse_args()
    return args


#%%
if __name__ == '__main__':
    args = GetArgs()

    filelist = glob.glob(args.path + '*_gt.png')
    filelist = [(x[:-7] + '.png') for x in filelist]

    random.shuffle(filelist)
    nums = len(filelist)//args.part # randomly selected 1/part data for compution
    Singlemean, Singlestd = ComputeMeanStd_List(filelist[0:nums])
    print("normMean = {:.5f}".format(Singlemean))
    print("normStd = {:.5f}".format(Singlestd))
    print('transforms.Normalize({}, {})'.format(Singlemean, Singlestd))

    # np.savez(args.savepath, NormMean = NormMean, NormStd = NormStd)
    ######################################## load the npz data
    # npzfile = np.load('MeanStd.npz')
    # Mean, Std = npzfile['NormMean'], npzfile['NormStd']
