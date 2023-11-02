import SimpleITK as sitk
from glob import glob
import os
import numpy as np
import time

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

def read_img_slide(path, slide):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data[slide]


path = '/mnt/WorkStation/Students/博士生/闫朝阳/DataSetPelvic/split/'

lines = open(path+'class.txt',encoding='utf-8').read().rstrip().split('\n')
catetory2idx = {}
for line in lines:
    line_list = line.strip().split(':')
    catetory2idx[line_list[0]] = int(line_list[1])
print(catetory2idx)

# this loop is for each case
for ind in range(1,28):
    print(time.ctime())
    print('Case:', ind)
    cls = glob(path + str(ind) + '/*nii')

    case_shape = read_img(cls[0]).shape

    all_class = np.zeros(case_shape, dtype='int16')

    # this loop is for slides
    for sld in range(case_shape[0]):
        print('Slide: ', sld)

        all_class_one = np.zeros(case_shape[1:], dtype='int16')
        # this loop is for class
        for cl in cls:
            print('Class: ', cl)
            this_slide_cl = read_img_slide(cl,sld)

            classname = cl.split('/')[-1][:-4] # class for id
            cls_id= catetory2idx[classname]

            overlapping = all_class_one * this_slide_cl
            new_slide_cl = this_slide_cl - overlapping
            all_class_one = all_class_one + new_slide_cl # getting a matrix for validation overlapping
            all_class[sld] = all_class[sld] + (new_slide_cl*cls_id)

    nii = sitk.GetImageFromArray(all_class)
    sitk.WriteImage(nii, path + str(ind) + 'label.nii')
    # break

