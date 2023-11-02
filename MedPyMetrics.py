from glob import glob
from medpy.io import load
import medpy.metric.binary as bmetrics
import numpy as np
import csv
import os


def one_class_metrics(pred, gt):
    sensitivity = bmetrics.sensitivity(pred, gt)
    specificity = bmetrics.specificity(pred, gt)
    dice = bmetrics.dc(pred, gt)
    # ppv = bmetrics.positive_predictive_value(pred, gt)
    if not np.sum(pred):
        hd = 'nan'
        hd95 = 'nan'
        assd = 'nan'
    else:
        hd = bmetrics.hd(pred, gt)
        hd95 = bmetrics.hd95(pred, gt)
        assd = bmetrics.assd(pred, gt)
    ravd = bmetrics.ravd(pred, gt)

    # hd = 'nan'
    # hd95 = 'nan'
    return np.array([sensitivity, specificity, dice, hd, hd95, assd, ravd])


if __name__ == "__main__":
    gtpath = '/home/cyyan/projects/ISICDM2019/data/3D/test/'
    filename = glob(gtpath + '*label.nii')
    task = 'minorRevision_strainetGAN' #
    predpath = os.path.join('/home/cyyan/projects/ISICDM2019/results', task)
    csvfile = predpath + '_evaluation_MinorRevision.csv'

    nums_class = 55

    tags = ['sensitivity', 'specificity', 'dice', 'hd', 'hd95', 'assd', 'ravd']
    for gtname in filename:
        print(gtname)
        predname = os.path.join(predpath, gtname.split('/')[-1].replace('label.nii', 'pred.nii'))

        gtdata, _ = load(gtname)
        preddata, _ = load(predname)

        with open(csvfile, 'a') as datacsv:
            csvw = csv.writer(datacsv)
            csvw.writerow(tags)

        one_case_metrics = []
        for ind in range(1, nums_class):
            gt = (gtdata == ind)
            pred = (preddata == ind)

            if np.sum(gt) == 0:
                one_case_metrics.append(np.array(['nan','nan','nan','nan','nan','nan','nan']))
                continue
            one_case_metrics.append(one_class_metrics(pred, gt))
        with open(csvfile, 'a') as datacsv:
            csvw = csv.writer(datacsv)
            csvw.writerows(one_case_metrics)

            # print(str(ind), ':', one_case_metrics)