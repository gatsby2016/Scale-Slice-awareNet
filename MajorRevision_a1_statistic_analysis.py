from scipy.stats import kruskal, levene, f_oneway, friedmanchisquare
import os
import numpy as np
import pandas as pd

def cal_iou(a):
    return a/(2-a)

def Case_Type_Metrics_Organization(path, num_case=6, num_category=55, num_metric=7, keepcase=True):
    all_case_type_metric = np.empty([num_case, num_category - 1, num_metric])

    content = pd.read_csv(path, header=None).squeeze()
    for nums in range(num_case):
         temp = np.array(content)[num_category * nums + 1:num_category * (nums + 1), :]
         all_case_type_metric[nums, :, :] = temp.astype(np.float)[np.newaxis, ...]

    all_case_type_metric = np.delete(all_case_type_metric, (3, 4), axis=2)
    # print(all_case_type_metric.shape)
    ious = cal_iou(all_case_type_metric[:, :, 2])
    all_case_type_metric = np.insert(all_case_type_metric, 0, values=ious, axis=2)

    all_case_type_metric = np.nan_to_num(all_case_type_metric)

    if keepcase:
        return np.nanmean(all_case_type_metric, axis=1)
        # return np.nanmean(all_case_type_metric, axis=(0, 1))
    else:
        return np.nanmean(all_case_type_metric, axis=0)
        # return all_case_type_metric.reshape(-1, 5+1)


if __name__ == '__main__':
    category = 55
    nums_case = 6
    nums_metric = 7

    # tasks = ['unet', 'macunet_w', 'msacunet_noshare', '2Dunet', '2Dpspnet', '2Dmacunet', 'a2unet', '2Dfcn', '2Daunet', '2Dacunet']
    ablation_tasks = ['unet', '2Dacunet', '2Dmacunet', 'macunet_w']
    other_models_tasks = ['2Dfcn', 'unet', '2Dpspnet', 'macunet_w']
    between_2D_3D_tasks = ['unet', '2Dunet', '3DUnet', 'macunet_w']
    placement_scam_tasks = ['2Daunet', 'a2unet', '2Dacunet', 'macunet_w']
    parameter_slim_tasks = ['2Dmacunet', 'msacunet_noshare', 'macunet_w']
    tasks = ablation_tasks

    all_tasks_metrics = np.empty([len(tasks), category-1, nums_metric-2+1]) # *nums_case
    # all_tasks_metrics = np.empty([len(tasks), nums_case*(category-1), nums_metric-2+1]) # *nums_case

    for i, t in enumerate(tasks):
        file_path = os.path.join('../results', '3Dtest_' + t + '_evaluation_MajorRevision.csv')
        # metrics_vals = Case_Type_Metrics_Organization(file_path, nums_case, category, nums_metric, keepcase=True)
        metrics_vals = Case_Type_Metrics_Organization(file_path, nums_case, category, nums_metric, keepcase=False)
        # print(metrics_vals)
        all_tasks_metrics[i, :, :] = metrics_vals[np.newaxis, ...]

    for ind in range(nums_metric-2+1):
        if len(tasks) == 4:
            args = [all_tasks_metrics[0, :, ind], all_tasks_metrics[1, :, ind], all_tasks_metrics[2, :, ind], all_tasks_metrics[3,:, ind]]
        else:
            args = [all_tasks_metrics[0, :, ind], all_tasks_metrics[1, :, ind], all_tasks_metrics[2, :, ind]]

        # method 1
        # stat,p = kruskal(*args, nan_policy='omit')
        # print("stat为：%f" %stat,"p值为：%f" %p)

        # method 2
        stat, p = friedmanchisquare(*args)
        print("stat为：%f" % stat, "p值为：%f" % p)

        # method 3
        # w, p = levene(*args)
        # print("w为：%f" %w, "p值为：%f" %p)
        # if p > 0.05:
        #     f, p = f_oneway(*args)
        #     print("METRICS: %d" % ind, "f为：%f" % f, "p值为：%f" % p)