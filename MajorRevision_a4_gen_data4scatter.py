import os
import numpy as np
import pandas as pd
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

    if keepcase:
        return np.nanmean(all_case_type_metric, axis=1)
        # return np.nanmean(all_case_type_metric, axis=(0, 1))
    else:
        # all_case_type_metric = np.nan_to_num(all_case_type_metric)
        return np.nanmean(all_case_type_metric, axis=0)

        # return all_case_type_metric.reshape(-1, 5)


if __name__ == '__main__':
    category = 55
    nums_case = 6
    nums_metric = 7

    # tasks = ['unet', '2Dacunet', '2Dmacunet', '2Dfcn', '2Dpspnet', 'macunet_w', '2Dunet', '3DUnet', '2Daunet', 'a2unet', 'msacunet_noshare']
    # tasks = ['5slices_macunet', '7slices_macunet', '2DAUnet', '2DA2Unet']
    tasks = ['unet', '2Dacunet', '2Dmacunet', 'macunet_w']

    # all_tasks_metrics = np.empty([len(tasks), (category-1), nums_metric-2]) # *nums_case
    all_tasks_metrics = np.empty([category-1, len(tasks)])  # *nums_case

    for i, t in enumerate(tasks):
        file_path = os.path.join('../results', '3Dtest_' + t + '_evaluation_MajorRevision.csv')
        # metrics_vals = Case_Type_Metrics_Organization(file_path, nums_case, category, nums_metric, keepcase=True)
        metrics_vals = Case_Type_Metrics_Organization(file_path, nums_case, category, nums_metric, keepcase=False)
        metrics_vals = metrics_vals[:, 5]
        print(metrics_vals)
        # all_tasks_metrics[i, :, :] = metrics_vals[np.newaxis, ...]
        all_tasks_metrics[:, i] = metrics_vals


    res = pd.DataFrame({tasks[0]:all_tasks_metrics[:, 0], tasks[1]: all_tasks_metrics[:, 1], tasks[2]: all_tasks_metrics[:, 2], tasks[3]:all_tasks_metrics[:, 3]})
    res.to_excel('../results/ablation_study_results_on_categories_for_scatter_RVD.xlsx')

    # print(all_tasks_metrics)

