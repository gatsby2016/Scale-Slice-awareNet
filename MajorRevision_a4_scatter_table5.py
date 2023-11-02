import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics._ranking import precision_recall_curve
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


rng = np.random.RandomState(0)

RVD_val = np.array(pd.read_excel('../results/ablation_study_results_on_categories_for_scatter_RVD.xlsx'))[:, 1:]
DSC_val = np.array(pd.read_excel('../results/ablation_study_results_on_categories_for_scatter_DSC.xlsx'))[:, 1:]
labelmap = np.array(pd.read_excel('../results/labelmap.xlsx').dropna())[:, 1:]
# colors = [i * 10 for i in rng.rand(len(DSC_val))]
tmp = list(labelmap.astype(np.uint8))
colors = []
for i in tmp:
    c = '#'
    for j in i:
        if len(hex(j)[2:]) == 1:
            c += '0'
            c += hex(j)[2:]
        else:
            c += hex(j)[2:]
    colors.append(c)
# sizes = [31.04, 20.82, 24.72, 35.47, 56.23, 24.72, 31.04, 31.25, 31.86, 25.47,28.63, 32.54] # paramaters 5.60,
sizes = []


# kwargs = dict(histtype='stepfilled', alpha=0.8, density=True, bins=10, edgecolor='none')
#
# for ind in range(DSC_val.shape[1]):
#     plt.hist(DSC_val[:, ind], **kwargs)
# plt.show()

#
# x4 = np.random.normal(1, 4, 1000)
#
# plt.hist(x4, bins=30, density=True, alpha=0.5,
#          histtype='stepfilled', color='steelblue',
#          edgecolor='none')
# plt.show()

# markers = ['^', '+', '*', 'o']
# for ind in range(DSC_val.shape[1]):
#     plt.scatter(DSC_val[:, ind], np.abs(RVD_val[:, ind]), c=colors, alpha=0.8, marker=markers[ind], cmap='jet') #viridis
# # plt.xticks([0.02, 0.95])
# # plt.yticks([0, 2.7])
# plt.colorbar()  # 显示颜色对比条
# plt.show()

val = RVD_val

x = range(1, 55)
markers = ['^', '.', '*', 'o']
plt.figure(figsize=[40, 20])
for ind in range(val.shape[1]):
    plt.scatter(x, np.abs(val[:, ind]), c=colors, s=2000, alpha=0.8, marker=markers[ind]) #viridis, cmap='viridis'
plt.yticks([])
plt.xticks([])
plt.xlabel('')
plt.ylabel('')
# plt.colorbar()  # 显示颜色对比条
# plt.savefig('../results/majorRevision_scatter.png', dpi=600, quality=95)
plt.show()