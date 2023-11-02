import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics._ranking import precision_recall_curve
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


# root = '/mnt/WorkStation/Students/FXC.xlsx'
#
# data = np.array(pd.read_excel(root))
# label = np.array(data[:, -2], dtype=np.float32)
# pred = np.array(data[:, -1], dtype=np.float32)
#
# precision, recall, thresholds = precision_recall_curve(label, pred, pos_label=1)
# for p, r in zip(precision, recall):
#     if p == r:
#         print(p)
#         break
#
# auc = roc_auc_score(label, pred)
# # print(auc)
# fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
#
# plt.figure(figsize=(5, 5), dpi=600)
# plt.plot(precision, recall, linewidth=2, axisbg='#7FFF00')
# plt.xlabel('Recall', {'size': 16})
# plt.ylabel('Precision', {'size': 16})
# plt.xticks(np.arange(0, 1.1, step=0.2))
# plt.yticks(np.arange(0, 1.1, step=0.2))
# plt.text(0.1, 0.1, 'F1 Score=0.909', {'size': 12})
#
# # plt.show()
# plt.savefig('pr_curve.png', dpi=600, quality=95)


# x1 = np.random.normal(0, 0.8, 1000)
# x2 = np.random.normal(-2, 1, 1000)
# x3 = np.random.normal(3, 2, 1000)
#
# kwargs = dict(histtype='stepfilled', alpha=0.8, density=True, bins=40)
#
# plt.hist(x1, **kwargs)
# plt.hist(x2, **kwargs)
# plt.hist(x3, **kwargs)
# plt.show()
#
# x4 = np.random.normal(1, 4, 1000)
#
# plt.hist(x4, bins=30, density=True, alpha=0.5,
#          histtype='stepfilled', color='steelblue',
#          edgecolor='none')
# plt.show()

rng = np.random.RandomState(0)
content = ['UNet', 'n1PUNet', 'n3FPUNet', 'FCN', 'PSPNet', 'S2aNet', '2D3SUNet', 'P1UNet', 'P2UNet', 'FPUNet_I', 'n5PUNet', 'n7PUNet']
x = [0.6890, 0.72705, 0.7312, 0.63163, 0.7380, 0.74934, 0.71352, 0.72392, 0.7331, 0.71742, 0.66867, 0.65598] # DSC 0.2337,
# x = rng.randn(100)
y = [0.1347, 0.06465, 0.0034, 0.1509, 0.09618, 0.0022, 0.1189, 0.10943, 0.0959, 0.0982, 0.0776, 0.09696] # RVD 0.7112,
# y = rng.randn(100)
colors = [i * 10 for i in rng.rand(len(x))]

sizes = [31.04, 20.82, 24.72, 35.47, 56.23, 24.72, 31.04, 31.25, 31.86, 25.47,28.63, 32.54] # paramaters 5.60,
sizes = [i * 5 for i in sizes]

plt.scatter(x, y, c=colors, s=sizes, alpha=0.8, cmap='jet') #viridis
# plt.xticks([min(x)-0.05,max(x)+0.05])
plt.xticks([0.6, 0.8])
# plt.yticks([min(y)-0.05, max(y)+0.05])
plt.yticks([0.2, 0])
for i in range(len(x)):
    if content[i] == 'P2UNet':
        plt.text(x[i]-0.01, y[i] - 0.012, content[i])
    elif content[i] == 'PSPNet':
        plt.text(x[i] + 0.01, y[i], content[i])
    elif content[i] == 'S2aNet':
        plt.text(x[i] + 0.01, y[i]-0.001, content[i])
    else:
        plt.text(x[i]+0.005, y[i]+0.005, content[i])
plt.colorbar()  # 显示颜色对比条
plt.show()
