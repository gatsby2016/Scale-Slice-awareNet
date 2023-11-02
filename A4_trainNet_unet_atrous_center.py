import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from _myUtils import *
from net_unet_atrous_center import AcUNet
# from net_pspnet import PSPNet
# from net_fcn import FCN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(3)
np.random.seed(3)
torch.cuda.manual_seed(3)

################ hyper-parameter
nums_class = 55

momentum = 0.9
learning_rate = 0.02

epoches = 100
loss_print = 20

# weights = torch.tensor([0.0866, 0.8463, 0.6306, 0.3679, 0.3606, 0.4326, 0.4114, 0.2683, 0.2270,
#                         0.3948, 0.6061, 0.6614, 0.6962, 0.8452, 0.7468, 0.9280, 0.8539, 0.8435,
#                         0.8701, 0.4992, 0.6405, 0.9478, 0.8889, 0.5930, 0.5579, 0.8963, 0.7710,
#                         0.9051, 0.6307]).cuda()
weights = None

root = '/home/cyyan/santi/MRI/data/2D/'
train_loader, val_loader, train_nums, val_nums = reach_data(root, batch_size=6)

net = AcUNet(in_channels=1, n_classes=nums_class, feature_scale=1).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model/unet_2567.pkl'))
# net = PSPNet(out_planes=nums_class).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model/pspnet_3867.pkl'))
# net = FCN(out_planes=nums_class).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model/fcn_2900.pkl'))
print(net)

if weights is None:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    # criterion = DiceCEloss(weights=weights)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.3, patience=3, verbose=True, cooldown=3, min_lr=0.00001)


# ######################################## training progress
metric = 0.25
print('Start training...')
for epoch in range(epoches):
    start = time.time()
    emIOU, eloss = train(net, train_loader, criterion, optimizer, nums_class, train_nums, loss_print)

    print('==== Epoch {:3d} ==== lr {} ==== Time(s) {:.2f} ==== Average loss {:.4f} ==== mIOU {:.6f} ===='
          .format(epoch, optimizer.param_groups[0]['lr'], time.time()-start, eloss, emIOU))
    scheduler.step()

    val_mIOU, val_mdice = validation(net, val_loader, nums_class, val_nums, meanIOU=True)

    ### cal the cross entroy loss weights reference to function: cal_iou()
    # val_mIOU = np.mean(val_mIOU)
    # val_mdice = np.mean(val_mdice)

    print('Validation mIoU:{:.6f} mDICE:{:.6f}'.format(val_mIOU, val_mdice))
    if val_mIOU > metric:
        metric = val_mIOU
        torch.save(net.state_dict(), '/home/cyyan/santi/MRI/model2D/AcUNet_%d.pkl'%(int(metric*10000)))