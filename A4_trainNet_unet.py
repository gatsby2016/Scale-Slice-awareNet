import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from _myUtils import *
from net_unet import UNet
# from net_pspnet import PSPNet
# from net_fcn import FCN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(3)
np.random.seed(3)
torch.cuda.manual_seed(3)

################ hyper-parameter
nums_class = 55

momentum = 0.9
learning_rate = 0.002

epoches = 10
loss_print = 20

weights = torch.tensor([0.0778, 0.9821, 0.9825, 1.0000, 1.0000, 0.9865, 0.9809, 0.9170, 0.9498, 0.9748, 0.9820,
                        0.6707, 0.6086, 0.2455, 0.2290, 0.2702, 0.2450, 0.1294, 0.1242, 0.4473, 0.5890, 0.5285,
                        0.5841, 0.7139, 0.6537, 0.7899, 0.6970, 0.9321, 0.8751, 0.9941, 0.9928, 0.8385, 0.7648,
                        0.7724, 0.8147, 0.4832, 0.5114, 0.8837, 0.8490, 0.8568, 0.4660, 0.4596, 0.7404, 0.6064,
                        0.7984, 0.6388, 0.4721, 0.8733, 0.9251, 0.6038, 0.8495, 0.9959, 0.9962, 0.9361, 0.8738]).cuda()
# weights = None

root = '/home/cyyan/projects/ISICDM2019/data/2D/'
train_loader, val_loader, train_nums, val_nums = reach_data(root, batch_size=6)

net = UNet(in_channels=1, n_classes=nums_class, feature_scale=1).cuda()
net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model2D/unet_2879.pkl'))
# net = PSPNet(out_planes=nums_class).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model/pspnet_3867.pkl'))
# net = FCN(out_planes=nums_class).cuda()
# net.load_state_dict(torch.load('/home/cyyan/projects/ISICDM2019/model/fcn_2900.pkl'))


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
metric = 0.2879
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
        torch.save(net.state_dict(), '/home/cyyan/projects/ISICDM2019/model2D/unet_%d.pkl'%(int(metric*10000)))