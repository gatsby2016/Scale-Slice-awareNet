import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from _readData import SegData, MajorRevison_SegData

import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ########################## reach_data
def reach_data(pth, batch_size):
    # norm_mean = [0.196,0.196,0.196]
    # norm_std = [0.205,0.205,0.205]
    norm_mean = [0.196]
    norm_std = [0.205]
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=norm_mean, std=norm_std)])
    train_data = SegData(pth, mode='trainAUG/', transform=img_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_data = SegData(pth, mode='val/', transform=img_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    return train_loader, val_loader, train_data.__len__(), val_data.__len__()


# ########################## reach_data
def majorRevision_reach_data(pth, batch_size):
    # norm_mean = [0.196,0.196,0.196]
    # norm_std = [0.205,0.205,0.205]
    norm_mean = [0.196]
    norm_std = [0.205]
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=norm_mean, std=norm_std)])
    train_data = MajorRevison_SegData(pth, mode='trainAUG/', transform=img_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_data = MajorRevison_SegData(pth, mode='val/', transform=img_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    return train_loader, val_loader, train_data.__len__(), val_data.__len__()




# ########################## cal_iou     # https://github.com/pytorch/pytorch/issues/1382
def cal_iou(output, target, K, allNums=2560, mean_iou= True):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    # assert (output.dim() in [1, 2, 3])
    # assert output.shape == target.shape
    output = output.view(output.shape[0], -1)
    target = target.view(target.shape[0], -1)

    # # average the batch_size
    for ind in range(output.shape[0]):
        one_output = output[ind, :]
        one_target = target[ind, :]
        # one_output[one_target == ignore_index] = ignore_index
        intersection = one_output[one_output == one_target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) # get n_{ii} in confuse matrix
        area_target = torch.histc(one_target, bins=K, min=0, max=K-1)
        area_output = torch.histc(one_output, bins=K, min=0, max=K-1)
        area_union = area_output + area_target - area_intersection
        one_iou_class = area_intersection.float() / (area_union.float() + 1e-10)
        one_dice_class = 2*area_intersection.float() / (area_union.float() + area_intersection.float() + 1e-10)

        if not ind:
            iou_class = one_iou_class
            dice_class = one_dice_class
        else:
            iou_class += one_iou_class
            dice_class += one_dice_class

    iou_class = iou_class / allNums
    dice_class = dice_class / allNums

    if mean_iou:
        mIoU = torch.mean(iou_class)
        mdice = torch.mean(dice_class)
        return mIoU.cpu().detach().numpy(), mdice.cpu().detach().numpy()
    else:
        # weightes = (np.argsort(np.argsort(iou_class.cpu().numpy())[::-1]) / (K-1)) + 0.0001
        # weights = torch.tensor(weightes).float().cuda()
        return iou_class.cpu().numpy(), dice_class.cpu().numpy()



class DiceCEloss(nn.Module):
    # __name__ = 'Dice_CE_loss'

    def __init__(self, classes, samples, beta=1):
        super(DiceCEloss, self).__init__()
        self.classes = classes
        self.samples = samples
        self.beta = beta

    def forward(self, pr, gt):
        pred = torch.argmax(F.softmax(pr, 1, _stacklevel=5), dim=1)
        _, dice_class = cal_iou(pred, gt, self.classes, self.samples, mean_iou=False)

        weight = torch.as_tensor(1 - dice_class, device=torch.device('cuda'))
        celoss = F.cross_entropy(pr, gt, weight=weight, reduction='mean')
        dice_oppo = torch.mean(weight)
        return self.beta*dice_oppo + celoss



# ########################## train
def train(network, loader, criterion, optimizer, class_nums, sample_nums, loss_print):
    losses = 0.0
    mIOUs = 0.0
    network.train()
    for i, (img, target, _) in enumerate(loader):
        img = img.cuda()
        target = target.cuda().long()

        # forward + backward + optimize
        outputs = network(img)
        loss = criterion(outputs, target)

        # main_output, aux_output = network(img)
        # loss1 = criterion(main_output, target)
        # loss2 = criterion(aux_output, target)
        #loss = loss1 + 0.4 * loss2

        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()
        optimizer.step()

        ########## for mIOU and loss value
        pr = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)
        m, _ = cal_iou(pr, target, class_nums, sample_nums)
        mIOUs += m
        losses += loss.item()

        if not ((i+1) % loss_print):
            print('Iteration {:3d} loss {:.6f}'.format(i+1, loss.item()))
    return mIOUs, losses/(i + 1)



# ######################### validation
def validation(network, loader, class_nums, sample_nums, meanIOU = True):
    if meanIOU:
        mIOUs = 0.0
        mdices = 0.0
    else:
        mIOUs = np.zeros((class_nums))
        mdices = np.zeros((class_nums))

    network.eval()
    with torch.no_grad():
        for i, (img, target, _) in tqdm(enumerate(loader)):
            img = img.cuda()
            target = target.cuda().long()

            outputs = network(img)
            prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)

            m, d = cal_iou(prediction, target, class_nums, sample_nums, mean_iou =meanIOU)
            mIOUs += m
            mdices += d
    return mIOUs, mdices