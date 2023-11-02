from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
# import cv2

class SegData(Dataset):
    """ SegData class,  images and GT are put in the same folder
        root (string): Root directory of the segmentation Dataset.      eg. '../data/train/';
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, mode = 'train/', transform=None):
        self.root = root
        self.transform = transform
        self.masks = glob.glob(root + mode + '*_gt.png') # self.mask is the list of gt images name
        self.images = [(x[:-7] + '.png') for x in self.masks] # self.images is the list of original images name

    def __getitem__(self, index):
        """
            index (int): Index
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]) #.convert('RGB')
        img = np.array(img)
        gt = Image.open(self.masks[index])
        gt = np.array(gt)
        if self.transform is not None:
            img = self.transform(img)
        return img, gt, self.images[index]

    def __len__(self):
        return len(self.images)


class MajorRevison_SegData(Dataset):
    """ SegData class,  images and GT are put in the same folder
        root (string): Root directory of the segmentation Dataset.      eg. '../data/train/';
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, mode = 'train/', transform=None):
        self.root = root
        self.transform = transform
        self.masks = glob.glob(root + mode + '*_gt.npy') # self.mask is the list of gt images name
        self.images = [(x.replace('_gt.npy', '.npy')) for x in self.masks] # self.images is the list of original images name

    def __getitem__(self, index):
        """
            index (int): Index
            tuple: (image, target) where target is the image segmentation.
        """
        # img = Image.open(self.images[index]) #.convert('RGB')
        # img = np.array(img)
        img = np.load(self.images[index])
        # gt = Image.open(self.masks[index])
        # gt = np.array(gt)
        gt = np.load(self.masks[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, gt, self.images[index]

    def __len__(self):
        return len(self.images)