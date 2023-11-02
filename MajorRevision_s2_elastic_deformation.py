import numpy as np
import os
import glob
from PIL import Image
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# import matplotlib.pyplot as plt


def single_elastic_trans(image, mask, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    image = np.concatenate((image[..., None], mask[..., None]), axis=2)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)#
    image = cv2.warpAffine(image, M, shape_size[::-1], flags = cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    image = map_coordinates(image, indices, order=0, mode='reflect').reshape(shape)
    return image[:, :, 0], image[:, :, 1]


# # Function to distort image
def multi_elastic_trans(image, mask, n_slices, alpha, sigma, alpha_affine, random_state=None):
    image = np.concatenate((image, mask[..., None]), axis=2)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],#raw point
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)#output point
    M = cv2.getAffineTransform(pts1, pts2)# affine matrix
    image = cv2.warpAffine(image, M, shape_size[::-1], flags = cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    # mask = cv2.warpAffine(mask, M, shape_size[::-1], flags = cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    image = map_coordinates(image, indices, order=0, mode='reflect').reshape(shape)
    # mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
    return image[..., :n_slices], image[..., n_slices]


def run(img_path, savepath, times=1, nums_slices=3):
    masklist = [f.split('/')[-1] for f in glob.glob(img_path+'*_gt.npy')]

    for mask_name in masklist:
        print(masklist.index(mask_name), mask_name)

        img_name = mask_name.replace('_gt.npy', '.npy')

        img = np.load(os.path.join(img_path, img_name))
        mask = np.load(os.path.join(img_path, mask_name)).astype(np.uint8)

        for nums in range(times):
            _size_ = img.shape[1]
            if nums_slices > 1:
                img, mask = multi_elastic_trans(img, mask, nums_slices, _size_ * 2, _size_ * 0.08, _size_ * 0.08)
            else:
                img, mask = single_elastic_trans(img, mask, _size_ * 2, _size_ * 0.08, _size_ * 0.08)

            np.save(os.path.join(savepath, str(nums) + '_' + img_name), img)
            np.save(os.path.join(savepath, str(nums) + '_' + img_name.replace('.npy', '_gt.npy')), mask)


if __name__ == "__main__":
    # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates

    n = 7
    img_path = '/home/cyyan/projects/ISICDM2019/data/' + str(n) + 'slices/train/'
    img2dir = '/home/cyyan/projects/ISICDM2019/data/' + str(n) + 'slices/trainAUG/'
    if not os.path.exists(img2dir):
        os.mkdir(img2dir)

    run(img_path, img2dir, times=20, nums_slices=n)