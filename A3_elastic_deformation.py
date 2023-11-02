import numpy as np
import os
import glob
from PIL import Image
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# import matplotlib.pyplot as plt


img_path = '/home/cyyan/projects/ISICDM2019/data/25D/train/'
img2dir = '/home/cyyan/projects/ISICDM2019/data/25D/trainAUG/'
N = 20 #numbers for generating


def main():
    masklist = [f.split('/')[-1] for f in glob.glob(img_path+'*gt.png')]

    for mask_name in masklist:
        print(masklist.index(mask_name), mask_name)

        img_name = mask_name[:-7] + '.png'

        img = np.array(Image.open(img_path + '/' + img_name))
        mask = np.array(Image.open(img_path + '/' + mask_name)).astype(np.uint8)
        # img = cv2.imread(img_path + '/' + img_name, cv2.CAP_MODE_GRAY)

        for nums in range(N):
            per_aug(img, mask, img_name, mask_name, nums)
        print('success')
    print('All done')


def per_aug(img, mask, img_name, mask_name, add):
    """
    :param img: raw img   :param mask: groundtruth   :param img_name: the name of img  :param mask_name: the name of gt
    :param add: increased name of output file    :param channel: the number of groundtruth's channel(1:binary;3:rgb)
    :return: none
    """
    img, mask = elastic_transform_3(img, mask, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
    # img, mask = elastic_transform_1(img, mask, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)

    # if not os.path.isdir(img2dir):
    #     os.mkdir(img2dir)
    Image.fromarray(img).save(img2dir + str(add) + '_' + img_name)
    Image.fromarray(mask).save(img2dir + str(add) + '_' + mask_name)
    # cv2.imwrite(img2dir + str(add) + '_' + mask_name, mask)



def elastic_transform_1(image, mask, alpha, sigma, alpha_affine, random_state=None):
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
def elastic_transform_3(image, mask, alpha, sigma, alpha_affine, random_state=None):
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
    return image[..., :3], image[..., 3]



# you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
if __name__ == "__main__":
    main()