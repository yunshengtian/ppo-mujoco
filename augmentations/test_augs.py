import numpy as np

import augmentations.rad as rad
import torch
import os
from os import listdir
from os.path import isfile, join
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_imgs(x, max_display=12):
    grid = make_grid(torch.from_numpy(x[:max_display]), 4).permute(1, 2, 0).cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)


def show_stacked_imgs(x, max_display=12):
    fig = plt.figure(figsize=(12, 12))
    stack = 3

    for i in range(1, stack + 1):
        grid = make_grid(torch.from_numpy(x[:max_display, (i - 1) * 3:i * 3, ...]), 4).permute(1, 2, 0).cpu().numpy()

        fig.add_subplot(1, stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title('frame ' + str(i))
        plt.imshow(grid)

if __name__ == "__main__":
    tnsrs_list = list()
    folder = '../mujoco_samples/'
    sample_imgs_files = [os.path.join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    for fp in sample_imgs_files:
        img = Image.open(fp=fp)
        img_tnsr = transforms.ToTensor()(img)
        img_np = img_tnsr.numpy()
        img_np = np.transpose(img_np, (1,2,0))
        tnsrs_list.append(img_tnsr)

    cat_tnsrs = torch.unsqueeze(torch.concat(tnsrs_list, dim=0), dim=0)

    show_stacked_imgs(cat_tnsrs.numpy())
    aug_tnsrs = rad.random_convolution(cat_tnsrs)
    show_stacked_imgs(aug_tnsrs.numpy())
    plt.show()

