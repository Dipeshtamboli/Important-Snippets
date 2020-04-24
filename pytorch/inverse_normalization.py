'''
# This code is for loading a single image, doing preprocessing(i.e. normalizarion)
# and then inverse normalization to get the singlw image
# This also supports visualization in tensorboard
'''

import os
import torch
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import transforms
import pdb
from torch.utils.tensorboard import SummaryWriter

from net import UnNormalize

writer = SummaryWriter("img")

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


test_normalize_simple = torchvision.datasets.ImageFolder(root="../test", transform=transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                # normalize_tf_mdk
            ]))
normalize_simple_data_loader  = data.DataLoader(test_normalize_simple, batch_size=1, shuffle=True, num_workers=1) 

for i in normalize_simple_data_loader:
    img = i[0]
    break

print(img.shape)
# pdb.set_trace()
img_chw = img[0,:,:,:]
writer.add_image('img_norm', img[0,:,:,:], 0)

un = UnNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
out = un(img_chw)
print(out.shape)
writer.add_image('img_norm_out', out, 0)


writer.close()