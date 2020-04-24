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
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

normalize_tf_mdk = transforms.Normalize(
    mean=[0.485, 0.485, 0.406],
    std=[1/255, 1/255, 1/255])


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

test_normalize_tf_mdk = torchvision.datasets.ImageFolder(root="../test", transform=transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize
                normalize_tf_mdk
            ]))
normalize_tf_mdk_data_loader  = data.DataLoader(test_normalize_tf_mdk, batch_size=1, shuffle=True, num_workers=1) 

for i in normalize_tf_mdk_data_loader:
    img = i[0]
    break

print(img.shape)
# pdb.set_trace()
img_chw = img[0,:,:,:]
# writer.add_image('img_norm', img[0,:,:,:], 1)
writer.add_image('img_tf_mdk', img[0,:,:,:], 0)

un = UnNormalize(mean=[0.485, 0.456, 0.406],std=[1/255, 1/255, 1/255])
out = un(img_chw)
print(out.shape)
# writer.add_image('img_norm_out', out, 1)

writer.add_image('img_tf_mdk_out', out, 0)
writer.close()