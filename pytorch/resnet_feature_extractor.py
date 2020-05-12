
import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np
import time
start_time = time.time()

subset = "mask"
train_mask_dir = "../mask/mask_classifier_dataset_including_blur_imgs/{}/{}".format("train", subset)
subset = "nomask"
train_nomask_dir = "../mask/mask_classifier_dataset_including_blur_imgs/{}/{}".format("train", subset)

resnet_based = torchvision.models.resnet18(pretrained=True)

transform = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.ToTensor(),])

train_mask = torchvision.datasets.ImageFolder(train_mask_dir, transform=transform)
train_nomask = torchvision.datasets.ImageFolder(train_nomask_dir, transform=transform)


t_mask = torch.utils.data.DataLoader(train_mask, batch_size=1 ,shuffle = False)
t_nomask = torch.utils.data.DataLoader(train_nomask, batch_size=1 ,shuffle = False)

model = resnet_based.eval()

t_nomask_blur_feat = np.zeros((1,1000))
t_nomask_noblur_feat = np.zeros((1,1000))
for batch_idx, (data, target) in enumerate(t_nomask):
    print(batch_idx, target)
    # print(target == 1)
    # print(data.shape)
    output = model(data)
    # print(output.shape)
    # exit()
    # t_mask_feat.append(output, dim=0)
    if target ==0 :
        t_nomask_blur_feat = np.append(t_nomask_blur_feat,output.detach().numpy(), axis=0)
    if target ==1 :
        t_nomask_noblur_feat = np.append(t_nomask_noblur_feat,output.detach().numpy(), axis=0)
    # print(t_nomask_feat.shape)
    # if batch_idx ==5:
    #     break
np.save("t_nomask_blur_feat.npy",t_nomask_blur_feat[1:])
np.save("t_nomask_noblur_feat.npy",t_nomask_noblur_feat[1:])

time_taken = time.time() - start_time
print("total_time_taken: {} min {} secs".format(time_taken//60,time_taken%60))