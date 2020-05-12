from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Vgg19edited(nn.Module):

    def __init__(self, num_classes=14):

        super(Vgg19edited, self).__init__()

        vgg = models.vgg19(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False

        self.part1 = nn.Sequential(
            *list(vgg.features.children())[:-5]
        )
        self.part2 = nn.Sequential(
            *list(vgg.features.children())[-5:]
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.part1(x)
        x = self.part2(x)
        # x = torch.tensor([[[1, 2],[ 3, 4]],[[5, 6],[ 7, 8]]],dtype=torch.float64)
        # print(x.shape)
        # x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == '__main__':

    model = Vgg19edited()
    print(model)

    input = Variable(torch.FloatTensor(1, 3, 512, 512))
    output = model(input)
    print('net output size:')
    print(output.shape)

