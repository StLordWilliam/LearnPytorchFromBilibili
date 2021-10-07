#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/5 16:04
# @Author : SuenDanny
# @Site : 
# @File : nn_linear.py
# @Software: PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train= False, transform= torchvision.transforms.ToTensor(),
                                       download= True)

dataloader = DataLoader(dataset, batch_size= 64, drop_last= True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self,input):
        output = self.linear1(input)
        return output

tudui = Tudui()


for data in dataloader:
    imgs,targets =data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)