#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/5 14:40
# @Author : SuenDanny
# @Site : 
# @File : nn_relu.py
# @Software: PyCharm



import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 0.5],
                     [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))



dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output

tudui = Tudui()

step = 0
writer = SummaryWriter("logs_relu")
for data in dataloader:
    imgs, targets = data
    writer.add_images(tag="input", img_tensor=imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step+1


writer.close()