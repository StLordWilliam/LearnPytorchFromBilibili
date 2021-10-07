#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/5 16:23
# @Author : SuenDanny
# @Site : 
# @File : nn_seq.py
# @Software: PyCharm


import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear, MaxPool2d, Flatten, Sequential

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # #卷积部分
        # self.conv1 = Conv2d(in_channels=3,
        #                     out_channels=32,
        #                     kernel_size= 5,
        #                     padding=2
        #                     )
        # #池化部分
        # self.maxpool1 = MaxPool2d(2)
        # #再次卷积
        # self.conv2 = Conv2d(in_channels=32,
        #                     out_channels=32,
        #                     kernel_size= 5,
        #                     padding=2
        #                     )
        # #再次池化
        # self.maxpool2 = MaxPool2d(2)
        # # 再次卷积
        # self.conv3 =  Conv2d(in_channels=32,
        #                     out_channels=64,
        #                     kernel_size= 5,
        #                     padding=2
        #                     )
        # # 再次池化
        # self.maxpool3 = MaxPool2d(2)
        # #展平
        # self.flatten = Flatten()
        # #线性层
        # self.linear1 = Linear(in_features= 1024, out_features=64)
        # # 线性层
        # self.linear2 = Linear(in_features=64, out_features=10)

        self.model1 = Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size= 5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features= 1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

tudui = Tudui()
print(tudui)

#检测
input = torch.ones((64, 3, 32, 32))
output = tudui(input)

print(output.shape)


writer = SummaryWriter("log_seq")
writer.add_graph(tudui, input)
writer.close()
