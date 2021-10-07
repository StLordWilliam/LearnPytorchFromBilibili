#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/6 21:52
# @Author : SuenDanny
# @Site : 
# @File : model_load.py
# @Software: PyCharm

import torch

#方式1, 加载模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)


vgg16 = torchvision.models.vgg16(pretrained= False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(model)

#陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x




# tudui = Tudui()
# torch.save(tudui, "tudui_method1.pth")


model = torch.load("tudui_method1.pth")
print(model)