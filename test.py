#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/10 10:25
# @Author : SuenDanny
# @Site : 
# @File : test.py
# @Software: PyCharm

import os

import torchvision.transforms
from PIL import Image

import torch
from torch import nn

# image_path = "./imgs/airplane20211010105957.png"
image_path = "./imgs/dog.PNG"
image = Image.open(image_path)
print(image)

image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("tudui_9_gpu",map_location=torch.device('cpu'))
print(model)

image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))







