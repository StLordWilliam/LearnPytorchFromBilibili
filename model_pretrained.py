#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/6 9:41
# @Author : SuenDanny
# @Site : 
# @File : model_pretrained.py
# @Software: PyCharm

import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download= True,
#                                            transform= torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)


print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(in_features=4096,out_features=10)
print(vgg16_false)