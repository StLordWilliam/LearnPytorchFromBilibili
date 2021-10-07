#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/6 21:47
# @Author : SuenDanny
# @Site : 
# @File : model_save.py
# @Software: PyCharm
import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained= False)

#保存方式1， 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

#保存方式2， 模型参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")



