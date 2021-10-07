#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/5 11:11
# @Author : SuenDanny
# @Site : 
# @File : nn_module.py
# @Software: PyCharm
import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input+1
        return output



tudui = Tudui()

x = torch.tensor(1.0)
output = tudui(x)
print(output)