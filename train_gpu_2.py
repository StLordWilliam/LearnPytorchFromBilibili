#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/7 16:30
# @Author : SuenDanny
# @Site :
# @File : train.py
# @Software: PyCharm

#导入各种包
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from model import *
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
import time

start_time = time.time()

#定义训练设备
device = torch.device("cuda")

#准备训练数据集
train_data = torchvision.datasets.CIFAR10(root= './dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
#准备测试训练集
test_data = torchvision.datasets.CIFAR10(root= './dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                        download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train_data_len is: {}".format(train_data_size))
print("test_data_len is: {}".format(test_data_size))

#利用dataloader 加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

#创建网络模型
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


if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)


tudui = Tudui()
tudui = tudui.to(device)
# if torch.cuda.is_available():
#     tudui = tudui.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# loss_fn = loss_fn.cuda()

# 创建优化器,使用随机梯度下降
learing_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learing_rate)


#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-----------第{}轮训练开始------------".format(i+1))
    #训练步骤开始
    for data in train_data_loader:
        imgs, targets = data
        # imgs = imgs.cuda()
        # targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)

        output = tudui(imgs)
        loss = loss_fn(output,targets)

        #优化器优化模型，梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #梯度下降
        optimizer.step()
        total_train_step = total_train_step+1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{}, Loss : {} ".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            # imgs = imgs.cuda()
            # targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuray", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step+1

    torch.save(tudui, "tudui_{}".format(i))
    print("模型已保存")

writer.close()












