# --------------------------------------------------------
# Fast Rank R-CNN
# Copyright (c) 2018 Tsinghua
# Licensed under The MIT License [see LICENSE for details]
# Written by Liuhe
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils.config import cfg


class _rankNet(nn.Module):
    def __init__(self, classes, class_agnostic=False):
        super(_rankNet, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)
        self.fc_1 = nn.Linear(6000, 3000)
        self.fc_2 = nn.Linear(3000, 1000)
        self.fc_3 = nn.Linear(1000, 600)
        self.relu = nn.Relu(inplace=True)


    def forward(self, dets, if_train=False):
    	if dets.shape[0] == 0:
    		return []

    	x = self.fc_1()
    	x = self.relu(x)
    	x = self.fc_2(x)
    	x = self.relu(x)
    	x = self.fc_3(x)
    	x = self.relu(x)
        return x










