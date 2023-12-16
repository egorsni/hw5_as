from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel
import torch
from torch import Tensor
from typing import Tuple

import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from hw_asr.model.sinc_conv import SincConv_fast
import numpy as np

import torch.nn as nn

class FMS(nn.Module):
    def __init__(self, fms_type, dim):
        super(FMS, self).__init__()
        self.fms_type = fms_type
        self.sig = nn.Sigmoid()
        self.lin = nn.Linear(dim, dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.pool(x).view(x.size(0), -1)
        y = self.lin(y)
        y = self.sig(y)
        y = y.view(x.size(0), x.size(1), -1)

        if self.fms_type == 'add':
            x = x + y
        elif self.fms_type == 'mul':
            x = x * y
        elif self.fms_type == 'comb':
            x = (x * y) + y
        return x

class Residual(nn.Module):
    def __init__(self, in_ch=128, out_ch=256, is_first=False, **batch):
        super(Residual, self).__init__()
        self.is_first = is_first
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.bn1 = nn.BatchNorm1d(num_features = in_ch)
        self.bn2 = nn.BatchNorm1d(num_features = out_ch)
        self.conv1 = nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = out_ch, out_channels = out_ch, kernel_size = 3, padding = 1, stride = 1)
        self.conv_final = nn.Conv1d(in_channels = in_ch, out_channels = out_ch, padding = 0, kernel_size = 1, stride = 1)
        self.maxpool = nn.MaxPool1d(3)
        self.fms = FMS(dim = out_ch, fms_type='comb')
        
        self.part = nn.Sequential(self.conv1, self.bn2, self.relu, self.conv2)
        
    def forward(self, x):
        x_copy = x
        x = self.relu(self.bn1(x))
        x = self.part(x)
        
        if self.in_ch != self.out_ch:
            x_copy = self.conv_final(x_copy)
            
        x = x + x_copy
        return self.fms(self.maxpool(x))
        

class RawNet2(BaseModel):
    def __init__(self, channels=[20,128], samp_len=64000, sinc_ks = 251, gru_hid=1024, len_classes=2, **batch): #channels = [ch1, ch2]
        super().__init__(channels, **batch)
        self.gamma = nn.Parameter(torch.ones(samp_len))
        self.beta = nn.Parameter(torch.zeros(samp_len))
        self.sinc_layer = SincConv_fast(out_channels = channels[0], kernel_size = 1024, min_band_hz=0, min_low_hz=0)
        self.bn = nn.BatchNorm1d(num_features = channels[0])
        self.relu = nn.LeakyReLU(negative_slope = 0.3)
        self.pool = nn.MaxPool1d(3)
        
        self.first_part = nn.Sequential(Residual(in_ch = channels[0], out_ch = channels[0], is_first=True),Residual(in_ch = channels[0], out_ch = channels[0]))
        self.second_part = nn.Sequential(*([Residual(in_ch = channels[0], out_ch = channels[1])] + [Residual(in_ch = channels[1], out_ch = channels[1]) for _ in range(3)]))
        
        self.bn_last = nn.BatchNorm1d(num_features = channels[1])
        self.gru = nn.GRU(input_size = channels[1],
            hidden_size = gru_hid,
            num_layers = 3,
            batch_first = True)
        self.lin1 = nn.Linear(in_features = gru_hid,
            out_features = gru_hid)
        self.lin2 = nn.Linear(in_features = gru_hid,
            out_features = len_classes,
            bias = True)
        

    def forward(self, x, is_test=False, **batch):
        x = self.gamma * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6) + self.beta #layer normalization
        x= x.view(x.shape[0],1,x.shape[1])
        x = self.sinc_layer(x)
        x = self.pool(x)
        x = self.relu(self.bn(x))
        x = self.first_part(x)
        x = self.second_part(x)
        x = self.relu(self.bn_last(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.lin1(x)
        if is_test:
            return x
        norm = x.norm(p=2,dim=1, keepdim=True) / 10.
        x = torch.div(x, norm)
        return self.lin2(x)