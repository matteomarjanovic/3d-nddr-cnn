from __future__ import print_function, division
from abc import ABC
import torch
import torch.nn as nn


class ConvLayer(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN_classification(nn.Module, ABC):
    def __init__(self, fil_num, drop_rate):
        super(_CNN_classification, self).__init__()

        self.block_c_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_c_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_c_3 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_c_4 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_c_5 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_c_6 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))

        self.dense_c = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(32 * fil_num, 32)
        )
        self.classify = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2), )

    def forward(self, x):
        x_c = self.block_c_1(x)
        x_c = self.block_c_2(x_c)
        x_c = self.block_c_3(x_c)
        x_c = self.block_c_4(x_c)
        x_c = self.block_c_5(x_c)
        x_c = self.block_c_6(x_c)
        batch_size = x.shape[0]
        x_c = x_c.view(batch_size, -1)
        x_c = self.dense_c(x_c)
        output_c = self.classify(x_c)
        return output_c