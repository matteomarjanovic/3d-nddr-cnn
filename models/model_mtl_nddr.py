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
    

class NDDRLayer(nn.Module, ABC):
    def __init__(self, out_channels, init_weights_type, init_weights):
        super().__init__()
        self.conv1 = nn.Conv3d(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channels)

        # weights initialization
        if init_weights_type == 'diagonal':
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1, 1))

    
    def forward(self, features):
        x = torch.cat(features, dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.BN1(x1)
        x2 = self.BN2(x2)
        return x1, x2


class _CNN(nn.Module, ABC):
    def __init__(self, fil_num, drop_rate, nddr_lr_mul=1, init_weights_type='random', init_weights=None):
        super(_CNN, self).__init__()
        self.nddr_lr_mul = nddr_lr_mul

        self.clf_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()
        self.nddr_layers = nn.ModuleList()

        self.block_c_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_c_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_c_3 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_c_4 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_c_5 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_c_6 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))
        self.block_r_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_r_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_r_3 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_r_4 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_r_5 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_r_6 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))
        self.nddr_1 = NDDRLayer(fil_num, init_weights_type, init_weights)
        self.nddr_2 = NDDRLayer(2 * fil_num, init_weights_type, init_weights)
        self.nddr_3 = NDDRLayer(2 * fil_num, init_weights_type, init_weights)
        self.nddr_4 = NDDRLayer(2 * fil_num, init_weights_type, init_weights)
        self.nddr_5 = NDDRLayer(2 * fil_num, init_weights_type, init_weights)

        self.dense_c = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(32 * fil_num, 32)
        )
        self.classify = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2),
        )
        self.dense_r = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(32 * fil_num, 32)
        )
        self.regress = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1),
        )

        self.clf_layers.append(self.block_c_1)
        self.clf_layers.append(self.block_c_2)
        self.clf_layers.append(self.block_c_3)
        self.clf_layers.append(self.block_c_4)
        self.clf_layers.append(self.block_c_5)
        self.clf_layers.append(self.block_c_6)
        self.clf_layers.append(self.dense_c)
        self.clf_layers.append(self.classify)
        self.reg_layers.append(self.block_r_1)
        self.reg_layers.append(self.block_r_2)
        self.reg_layers.append(self.block_r_3)
        self.reg_layers.append(self.block_r_4)
        self.reg_layers.append(self.block_r_5)
        self.reg_layers.append(self.block_r_6)
        self.reg_layers.append(self.dense_r)
        self.reg_layers.append(self.regress)
        self.nddr_layers.append(self.nddr_1)
        self.nddr_layers.append(self.nddr_2)
        self.nddr_layers.append(self.nddr_3)
        self.nddr_layers.append(self.nddr_4)
        self.nddr_layers.append(self.nddr_5)

    def forward(self, x):
        x_c = self.block_c_1(x)
        x_r = self.block_r_1(x)
        x_c, x_r = self.nddr_1([x_c, x_r])

        x_c = self.block_c_2(x_c)
        x_r = self.block_r_2(x_r)
        x_c, x_r = self.nddr_2([x_c, x_r])

        x_c = self.block_c_3(x_c)
        x_r = self.block_r_3(x_r)
        x_c, x_r = self.nddr_3([x_c, x_r])

        x_c = self.block_c_4(x_c)
        x_r = self.block_r_4(x_r)
        x_c, x_r = self.nddr_4([x_c, x_r])

        x_c = self.block_c_5(x_c)
        x_r = self.block_r_5(x_r)
        x_c, x_r = self.nddr_5([x_c, x_r])

        x_c = self.block_c_6(x_c)
        x_r = self.block_r_6(x_r)
        
        batch_size = x.shape[0]
        x_c = x_c.view(batch_size, -1)
        x_c = self.dense_c(x_c)
        output_c = self.classify(x_c)

        x_r = x_r.view(batch_size, -1)
        x_r = self.dense_r(x_r)
        output_r = self.regress(x_r)

        return output_c, output_r
    
    def configure_optimizers(self, learning_rate):
        # here the specific learning rate for nddr layer is set
        optimizer = torch.optim.AdamW(
            [
                {'params': self.clf_layers.parameters()},
                {'params': self.reg_layers.parameters()},
                {'params': self.nddr_layers.parameters(), 'lr': learning_rate*self.nddr_lr_mul}
            ],
            lr=learning_rate,
            weight_decay=learning_rate*1e-2
        )

        return optimizer