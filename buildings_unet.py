import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor


class BuildingsModel(nn.Module):

    def __init__(self, in_channels: int,
                 ):
        super().__init__()
        self._1 = DownSamplingBlock(4, 8)
        self._2 = DownSamplingBlock(self._1.conv2.out_channels, 2)
        self._3 = DownSamplingBlock(self._2.conv2.out_channels, 2)
        self._4 = DownSamplingBlock(self._3.conv2.out_channels, 2)

    def forward(self, x):
        pass


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels: int,
                 channel_up_factor: int=1,
                 max_pooling: bool=True):
        super().__init__()
        out_channels = in_channels*channel_up_factor
        self.max_pooling = max_pooling
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu2 = nn.ReLU()
        if self.max_pooling:
            self.max = nn.MaxPool2d(kernel_size=2,
                                    stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        skip_connection = x
        if self.max_pooling:
            x = self.max(x)
            return x, skip_connection
        return x
        
            
class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int,
                 channel_down_factor: int,
                 skip_channels: int):
        super().__init__()
        out_channels = in_channels // channel_down_factor
        self.transpose2d = nn.ConvTranspose2d(in_channels,
                                              out_channels,
                                              kernel_size=3,
                                              stride=2,
                                              output_padding=1)
        self.conv1 = nn.Conv2d(out_channels+skip_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.activation = nn.ReLU()
        
    def forward(self, x: Tensor, skip_connection: Tensor):
        x = self.tconv(x)
        x = torch.cat([x, skip_connection], -3)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
        