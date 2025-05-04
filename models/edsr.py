import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, scaling=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.scaling = scaling

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x * self.scaling + residual

class EDSR(nn.Module):
    def __init__(self, upscale_factor=10, num_blocks=8, channels=64):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.mid_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        residual = x
        x = self.res_blocks(x)
        x = self.mid_conv(x) + residual
        return self.upscale(x)