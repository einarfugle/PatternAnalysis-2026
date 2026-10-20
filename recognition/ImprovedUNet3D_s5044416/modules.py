"""Standard 2D U-Net baseline: four down/up levels and raw class logits.

Based on Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597, with
padded convolutions and bilinear upsampling to preserve the input image size.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two padded 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Down(nn.Module):
    """Halve the spatial dimensions, then apply a double convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Up(nn.Module):
    """Bilinear upsampling, concatenation with the skip, and double convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Matching the skip size also handles input dimensions not divisible by 16.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class OutConv(nn.Module):
    """Project decoder features to one logit per class using a 1x1 convolution."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """U-Net with bilinear upsampling and no normalization or residual blocks.

    Input is [B, in_channels, H, W], where H and W are at least 16.
    Output is [B, num_classes, H, W]. Apply softmax only in the loss or when
    probabilities are needed; argmax can operate directly on the output logits.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 32):
        super().__init__()
        if in_channels < 1 or num_classes < 2 or base_channels < 1:
            raise ValueError("Use positive channel counts and at least two classes.")
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.outc = OutConv(b, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or min(x.shape[-2:]) < 16:
            raise ValueError("Expected [B, C, H, W] with H and W at least 16.")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
