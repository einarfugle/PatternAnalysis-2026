"""

Based on Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597, with
padded convolutions and bilinear upsampling to preserve the input image size.
"""

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# Standard 2D U-Net baseline
# 
# (Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597)                                                                
# --------------------------------------------------------------------------- #
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



# --------------------------------------------------------------------------- #
# 3D Improved U-Net
# 
# (Isensee et al. 2017, https://arxiv.org/abs/1802.10508)                                                                
# --------------------------------------------------------------------------- #

def conv_norm_act(in_channels: int, out_channels: int, k: int) -> nn.Sequential:
    """Conv -> InstanceNorm -> LeakyReLU."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(1e-2, inplace=True),
    )


class ContextModule(nn.Module):
    """Pre-activation residual block: (norm, act, 3x3x3 conv) twice, dropout between."""

    def __init__(self, channels: int, p_drop: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(p_drop),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class Down(nn.Module):
    """Stride-2 3x3x3 conv (instead of max pooling), then a context module."""

    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2,
                              padding=1, bias=False)
        self.context = ContextModule(out_channels, p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(self.conv(x))


class Up(nn.Module):
    """Upsample, 3x3x3 conv halving channels, concat skip, localization module.
    The last decoder level (final=True) uses a single 3x3x3 conv instead."""

    def __init__(self, in_channels: int, out_channels: int, final: bool = False):
        super().__init__()
        self.up_conv = conv_norm_act(in_channels, out_channels, 3)
        cat = 2 * out_channels
        if final:
            self.loc = conv_norm_act(cat, cat, 3)
        else:
            self.loc = nn.Sequential(conv_norm_act(cat, cat, 3),
                                     conv_norm_act(cat, out_channels, 1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Matching the skip size handles volumes not divisible by 16.
        x = F.interpolate(x, size=skip.shape[-3:], mode="nearest")
        return self.loc(torch.cat((skip, self.up_conv(x)), dim=1))


class OutConv(nn.Module):
    """1x1x1 conv to one logit per class."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ImprovedUNet3D(nn.Module):
    """Improved U-Net (Isensee et al. 2017) with deep supervision.

    Input is [B, in_channels, D, H, W]. Output is [B, num_classes, D, H, W] logits.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 16, p_drop: float = 0.3):
        super().__init__()
        if in_channels < 1 or num_classes < 2 or base_channels < 1:
            raise ValueError("Use positive channel counts and at least two classes.")
        b = base_channels
        self.inc = nn.Sequential(
            nn.Conv3d(in_channels, b, kernel_size=3, padding=1, bias=False),
            ContextModule(b, p_drop),
        )
        self.down1 = Down(b, b * 2, p_drop)
        self.down2 = Down(b * 2, b * 4, p_drop)
        self.down3 = Down(b * 4, b * 8, p_drop)
        self.down4 = Down(b * 8, b * 16, p_drop)
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b, final=True)
        # Deep supervision heads
        self.seg2 = OutConv(b * 4, num_classes)
        self.seg3 = OutConv(b * 2, num_classes)
        self.outc = OutConv(b * 2, num_classes)

    @staticmethod
    def _upsum(s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(s, size=ref.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected [B, C, D, H, W].")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        s = self.seg2(x)
        x = self.up3(x, x2)
        s = self._upsum(s, x) + self.seg3(x)
        x = self.up4(x, x1)
        return self._upsum(s, x) + self.outc(x)