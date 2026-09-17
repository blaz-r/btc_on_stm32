"""UNet model code.

Thanks to VitjanZ for original code: https://github.com/VitjanZ/DSR_anomaly_detection/blob/main/dsr_model.py#L221
"""

import torch.nn as nn
import torch

import torch.nn.functional as F
from sympy import factor

from models.modules.base import BaseCDDecoder


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class UnetDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1, first_up=2, first_channels=None):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        # TODO - test allign corners?
        first_channels = (
            first_channels if first_channels is not None else base_width * 4
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=first_up, mode="bilinear"),
            nn.Conv2d(first_channels, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )
        # cat with base*4
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )
        # cat with base*2
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, b1, b2, b3, b4):
        up1 = self.up1(b4)
        cat1 = torch.cat((up1, b3), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b2), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b1), dim=1)
        db3 = self.db3(cat3)

        out = self.fin_out(db3)
        return out


class UnetModel:
    name = "unet"

    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4)
        return output

    def get_parameters(self, pretrain=False) -> list[dict]:
        return [
            {
                "params": list(self.encoder.parameters())
                + list(self.decoder.parameters()),
            }
        ]


class Scale2UNet(UnetModel, BaseCDDecoder):
    """
    Scale feats to match output before passing through UNet.

    """

    name = "scale2unet"

    def __init__(self, out_size, in_dims, *args, **kwargs):
        super().__init__(in_channels=sum(in_dims), *args, **kwargs)
        self.out_size = out_size

    def upsample_cat(self, features: list[torch.Tensor]) -> torch.Tensor:
        # upscale all to size of first (largest)
        h = w = self.out_size

        feature_map = []
        # rescale to dims to match output
        for layer in features.values():
            # upscale all to 2x the size of the first (largest)
            # TODO - test align corners
            resized = F.interpolate(layer, size=(h, w), mode="bilinear")
            feature_map.append(resized)
        # channel-wise concat
        return torch.cat(feature_map, dim=1)

    def forward(self, x):
        # upsample all to output size then pass through UNet
        x = self.upsample_cat(x)
        output = super().forward(x)
        return output


class UNetFeatDec(BaseCDDecoder):
    """
    Unet decoder but take features from any encoder.

    """

    name = "unetDec"

    def __init__(
        self,
        input_sizes,
        encoder_strides,
        out_channels=1,
        out_size=None,
        upscale_to_original=False,
        *args,
        **kwargs,
    ):
        if len(input_sizes) != 4:
            raise ValueError("UnetDecoder currently requires 4layer input")

        # does the first feature map need to be upscaled?
        first_up = 1 if input_sizes[-2] == input_sizes[-1] else 2

        super().__init__(
            input_sizes=input_sizes,
            encoder_strides=encoder_strides,
            out_size=out_size,
            out_channels=out_channels,
            *args,
            **kwargs,
        )
        self.unet_dec = UnetDecoder(
            base_width=input_sizes[0],
            first_up=first_up,
            first_channels=input_sizes[
                -1
            ],  # custom first channel dim for first conv, which then outputs base*4
            out_channels=out_channels,
        )
        self.upscale_to_original = upscale_to_original
        if not upscale_to_original:
            self.name += "NS"

    def forward(self, x):
        vals = []
        if self.upscale_to_original:
            factor = self.encoder_strides[0]
            for val in x.values():
                vals.append(F.interpolate(val, scale_factor=factor, mode="bilinear"))
            return self.unet_dec.forward(*vals)
        else:
            output = self.unet_dec.forward(*x.values())
            return F.interpolate(output, size=self.out_size, mode="bilinear")

    def get_parameters(self, pretraining=False) -> list[dict]:
        return [
            {
                "params": self.parameters(),
            }
        ]
