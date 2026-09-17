import math

from torchvision.transforms.v2 import GaussianBlur

from models.modules.base import BaseCDOutProc, BaseCDDecoder


class GaussSmooth(BaseCDOutProc):
    name = "GSmooth"

    def __init__(self, pretraining: bool, sigma: float = 4):
        super().__init__(pretraining=pretraining)
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, image):
        return self.blur(image)

    def get_parameters(self, pretraining: bool) -> list[dict]:
        return []

class NoDec(BaseCDDecoder):
    """
    NoDec
    """

    name = ""

    def __init__(
        self,
        input_sizes: list[int],
        encoder_strides: list[int],
        out_channels: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes,
            encoder_strides=encoder_strides,
            out_channels=out_channels,
            *args,
            **kwargs,
        )

    def forward(self, x):
        return x[1][:, :1]

    def get_parameters(self, pretraining=False):
        return []
