import torch
from torch import nn

class Discriminator(nn.Sequential):
    def __init__(self, image_shape, features_d):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(image_shape, features_d, kernel_size = 4, stride = 2, padding = 2),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.InstanceNorm2d(out_channels, affine = True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)
class Generator():
    pass

