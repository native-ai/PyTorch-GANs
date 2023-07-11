import torch 
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels = channels_img, out_channels = features_d, kernel_size = 4, stride = 2, padding = 1), # Input shape: N x channels_img x 64 x 64
            nn.LeakyReLU(0.2),
            self._block(in_channels = features_d, out_channels = features_d * 2, kernel_size = 4, stride = 2, padding = 1),
            self._block(in_channels = features_d * 2, out_channels = features_d * 4, kernel_size = 4, stride = 2, padding = 1),
            self._block(in_channels = features_d * 4, out_channels = features_d * 8, kernel_size = 4, stride = 2, padding = 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),  # because of batchnorm this is unneceassary
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
        )
