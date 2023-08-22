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
            self._block(in_channels = features_d * 4, out_channels = features_d * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),  # because of batchnorm this is unneceassary
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
        )

class Generator(nn.Module):
    def __init__(self, latent_dim, channels_img, featurs_g):
        super().__init__()
        self.generator = nn.Sequential(
            self._block(in_channels = latent_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, fatures_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channls, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
 
