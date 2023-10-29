import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self, num_channels, features_d, num_classes, image_size):
        super().__init__()
        self.image_size = image_size
        self.discriminator = nn.Sequential(
            nn.Conv2d(num_channels + 1, features_d, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0)
        )
        self.embed = nn.Embedding(num_classes, image_size * image_size)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.InstanceNorm2d(out_channels, affine = True),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim = 1)
        return self.discriminator(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, features_g, num_classes, image_size, embed_size):
        super().__init__()
        self.image_size = image_size
        self.generator = nn.Sequential(
            self._block(latent_dim + embed_size, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, num_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim = 1)
        return self.net(x)