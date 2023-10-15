import torch 
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, img_shape, features_d):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels = img_shape, out_channels = features_d, kernel_size = 4, stride = 2, padding = 1), # Input shape: N x channels_img x 64 x 64
            nn.LeakyReLU(0.2),
            self._block(in_channels = features_d, out_channels = features_d * 2, kernel_size = 4, stride = 2, padding = 1),
            self._block(in_channels = features_d * 2, out_channels = features_d * 4, kernel_size = 4, stride = 2, padding = 1),
            self._block(in_channels = features_d * 4, out_channels = features_d * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv2d(in_channels = features_d * 8, out_channels = 1, kernel_size = 4, stride = 2, padding = 0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),  # because of batchnorm this is unneceassary
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
        )
        
        
    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, features_g):
        super().__init__()
        self.generator = nn.Sequential(
            self._block(latent_dim, features_g * 16, kernel_size = 4, stride = 1, padding = 0),
            self._block(features_g * 16, features_g * 8, kernel_size = 4, stride = 2, padding = 1),
            self._block(features_g * 8, features_g * 4, kernel_size = 4, stride = 2, padding = 1),
            self._block(features_g * 4, features_g * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.ConvTranspose2d(features_g * 2, img_shape, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
        
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.generator(x)
    
    
def initialize_weights(model):
    for weight in model.modules():
        if isinstance(weight, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(weight.weight.data, 0.0, 0.02)


def test_model():
    N, in_channels, Height, Width = 8, 3, 64, 64
    latent_dim = 100
    x = torch.randn((N, in_channels, Height, Width))
    discriminator = Discriminator(in_channels, 8)
    initialize_weights(discriminator)
    assert discriminator(x).shape == (N, 1, 1, 1)
    y = torch.randn((N, latent_dim, 1, 1)) 
    generator = Generator(latent_dim, in_channels, 8)
    initialize_weights(generator)
    assert generator(y).shape == (N, in_channels, Height, Width)
    print("Succesfully finished!")
    
if __name__ == "__main__":
    test_model()            
