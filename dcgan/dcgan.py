import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
            
            
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_SHAPE = 1
LATENT_DIM = 100
EPOCHS = 100
FEATURES_D = 64
FEATURES_G = 64

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.5 for _ in range(IMG_SHAPE)], [0.5 for _ in range(IMG_SHAPE)])
])

dataset = datasets.MNIST(root = "dataset", train = True, transform = transforms, download = True)
data_loader = DataLoader(dataset, BATCH_SIZE, shuffle = True)

generator = Generator(LATENT_DIM, IMG_SHAPE, FEATURES_G).to(device)
discriminator = Discriminator(IMG_SHAPE, FEATURES_D).to(device)
initialize_weights(generator)
initialize_weights(discriminator)
optimizer_gen = optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
optimizer_disc = optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, LATENT_DIM, 1, 1).to(device)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in tqdm(range(EPOCHS)):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
        fake = generator(noise)

        discriminator_real = discriminator(real).view(-1)
        loss_disc_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
        discriminator_fake = discriminator(fake)
        loss_disc_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward(retain_graph = True)
        optimizer_disc.step()

        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(f"[Epoch {epoch} / {EPOCHS} \nDiscriminator Loss: {loss_disc:.4f}, Generator Loss: {loss_gen:.4f}")

            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(real, normalize = True)
                img_grid_real = torchvision.utils.make_grid(fake, normalize = True)

                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step = step)
                writer_real.add_image("MNIST Real Images", img_grid_real, global_step = step)

                step += 1

# def test_model():
#     N, in_channels, Height, Width = 8, 3, 64, 64
#     latent_dim = 100
#     x = torch.randn((N, in_channels, Height, Width))
#     discriminator = Discriminator(in_channels, 8)
#     initialize_weights(discriminator)
#     assert discriminator(x).shape == (N, 1, 1, 1)
#     y = torch.randn((N, latent_dim, 1, 1)) 
#     generator = Generator(latent_dim, in_channels, 8)
#     initialize_weights(generator)
#     assert generator(y).shape == (N, in_channels, Height, Width)
#     print("Succesfully finished!")
    
# if __name__ == "__main__":
#     test_model()            


