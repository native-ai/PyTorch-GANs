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
        return self.generator(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(discriminator, labels, real, fake, device = "cpu"):

    BATCH_SIZE, Channels, Height, Width = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, Channels, Height, Width).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon) # 10 percent of the real image and 90 percent of the fake images
    mixed_scores = discriminator(interpolated_images, labels)
    gradient = torch.autograd.grad(inputs = interpolated_images, outputs = mixed_scores, grad_outputs = torch.ones_like(mixed_scores), create_graph = True, retain_graph = True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 10
GEN_EMBED = 100
LATENT_DIM = 100
EPOCHS = 100
FEATURES_D = 16
FEATURES_G = 16
DISC_ITERATIONS = 5
LAMBDA_GRADIENT_PEN = 10
BETAS = (0.0, 0.9)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
transforms.Normalize([0.5 for _ in range(NUM_CHANNELS)], [0.5 for _ in range(NUM_CHANNELS)])
])

dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
generator = Generator(LATENT_DIM, NUM_CHANNELS, FEATURES_G, NUM_CLASSES, IMAGE_SIZE, GEN_EMBED).to(device)
discriminator = Discriminator(NUM_CHANNELS, FEATURES_D, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(generator)
initialize_weights(discriminator)

gen_optimizer = optim.Adam(generator.parameters(), lr = LR, betas = BETAS)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = LR, betas = BETAS)

fixed_noise = torch.randn(32, LATENT_DIM, 1, 1).to(device)
def set_writers(first, second, step = 0):
    writer_real = SummaryWriter(f"logs/{first}")
    writer_fake = SummaryWriter(f"logs/{second}")
    step = step
    return writer_real, writer_fake, step

writer_real, writer_fake, step = set_writers("real", "fake", step = 0)

generator.train()
discriminator.train()

for epoch in range(EPOCHS):
    for batch_idx, (real, labels) in enumerate(data_loader):
        real = real.to(device)
        labels = labels.to(device)
        for _ in range(DISC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
            fake = generator(noise, labels)
            critic_real = discriminator(real, labels).reshape(-1)
            critic_fake = discriminator(fake, labels).reshape(-1)
            gp = gradient_penalty(discriminator, labels, real, fake, device = device)
            # loss_disc = -(torch.mean(critic_real) - torch.mean(critic_fake))
            loss_disc = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GRADIENT_PEN * gp)
            loss_disc.backward(retain_graph = True)
            disc_optimizer.step()

        output = discriminator(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward()
        gen_optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(data_loader)} Loss Discriminator: {loss_disc:.4f}, Loss Generator: {loss_gen:.4f}")

            with torch.no_grad():
                fake = generator(noise, labels)
                img_grid_real = torchvision.utils.make_grid(real[:64], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize = True)

                writer_real.add_image("Real", img_grid_real, global_step = step)
                writer_fake.add_image("Fake", img_grid_fake, global_step = step)

            step += 1
            generator.train()
            discriminator.train()