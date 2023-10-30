import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, num_channels, features_d):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(num_channels, features_d, kernel_size = 4, stride = 2, padding = 1),
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

class Generator(nn.Module):
    def __init__(self, latent_dim, features_g, num_channels):
        super().__init__()
        self.generator = nn.Sequential(
            self._block(latent_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, num_channels, kernel_size = 4, stride = 2, padding = 1),
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
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(discriminator, real, fake, device = "cpu"):

    BATCH_SIZE, Channels, Height, Width = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, Channels, Height, Width).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon) # 10 percent of the real image and 90 percent of the fake images
    mixed_scores = discriminator(interpolated_images)
    gradient = torch.autograd.grad(inputs = interpolated_images, outputs = mixed_scores, grad_outputs = torch.ones_like(mixed_scores), create_graph = True, retain_graph = True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters according to wgan paper
LR = 5e-5
LR_with_grad_pen = 1e-5
BATCH_SIZE = 64
EPOCHS = 5
BETAS = (0.5, 0.999)
IMAGE_SIZE = 64
IMAGE_SHAPE = 64
NUM_CHANNELS = 1
LATENT_DIM = 128
FEATURES_D = 64
FEATURES_G = 64
DISC_ITERATIONS = 5
CLIP_VAL = 0.01
LAMBDA_GRADIENT_PEN = 10
BETAS_GRADIENT_PENT = (0.0, 0.9)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.5 for _ in range(NUM_CHANNELS)], [0.5 for _ in range(NUM_CHANNELS)])
])

dataset = datasets.MNIST(root = 'dataset', train = True, transform = transforms, download = True)
data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

generator = Generator(LATENT_DIM, FEATURES_G, NUM_CHANNELS).to(device)
discriminator = Discriminator(NUM_CHANNELS, FEATURES_D).to(device)
initialize_weights(generator)
initialize_weights(discriminator)

gen_optimizer = optim.RMSprop(generator.parameters(), lr = LR)
disc_optimizer = optim.RMSprop(discriminator.parameters(), lr = LR)

fixed_noise = torch.randn(32, LATENT_DIM, 1, 1).to(device)
def set_writers(first, second, step = 0):
    writer_real = SummaryWriter(f"logs/{first}")
    writer_fake = SummaryWriter(f"logs/{second}")
    step = step
    return writer_real, writer_fake, step

writer_real, writer_fake, step = set_writers("real", "fake", step = 0)

generator.train()
discriminator.train()
for epoch in tqdm(range(5)):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.to(device)
        for _ in range(DISC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
            fake = generator(noise)
            critic_real = discriminator(real).reshape(-1)
            critic_fake = discriminator(fake).reshape(-1)
            # gp = gradient_penalty(discriminator, real, fake, device = device)
            loss_disc = -(torch.mean(critic_real) - torch.mean(critic_fake))
            # loss_disc = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GRADIENT_PEN * gp)
            loss_disc.backward(retain_graph = True)
            disc_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.data.clamp_(-CLIP_VAL, CLIP_VAL)


        output = discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward()
        gen_optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(data_loader)} Loss Discriminator: {loss_disc:.4f}, Loss Generator: {loss_gen:.4f}"   )

            with torch.no_grad():
                fake = generator(noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            generator.train()
            discriminator.train()