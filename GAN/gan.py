import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_shape: int, output_shape: int = 1):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_shape, 128), 
            nn.LeakyReLU(0.1), # for GANs LeakyRelu is better choice
            nn.Linear(128, output_shape), # output_shape = 1 (fake or real)
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor):
        return self.discriminator(x)
    

class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_shape), # img_Shape will be equal to 784 (28 * 28 * 1)
            nn.Tanh() # we'll normalize mnist dataset and it's going to be between -1 and 1 either
        )

    def forward(self, x: torch.Tensor):
        return self.generator(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # best lr for Adam
latent_dim = 64 # or 128, 256 or even smaller
img_shape  = 784 # 28 * 28 * 1
BATCH_SIZE = 32
epochs = 50

discriminator = Discriminator(img_shape = img_shape).to(device) 
generator = Generator(latent_dim = latent_dim, img_shape = img_shape).to(device)
fixed_noise = torch.randn((BATCH_SIZE, latent_dim)).to(device) # for visualization purposes
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root = 'dataset',
    transform = transforms,
    download = True
)

data_loader = DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

optimizer_disc = torch.optim.Adam(params = discriminator.parameters(), lr = lr)
optimizer_gen = torch.optim.Adam(params = generator.parameters(), lr = lr)
criterion = nn.BCELoss() 

# for tensorboard
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0


for epoch in range(epochs):
    for batch, (real, _) in enumerate(data_loader):
        real = real.view((-1, 784)).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(noise)
        discriminator_real = discriminator(real).view(-1)
        loss_disc_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
        discriminator_fake = discriminator(fake)
        loss_disc_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward(retain_graph = True)
        optimizer_disc.step()

        #  Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch == 0:
            print(f"[Epoch {epoch} / {epochs}] \nDiscriminator Loss: {loss_disc:.4f}, Generator Loss: {loss_gen:.4f}")

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                img_grid_real = torchvision.utils.make_grid(data, normalize = True)
            
                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step = step)
                writer_real.add_image("MNIST Real Image", img_grid_real, global_step = step)
                step += 1