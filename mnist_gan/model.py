import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1440, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class Generator(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 49*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)

        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.leaky_relu(x)

        return self.conv(x)


class GAN(pl.LightningModule):

    def __init__(self, latent_dim: int = 100, lr: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.criterion = self.adversarial_loss
    
    def sample_z(self, n) -> torch.Tensor:
        sample = torch.randn((n, self.hparams.latent_dim), device=self.device)
        return sample
    
    def sample_G(self, n) -> torch.Tensor:
        z = self.sample_z(n)
        return self.generator(z)
    
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        y = torch.ones((batch_size, 1), device=self.device)
        y = y.type_as(real_imgs)
        g_loss = self.criterion(y_hat, y)

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)
        
        d_x = self.discriminator(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.discriminator(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.discriminator(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        d_z = self.discriminator(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({
            "g_loss": errG,
            "d_loss": errD,
        }, prog_bar=True)

        
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d]
    
    def plot_imgs(self):
        z = self.sample_z(6)
        sample_imgs = self(z).cpu()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.imshow(
                sample_imgs.detach()[i, 0, :, :], cmap="gray_r", interpolation="none"
            )
            plt.title(f"generated data at epoch {self.current_epoch}")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        plt.show()
    