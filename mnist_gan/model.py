import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple = (28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class Generator(nn.Module):

    def __init__(self, latent_dim: int, img_shape: tuple = (28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class GAN(pl.LightningModule):

    def __init__(
        self,
        channels: int = 1,
        width: int = 28,
        height: int = 28,
        latent_dim: int = 100,
        lr: float = 0.001,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape
        )
        self.discriminator = Discriminator(img_shape=data_shape)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
    
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        """
        The advaserial loss is the loss function that calculates the
        distance between the GAN distribution of the generated images
        and the real images.
        """
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch: tuple):
        imgs, _ = batch
        
        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Train Generator
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train Discriminator
        self.toggle_optimizer(optimizer_d)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2)
        )
        return [opt_g, opt_d]

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(
            "generated_images",
            grid,
            self.current_epoch
        )
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
