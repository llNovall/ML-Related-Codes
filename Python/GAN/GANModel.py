from torch import nn
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(in_features=25, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        logits = self.model(img)

        return logits


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 16, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size, 3, 4, 2, 1, bias=False),

            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class GAN(LightningModule):
    def __init__(
            self,
            d_hidden_size: int = 32,
            g_latent_dim: int = 100,
            g_hidden_size=32,
            learning_rate: float = 0.001,
            beta: tuple[float, float] = (0.5, 0.999),
            weight_decay: float = 0.0,
            sample_interval: int = 50
    ):
        super().__init__()
        self.d_hidden_size = d_hidden_size
        self.g_latent_dim = g_latent_dim
        self.g_hidden_size = g_hidden_size
        self.sample_interval = sample_interval
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=g_latent_dim, hidden_size=g_hidden_size)
        self.discriminator = Discriminator(hidden_size=d_hidden_size)

        self.g_optimizer = None
        self.d_optimizer = None

        self.constant_noise = torch.randn(36, self.g_latent_dim, 1, 1, device=torch.device('cuda'))

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def initialize_weights(self):
        self.generator.model.apply(fn=self.weights_init)
        self.discriminator.model.apply(fn=self.weights_init)

    def weights_init(self, m):

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            print(f'Appling weights to Conv layer..')
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            print(f'Appling weights to BatchNorm layer..')
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        z = torch.randn(imgs.shape[0], self.g_latent_dim, 1, 1)
        z = z.type_as(imgs)

        if optimizer_idx == 0:

            generated_imgs = self(z)

            if self.global_step % self.sample_interval == 0:
                print(f'Saving Sample..')

                sample_imgs = generated_imgs[:64]
                grid = torchvision.utils.make_grid(sample_imgs, nrow=8)
                save_image(grid, fp=f'Samples/sample_e_{self.current_epoch}_{self.global_step}.png')

            y = torch.ones(imgs.size(0), 1)
            y = y.type_as(imgs)

            y_hat = self.discriminator(generated_imgs)
            g_loss = self.adversarial_loss(y_hat, y)

            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)

            return g_loss

        if optimizer_idx == 1:

            y_real = torch.ones(imgs.size(0), 1)
            y_real = y_real.type_as(imgs)
            y_real_hat = self.discriminator(imgs)

            real_loss = self.adversarial_loss(y_real_hat, y_real)

            y_fake = torch.zeros(imgs.size(0), 1)
            y_fake = y_fake.type_as(imgs)
            y_fake_hat = self.discriminator(self(z).detach())

            fake_loss = self.adversarial_loss(y_fake_hat, y_fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)

            return d_loss

    def configure_optimizers(self):

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                            betas=self.beta, weight_decay=self.weight_decay)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                            betas=self.beta, weight_decay=self.weight_decay)
        return [self.g_optimizer, self.d_optimizer], []
