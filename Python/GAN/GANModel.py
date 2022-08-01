from torch import nn
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from GANConfig import ModelParams


class Discriminator(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, params.d_hidden_size, kernel_size=4, stride=2, padding=1, bias=False),  # 32 x 32
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params.d_hidden_size, params.d_hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # 16 x 16
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params.d_hidden_size * 2, params.d_hidden_size * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),  # 8 x 8
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params.d_hidden_size * 4, params.d_hidden_size * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),  # 4 x 4
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params.d_hidden_size * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 1 x 1

            nn.Flatten(),
            nn.Linear(in_features=4, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        logits = self.model(img)

        return logits


class Generator(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(params.g_latent_size, params.g_hidden_size * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),  # 4 x 4
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(params.g_hidden_size * 8, params.g_hidden_size * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),  # 8 x 8
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(params.g_hidden_size * 4, params.g_hidden_size * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),  # 16 x 16
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(params.g_hidden_size * 2, params.g_hidden_size, kernel_size=4, stride=2, padding=1,
                               bias=False),  # 32 x 32
            nn.GroupNorm(num_groups=32, num_channels=params.g_hidden_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(params.g_hidden_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # 64 x 64
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class GAN(LightningModule):
    def __init__(
            self, params: ModelParams
    ):
        super().__init__()

        self.params = params

        self.save_hyperparameters()

        self.generator = Generator(params)
        self.discriminator = Discriminator(params)

        self.g_optimizer = None
        self.d_optimizer = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        loss = F.binary_cross_entropy(y_hat, y)
        return loss

    def initialize_weights(self):
        self.generator.model.apply(fn=self.weights_init)
        self.discriminator.model.apply(fn=self.weights_init)

    def weights_init(self, module):

        if isinstance(module, nn.ConvTranspose2d):
            print("Assigning weights for ConvTranspose2d layer.")
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.Conv2d):
            print("Assigning weights for ConvTranspose2d layer.")
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            print("Assigning weights for BatchNorm")
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        z = torch.randn(imgs.shape[0], self.params.g_latent_size, 1, 1)
        z = z.type_as(imgs)

        if optimizer_idx == 0:

            generated_imgs = self(z)

            y = torch.ones(imgs.size(0), 1)
            y = y.type_as(imgs)

            y_hat = self.discriminator(generated_imgs)
            g_loss = self.adversarial_loss(y_hat, y)

            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)

            if self.global_step % self.params.training_sample_interval == 0:
                print(f'Saving Sample..')

                sample_imgs = generated_imgs[:25]
                grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
                save_image(grid, fp=f'Samples/sample_e_{self.current_epoch}_{self.global_step}.png')

            return g_loss

        if optimizer_idx == 1:
            y_real = torch.ones(imgs.size(0), 1)
            y_real = y_real.type_as(imgs)
            y_real_hat = self.discriminator(imgs)
            y_real_hat = torch.clamp(y_real_hat, min=0.0, max=0.9)
            real_loss = self.adversarial_loss(y_real_hat, y_real)

            y_fake = torch.zeros(imgs.size(0), 1)
            y_fake = y_fake.type_as(imgs)
            y_fake_hat = self.discriminator(self(z).detach())

            fake_loss = self.adversarial_loss(y_fake_hat, y_fake)

            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)

            return d_loss

    def configure_optimizers(self):

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.params.opt_learning_rate,
                                            betas=self.params.opt_betas, weight_decay=self.params.opt_weight_decay)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.params.opt_learning_rate,
                                            betas=self.params.opt_betas, weight_decay=self.params.opt_weight_decay)
        return [self.g_optimizer, self.d_optimizer], []
