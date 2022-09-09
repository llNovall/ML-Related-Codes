from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import math
from Utils.HelperUtils import get_noise

from Losses.losses import GANWassersteinLoss, GANBCELoss
from torchvision.utils import save_image, make_grid


class GenShuffleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GenShuffleBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.PReLU(),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        output = self.block(x)

        return output

class GenTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GenTransposeBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        output = self.block(x)

        return output


class Generator(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int, block_type: str = "pixel_shuffle", num_blocks=5):
        super(Generator, self).__init__()

        self.first_layer = None
        self.last_layer = None

        """
        Blocks starts from 4x4. Each new block multiplies size of previous one by 2.
        """

        blocks = []
        for n in range(num_blocks):
            out_multiplier = math.floor(math.pow(2, (num_blocks - 1) - n))
            in_multiplier = math.floor(math.pow(2, num_blocks - n))

            #print(f'in : {in_multiplier}, out : {out_multiplier}')

            if block_type == "transpose":
                if n == 0:
                    blocks += [
                        nn.ConvTranspose2d(in_channels=latent_dim, out_channels=base_channels * out_multiplier,
                                           kernel_size=4,
                                           stride=1,
                                           padding=0, bias=False),
                        nn.BatchNorm2d(base_channels * out_multiplier),
                        nn.PReLU()]

                elif n == (num_blocks - 1):
                    blocks += [
                        nn.ConvTranspose2d(in_channels=base_channels * in_multiplier, out_channels=3, kernel_size=4,
                                           stride=2, padding=1,
                                           bias=False)]

                else:
                    blocks += [GenTransposeBlock(in_channels=base_channels * in_multiplier,
                                                 out_channels=base_channels * out_multiplier)]

            elif block_type == "pixel_shuffle":
                if n == 0:
                    blocks += [
                        nn.Conv2d(in_channels=latent_dim, out_channels=base_channels * out_multiplier * 16,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False),
                        nn.BatchNorm2d(base_channels * out_multiplier * 16),
                        nn.PReLU(),
                        nn.PixelShuffle(upscale_factor=4)
                    ]
                elif n == (num_blocks - 1):
                    blocks += [
                        GenShuffleBlock(in_channels=base_channels * in_multiplier, out_channels=3)
                    ]
                else:
                    blocks += [
                        GenShuffleBlock(in_channels=base_channels * in_multiplier,
                                        out_channels=base_channels * out_multiplier)
                    ]

        self.first_layer = nn.Sequential(
            *blocks
        )

        self.last_layer = nn.Tanh()

    def forward(self, x):
        output = self.first_layer(x)
        output = self.last_layer(output)

        return output


class DiscBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DiscBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        output = self.block(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, base_channels: int, num_blocks: int = 5):
        super(Discriminator, self).__init__()

        self.first_layer = None

        """
           Each new block divides size of previous one by 2.
        """

        blocks = []

        for n in range(num_blocks):
            in_multiplier = math.floor(math.pow(2, n - 1))
            out_multiplier = math.floor(math.pow(2, n))

            print(f'D in : {in_multiplier}, out : {out_multiplier}')
            if n == 0:
                blocks += [DiscBlock(in_channels=3, out_channels=base_channels * out_multiplier)]
            elif n == (num_blocks - 1):
                blocks += [
                    nn.Conv2d(in_channels=base_channels * in_multiplier, out_channels=base_channels * out_multiplier,
                              kernel_size=4, stride=2,
                              padding=0,
                              bias=False)]
            else:
                blocks += [DiscBlock(in_channels=base_channels * in_multiplier,
                                     out_channels=base_channels * out_multiplier)]

        self.first_layer = nn.Sequential(
            *blocks
        )


        # self.last_layer = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=base_channels * 16, out_channels=base_channels * 16, kernel_size=1, stride=1,
        #               padding=0),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=base_channels * 16, out_channels=1, kernel_size=1, stride=1, padding=0),
        #     nn.Flatten()
        # )

        self.last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=base_channels * math.floor(math.pow(2, num_blocks - 1)), out_features=1)
        )

    def forward(self, x):
        # assert x.shape[3] == 64
        output = self.first_layer(x)
        output = self.last_layer(output)

        return output


class DCGANModel(LightningModule):
    def __init__(self, latent_dim: int, gen_base_channels: int,
                 disc_base_channels: int,
                 lr_gen: float, lr_disc: float,
                 gradient_accumulation_steps: int = 1,
                 lr_scheduler_steps: int = 2000,
                 critic_lambda: int = 10,
                 critic_opt_repeat: int = 5,
                 loss: str = "bce",
                 num_blocks: int = 5,
                 block_type:str = "transpose"):
        super(DCGANModel, self).__init__()

        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.gradient_accumulation = gradient_accumulation_steps
        self.lr_scheduler_steps = lr_scheduler_steps
        self.latent_dim = latent_dim
        self.critic_opt_repeat = critic_opt_repeat
        self.num_blocks = num_blocks
        self.block_type = block_type

        self.generator = Generator(latent_dim=latent_dim, base_channels=gen_base_channels,
                                   num_blocks=num_blocks, block_type=block_type)
        self.discriminator = Discriminator(base_channels=disc_base_channels, num_blocks=num_blocks)

        self.loss = loss

        self.wasserstein_loss = GANWassersteinLoss(discriminator_lambda=critic_lambda)
        self.bce_loss = GANBCELoss()

        self.automatic_optimization = False

        self.save_hyperparameters()

        self.disc_total_steps = 0
        self.gen_total_steps = 0
        self.total_training_steps = 0

    def initialize_weights(self):
        self.generator.apply(fn=self.weights_init)
        self.discriminator.apply(fn=self.weights_init)

    @staticmethod
    def weights_init(module):

        if isinstance(module, nn.ConvTranspose2d):
            print("Assigning weights for ConvTranspose2d layer.")
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.Conv2d):
            print("Assigning weights for Conv2d layer.")
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            print("Assigning weights for BatchNorm")
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def training_step(self, batch, batch_idx):
        real_images = batch

        if self.loss == "wasserstein":
            self.wasserstein_step(real_images=real_images)
        elif self.loss == "bce":
            self.bce_step(real_images=real_images)

        if self.total_training_steps % 50 == 0:
            with torch.no_grad():
                fake_images = self.generator(get_noise(batch_size=len(real_images),
                                                       latent_dim=self.latent_dim,
                                                       device=self.device))

            self.save_images(images_to_save=fake_images, num_images_to_save=25)

    def wasserstein_step(self, real_images: torch.Tensor):

        def critic_step():
            losses = torch.zeros(self.critic_opt_repeat, device=self.device)

            for n in range(self.critic_opt_repeat):
                critic_loss = self.wasserstein_loss.get_discriminator_loss(discriminator=self.discriminator,
                                                                           generator=self.generator,
                                                                           latent_dim=self.latent_dim,
                                                                           batch_size=len(real_images),
                                                                           real_images=real_images,
                                                                           device=self.device)

                self.manual_backward(critic_loss)

                if (self.disc_total_steps + 1) % self.gradient_accumulation == 0:
                    critic_opt.step()
                    critic_lr_scheduler.step()
                    critic_opt.zero_grad()

                self.disc_total_steps += 1
                losses[n] = critic_loss

            return losses

        def generator_step():
            loss = self.wasserstein_loss.get_generator_loss(discriminator=self.discriminator,
                                                            generator=self.generator,
                                                            latent_dim=self.latent_dim,
                                                            batch_size=len(real_images),
                                                            real_images=real_images,
                                                            device=self.device)

            self.manual_backward(loss)

            if (self.gen_total_steps + 1) % self.gradient_accumulation == 0:
                gen_opt.step()
                gen_lr_scheduler.step()
                gen_opt.zero_grad()

            self.gen_total_steps += 1

            return loss

        gen_opt, critic_opt = self.optimizers(use_pl_optimizer=True)
        gen_lr_scheduler, critic_lr_scheduler = self.lr_schedulers()

        critic_losses = critic_step()
        gen_loss = generator_step()

        self.log_dict(dictionary={"gen_loss": gen_loss, "disc_loss": torch.mean(critic_losses, dim=-1)}, on_step=True,
                      on_epoch=False, prog_bar=True)

        self.total_training_steps += 1

    def bce_step(self, real_images):
        def discriminator_step():
            loss = self.bce_loss.get_discriminator_loss(discriminator=self.discriminator,
                                                        generator=self.generator,
                                                        latent_dim=self.latent_dim,
                                                        real_images=real_images,
                                                        batch_size=len(real_images),
                                                        device=self.device)

            self.manual_backward(loss)

            if (self.disc_total_steps + 1) % self.gradient_accumulation == 0:
                disc_opt.step()
                disc_lr_scheduler.step()
                disc_opt.zero_grad()

            self.disc_total_steps += 1

            return loss

        def generator_step():
            loss = self.bce_loss.get_generator_loss(discriminator=self.discriminator,
                                                    generator=self.generator,
                                                    latent_dim=self.latent_dim,
                                                    batch_size=len(real_images),
                                                    device=self.device)

            self.manual_backward(loss)

            if (self.gen_total_steps + 1) % self.gradient_accumulation == 0:
                gen_opt.step()
                gen_lr_scheduler.step()
                gen_opt.zero_grad()

            self.gen_total_steps += 1
            return loss

        gen_opt, disc_opt = self.optimizers(use_pl_optimizer=True)
        gen_lr_scheduler, disc_lr_scheduler = self.lr_schedulers()

        disc_loss = discriminator_step()
        gen_loss = generator_step()

        self.log_dict(dictionary={"gen_loss": gen_loss, "disc_loss": disc_loss}, on_step=True,
                      on_epoch=False, prog_bar=True)

        self.total_training_steps += 1

    def save_images(self, images_to_save, num_images_to_save):

        if len(images_to_save) > num_images_to_save:
            grid = make_grid(images_to_save[:num_images_to_save], nrow=math.floor(math.sqrt(num_images_to_save)),
                             padding=5)
        else:
            grid = make_grid(images_to_save, nrow=math.floor(math.sqrt(len(images_to_save))), padding=5)

        save_image(grid,
                   fp=f'../Samples/sample_resnet_e_{self.current_epoch}_gen_{self.total_training_steps}.png')

    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(f'../Checkpoints/ResNetGAN-epoch_{self.current_epoch}.ckpt')

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999))
        critic_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc, betas=(0.5, 0.999))

        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=self.lr_scheduler_steps, gamma=0.1)
        critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_opt, step_size=self.lr_scheduler_steps, gamma=0.1)

        return [gen_opt, critic_opt], [gen_scheduler, critic_scheduler]
