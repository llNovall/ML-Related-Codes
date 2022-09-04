from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.utils import save_image, make_grid
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional import structural_similarity_index_measure
"""
GENERATOR
"""


class GenResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(GenResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # print(f'Start : {x.shape}')
        output = self.block(x)
        # print(f'Output : {output.shape}')
        output = torch.add(output, x)

        return output


class GenPixelShuffle(nn.Module):
    def __init__(self, channels: int):
        super(GenPixelShuffle, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(channels)
        )

    def forward(self, x):
        output = self.block(x)
        return output


class SRGANGenerator(nn.Module):
    def __init__(self, base_channels: int, num_res_blocks: int, num_ps_blocks: int):
        super(SRGANGenerator, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU(base_channels)
        )

        res_blocks = []

        for _ in range(num_res_blocks):
            res_blocks += [GenResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels)
        ]

        self.residual_blocks = nn.Sequential(*res_blocks)

        ps_blocks = []

        for _ in range(num_ps_blocks):
            ps_blocks += [GenPixelShuffle(base_channels)]

        self.pixel_shuffle_layer = nn.Sequential(*ps_blocks)

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        skip = self.first_layer(x)

        output = self.residual_blocks(skip)
        output = torch.add(output, skip)

        output = self.pixel_shuffle_layer(output)
        # print(f'PIXELSHUFFLE : {output.shape}')
        output = self.last_layer(output)
        # print(f'GENOUT : {output.shape}')
        return output


"""
DISCRIMINATOR
"""


class DiscBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DiscBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        # print(f'DStart : {x.shape}')
        output = self.block(x)
        # print(f'DEnd : {output.shape}')

        return output


class SRGANDiscriminator(nn.Module):
    def __init__(self, base_channels: int, num_blocks: int):
        super(SRGANDiscriminator, self).__init__()

        self.first_layer = nn.Sequential(
            DiscBlock(in_channels=3, out_channels=base_channels)
        )

        disc_blocks = []

        current_channels = base_channels

        for _ in range(num_blocks):
            disc_blocks += [DiscBlock(in_channels=current_channels, out_channels=current_channels * 2)]
            current_channels *= 2

        self.disc_blocks = nn.Sequential(*disc_blocks)

        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=current_channels, out_channels=current_channels * 2, kernel_size=1, stride=1,
                      padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=current_channels * 2, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        # print(f'Disc :{x.shape}')
        output = self.first_layer(x)
        output = self.disc_blocks(output)
        output = self.last_layer(output)
        # print(f'Disc : {output.shape}')
        return output


"""
LOSS
"""


class SRGANLoss(nn.Module):
    def __init__(self):
        super(SRGANLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1])

        # for p in self.vgg.parameters():
        #   p.requires_grad = False

    @staticmethod
    def mse_loss(r_images, f_images):
        return F.mse_loss(r_images, f_images)

    def adv_loss(self, x, is_real: bool):
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def vgg_loss(self, r_images, f_images):
        return F.mse_loss(self.vgg(r_images), self.vgg(f_images))

    def forward(self, generator, discriminator, hr_real, lr_real):
        hr_fake = generator(lr_real)
        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        g_loss = (
                0.001 * self.adv_loss(fake_preds_for_g, False) + 0.6 * self.vgg_loss(hr_real, hr_fake) +
                self.mse_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
                self.adv_loss(real_preds_for_d, True) +
                self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss, hr_fake


class SRGANModel(LightningModule):
    def __init__(self, gen_base_channels: int, gen_num_res_blocks: int, gen_num_ps_blocks: int,
                 disc_base_channels: int, disc_num_blocks: int,
                 lr_srresnet: float, lr_gen: float, lr_disc: float,
                 gradient_accumulation_steps: int = 1,
                 srgan_lr_scheduler_steps: int = 2000):
        super(SRGANModel, self).__init__()

        self.gen_base_channels = gen_base_channels
        self.gen_num_res_blocks = gen_num_res_blocks
        self.gen_num_ps_blocks = gen_num_ps_blocks
        self.disc_base_channels = disc_base_channels
        self.disc_num_blocks = disc_num_blocks
        self.lr_srresnet = lr_srresnet
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.srgan_lr_scheduler_steps = srgan_lr_scheduler_steps

        self.generator = SRGANGenerator(base_channels=gen_base_channels,
                                        num_res_blocks=gen_num_res_blocks,
                                        num_ps_blocks=gen_num_ps_blocks
                                        )

        self.discriminator = SRGANDiscriminator(base_channels=disc_base_channels,
                                                num_blocks=disc_num_blocks
                                                )

        self.loss = SRGANLoss()

        #self.train_metrics = MetricCollection([PeakSignalNoiseRatio(), StructuralSimilarityIndexMeasure()])
        self.psnr = PeakSignalNoiseRatio()
        #self.ssim = StructuralSimilarityIndexMeasure()
        #self.train_metrics = PeakSignalNoiseRatio()

        self.automatic_optimization = False

        self.is_training_srresnet = True

        self.srgan_training_steps = 0
        self.srresnet_training_steps = 0
        self.total_training_steps = 0

    def forward(self, img):
        return self.generator(img)

    def enable_training_srresnet(self, enable: bool):
        self.is_training_srresnet = enable

    def training_step(self, batch, batch_idx):

        hr_images, lr_images = batch
        srresnet_opt, gen_opt, disc_opt = self.optimizers()

        if self.is_training_srresnet:
            hr_fake = self.train_srresnet(lr_real=lr_images, hr_real=hr_images, opt=srresnet_opt)
        else:
            hr_fake = self.train_srgan(lr_real=lr_images, hr_real=hr_images, gen_opt=gen_opt,
                                       disc_opt=disc_opt)

        if self.total_training_steps % 200 == 0:
            grid = make_grid(hr_fake[:16], nrow=4, padding=5)
            save_image(grid,
                       fp=f'../Samples/sample_{"srresnet" if self.is_training_srresnet else "srgan"}_e_{self.current_epoch}_gen_{self.total_training_steps}.png')

        self.psnr(hr_fake.type_as(hr_images), hr_images)
        ssim = structural_similarity_index_measure(hr_fake.type_as(hr_images), hr_images)
        self.log('psnr', self.psnr, prog_bar=True, on_step=True, on_epoch=False)
        self.log('ssim', ssim, prog_bar=True, on_step=True, on_epoch=False)

        #self.log_dict(self.train_metrics(hr_fake, hr_images.type_as(hr_fake)), prog_bar=True, on_step=True, on_epoch=False)

        self.total_training_steps += 1

    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(f'../Checkpoints/SRGAN-{"srresnet" if self.is_training_srresnet else "srgan"}'
                                     f'-epoch_{self.current_epoch}.ckpt')

    def train_srresnet(self, lr_real: torch.Tensor, hr_real: torch.Tensor, opt: torch.optim.Optimizer):

        with torch.cuda.amp.autocast(enabled=True):
            hr_fake = self.generator(lr_real)
            loss = self.loss.mse_loss(r_images=hr_real, f_images=hr_fake)

        self.manual_backward(loss)

        if (self.srresnet_training_steps + 1) % self.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()

        self.log(name="srresnet_mse_loss", value=loss, on_step=True, on_epoch=False, prog_bar=True)

        self.srresnet_training_steps += 1

        return hr_fake

    def train_srgan(self, lr_real: torch.Tensor, hr_real: torch.Tensor, gen_opt: torch.optim.Optimizer,
                    disc_opt: torch.optim.Optimizer):

        with torch.cuda.amp.autocast(enabled=True):
            g_loss, d_loss, hr_fake = self.loss(self.generator, self.discriminator, hr_real, lr_real)

        self.manual_backward(g_loss)
        self.manual_backward(d_loss)

        if (self.srgan_training_steps + 1) % self.gradient_accumulation_steps == 0:
            gen_opt.step()
            gen_opt.zero_grad()

            disc_opt.step()
            disc_opt.zero_grad()

            gen_scheduler, disc_scheduler = self.lr_schedulers()

            gen_scheduler.step()
            disc_scheduler.step()

            self.log_dict(
                dictionary={"Gen_Opt_LR": gen_scheduler.get_last_lr()[0], "Disc_Opt_LR": disc_scheduler.get_last_lr()[0]},
                prog_bar=True, on_step=True, on_epoch=False)

        self.log_dict(dictionary={"gen_loss": g_loss, "disc_loss": d_loss}, on_step=True, on_epoch=False, prog_bar=True)

        self.srgan_training_steps += 1

        return hr_fake

    def validation_step(self, batch, batch_idx):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            hr_images, lr_images = batch
            print("\033[92m Saving samples..")

            # grid = make_grid(hr_images, nrow=6, padding=5)
            # save_image(grid, fp=f'../Samples/sample_e_{self.current_epoch}_orig_{batch_idx}.png')
            #
            # grid = make_grid(lr_images, nrow=6, padding=5)
            # save_image(grid, fp=f'../Samples/sample_e_{self.current_epoch}_lr_{batch_idx}.png')

            gen_images = self.generator(lr_images)
            grid = make_grid(gen_images, nrow=6, padding=5)

            save_image(grid,
                       fp=f'../Samples/sample_{"srresnet" if self.is_training_srresnet else "srgan"}_val_{self.current_epoch}_gen_{batch_idx}.png')

        self.generator.train()
        self.discriminator.train()

    def configure_optimizers(self):

        srresnet_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr_srresnet)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)

        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=self.srgan_lr_scheduler_steps, gamma=0.1)
        disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_opt, step_size=self.srgan_lr_scheduler_steps, gamma=0.1)

        return [srresnet_opt, gen_opt, disc_opt], [gen_scheduler, disc_scheduler]
