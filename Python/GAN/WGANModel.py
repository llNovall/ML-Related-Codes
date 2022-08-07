import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from GANConfig import ModelParams


class GenBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 is_final_layer: bool = False):
        super(GenBlock, self).__init__()

        self.gen_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        )

        if is_final_layer:
            self.gen_block.add_module(name="Activation_Tanh", module=nn.Tanh())
        else:
            self.gen_block.add_module(name="BatchNorm", module=nn.BatchNorm2d(out_channels))
            self.gen_block.add_module(name="Activation_ReLU", module=nn.ReLU(inplace=True))

    def forward(self, z):
        output = self.gen_block(z)
        return output


class CriticBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 is_final_layer: bool = False):
        super(CriticBlock, self).__init__()

        self.critic_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        )

        if not is_final_layer:
            self.critic_block.add_module(name="BatchNorm", module=nn.BatchNorm2d(out_channels))
            self.critic_block.add_module(name="Activation_Leaky_ReLU", module=nn.LeakyReLU(negative_slope=0.2))

    def forward(self, img):
        output = self.critic_block(img)

        return output


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, hidden_size: int = 64):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            GenBlock(in_channels=z_dim, out_channels=hidden_size * 8, kernel_size=4, stride=1, padding=0),
            GenBlock(in_channels=hidden_size * 8, out_channels=hidden_size * 4, kernel_size=4, stride=2, padding=1),
            GenBlock(in_channels=hidden_size * 4, out_channels=hidden_size * 2, kernel_size=4, stride=2, padding=1),
            GenBlock(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=4, stride=2, padding=1),
            GenBlock(in_channels=hidden_size, out_channels=3, kernel_size=4, stride=2, padding=1, is_final_layer=True)
        )

    def forward(self, z):
        output = self.generator(z)
        return output


class Critic(nn.Module):
    def __init__(self, num_img_channels: int = 3, hidden_size: int = 64):
        super(Critic, self).__init__()

        self.generator = nn.Sequential(
            CriticBlock(in_channels=num_img_channels, out_channels=hidden_size, kernel_size=4, stride=2, padding=1),
            CriticBlock(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=4, stride=2, padding=1),
            CriticBlock(in_channels=hidden_size * 2, out_channels=hidden_size * 4, kernel_size=4, stride=2, padding=1),
            CriticBlock(in_channels=hidden_size * 4, out_channels=hidden_size * 8, kernel_size=4, stride=2, padding=1),
            CriticBlock(in_channels=hidden_size * 8, out_channels=3, kernel_size=4, stride=2, padding=1,
                        is_final_layer=True)
        )

    def forward(self, img):
        output = self.generator(img)

        return output


class WGAN(LightningModule):
    def __init__(self, params: ModelParams):
        super(WGAN, self).__init__()

        self.params = params
        self.critic_lambda = 10
        self.critic_opt_repeat = 5
        self.automatic_optimization = False

        self.generator = Generator(z_dim=params.g_latent_size, hidden_size=params.g_hidden_size)
        self.critic = Critic(num_img_channels=3, hidden_size=params.d_hidden_size)

        self.gen_optimizer = None
        self.critic_optimizer = None

        self.save_hyperparameters()

    def forward(self, noise):
        output = self.generator(noise)

        return output

    def initialize_weights(self):
        self.generator.apply(self.weights_init)
        self.critic.apply(self.weights_init)

    @staticmethod
    def weights_init(m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            print("\033[92m Assigning weights for Conv layer.")
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            print("\033[92m Assigning weights for BatchNorm layer")
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def get_gradient(self, r_images: torch.Tensor, f_images: torch.Tensor, epsilon) -> torch.Tensor:

        mixed_images = r_images * epsilon + f_images * (1 - epsilon)
        mixed_scores = self.critic(mixed_images)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        return gradient

    @staticmethod
    def calculate_gradient_penalty(gradient: torch.Tensor) -> torch.Tensor:

        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    @staticmethod
    def get_gen_loss(fake_preds: torch.Tensor) -> torch.Tensor:

        gen_loss = -fake_preds.mean()

        return gen_loss

    def get_critic_loss(self, fake_preds: torch.Tensor, real_preds: torch.Tensor, gradient_penalty: torch.Tensor) -> torch.Tensor:
        critic_loss = fake_preds.mean() - real_preds.mean() + gradient_penalty * self.critic_lambda

        return critic_loss

    def get_noise(self, batch_size, latent_size) -> torch.Tensor:
        noise = torch.randn(batch_size, latent_size, 1, 1, device=self.device)
        return noise

    def training_step(self, batch, batch_idx):

        gen_opt, critic_opt = self.optimizers(use_pl_optimizer=False)

        # Update Critic

        critic_losses = torch.zeros(5, device=self.device)

        for n in range(self.critic_opt_repeat):
            critic_opt.zero_grad()

            c_noise = self.get_noise(batch.shape[0], 100).type_as(batch)

            real_images = batch
            fake_images = self.generator(c_noise)

            fake_preds = self.critic(fake_images.detach())
            real_preds = self.critic(real_images)

            epsilon = torch.rand(len(real_images), 1, 1, 1, requires_grad=True, device=self.device)
            gradient = self.get_gradient(real_images, fake_images.detach(), epsilon)
            gradient_penalty = self.calculate_gradient_penalty(gradient)
            critic_loss = self.get_critic_loss(fake_preds, real_preds, gradient_penalty)

            self.manual_backward(critic_loss, retain_graph=True)

            critic_opt.step()

            critic_losses[n] = critic_loss

        # Update  Generator

        gen_opt.zero_grad()

        g_noise = self.get_noise(batch.shape[0], 100).type_as(batch)

        fake_images = self.generator(g_noise)
        fake_preds = self.critic(fake_images)

        gen_loss = self.get_gen_loss(fake_preds)

        self.manual_backward(gen_loss)
        gen_opt.step()

        self.log_dict({"gen_loss": gen_loss, "critic_loss": torch.mean(critic_losses, dim=-1)}, prog_bar=True)

    def on_train_epoch_end(self):

        self.eval()

        with torch.no_grad():
            noise = self.get_noise(64, 100)
            sample_imgs = self(noise)
            grid = make_grid(sample_imgs, nrow=8)
            print("\033[92m Saving samples..")
            save_image(grid, fp=f'Samples/sample_e_{self.current_epoch}_{self.global_step}.png')

        self.train()

    def configure_optimizers(self):

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.params.opt_learning_rate,
                                              betas=self.params.opt_betas, weight_decay=self.params.opt_weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.opt_learning_rate,
                                                 betas=self.params.opt_betas, weight_decay=self.params.opt_weight_decay)
        return self.gen_optimizer, self.critic_optimizer
