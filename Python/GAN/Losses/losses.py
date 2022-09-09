import torch.nn as nn
import torch
from abc import abstractmethod
from typing import overload, Any


def get_epsilon(batch_size: int, device):
    epsilon = torch.rand(batch_size, 1, 1, 1, requires_grad=True, device=device)
    return epsilon


def get_gradient(critic, r_images: torch.Tensor, f_images: torch.Tensor,
                 epsilon: torch.Tensor) -> torch.Tensor:

    mixed_images = r_images * epsilon + f_images * (1 - epsilon)
    mixed_images.requires_grad = True
    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    return gradient


def calculate_gradient_penalty(gradient: torch.Tensor) -> torch.Tensor:

    gradient = gradient.view(gradient.size(0), -1)
    penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

def get_noise(batch_size, latent_dim, device) -> torch.Tensor:
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    return noise


class Losses(nn.Module):

    @abstractmethod
    def get_discriminator_loss(self, *args: Any):
        ...

    @abstractmethod
    def get_generator_loss(self, *args: Any):
        ...


class GANBCELoss(Losses):
    class DiscriminatorLoss(nn.Module):
        def __init__(self):
            super(GANBCELoss.DiscriminatorLoss, self).__init__()

            self.loss_fn = nn.BCEWithLogitsLoss()

        def forward(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                    batch_size: int, real_images, device):
            noise = get_noise(batch_size=batch_size, latent_dim=latent_dim, device=device)
            fake_images = generator(noise)

            fake_preds = discriminator(fake_images.detach())
            real_preds = discriminator(real_images.detach())

            f_loss = self.loss_fn(fake_preds, torch.zeros_like(fake_preds, device=device))
            r_loss = self.loss_fn(real_preds, torch.ones_like(real_preds, device=device))

            return 0.5 * (f_loss + r_loss)

    class GeneratorLoss(nn.Module):
        def __init__(self):
            super(GANBCELoss.GeneratorLoss, self).__init__()

            self.loss_fn = nn.BCEWithLogitsLoss()

        def forward(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                    batch_size: int, device):
            noise = get_noise(batch_size=batch_size, latent_dim=latent_dim, device=device)
            fake_images = generator(noise)
            fake_preds = discriminator(fake_images)

            gen_loss = self.loss_fn(fake_preds, torch.ones_like(fake_preds, device=device))

            return gen_loss

    def __init__(self):
        super(GANBCELoss, self).__init__()
        self.discriminator_loss = self.DiscriminatorLoss()
        self.generator_loss = self.GeneratorLoss()

    def get_discriminator_loss(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                               batch_size: int, real_images, device):
        return self.discriminator_loss(discriminator, generator, latent_dim,
                                       batch_size, real_images, device)

    def get_generator_loss(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                           batch_size: int, device):
        return self.generator_loss(discriminator, generator, latent_dim, batch_size, device)


class GANWassersteinLoss(Losses):
    class DiscriminatorLoss(nn.Module):
        def __init__(self, discriminator_lambda):
            super(GANWassersteinLoss.DiscriminatorLoss, self).__init__()

            self.discriminator_lambda = discriminator_lambda

        def forward(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                    batch_size: int, real_images, device):
            noise = get_noise(batch_size=batch_size, latent_dim=latent_dim, device=device).type_as(real_images)
            fake_images = generator(noise)
            fake_preds = discriminator(fake_images.detach())
            real_preds = discriminator(real_images.detach())

            epsilon = get_epsilon(batch_size=batch_size, device=device)
            gradient = get_gradient(critic=discriminator, r_images=real_images, f_images=fake_images, epsilon=epsilon)
            gradient_penalty = calculate_gradient_penalty(gradient=gradient)
            discriminator_loss = fake_preds.mean() - real_preds.mean() + gradient_penalty * self.discriminator_lambda

            return discriminator_loss

    class GeneratorLoss(nn.Module):

        def forward(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                    batch_size: int, real_images, device
                    ):
            noise = get_noise(batch_size=batch_size, latent_dim=latent_dim, device=device).type_as(real_images)
            fake_images = generator(noise)
            fake_preds = discriminator(fake_images)
            gen_loss = -fake_preds.mean()

            return gen_loss

    def __init__(self, discriminator_lambda: int):
        super(GANWassersteinLoss, self).__init__()

        self.discriminator_loss = self.DiscriminatorLoss(discriminator_lambda=discriminator_lambda)
        self.generator_loss = self.GeneratorLoss()

    def get_discriminator_loss(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                               batch_size: int, real_images, device):
        return self.discriminator_loss(discriminator, generator, latent_dim, batch_size, real_images, device)

    def get_generator_loss(self, discriminator: nn.Module, generator: nn.Module, latent_dim: int,
                           batch_size: int, real_images, device):
        return self.generator_loss(discriminator, generator, latent_dim, batch_size, real_images, device)
