import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics
from torch.optim import Adam
from enum import Enum


class ConvSize(Enum):
    Conv3x3 = 1,
    Conv1x1 = 2


def initialize_conv_3x3(in_channels: int,
                        out_channels: int,
                        stride: int = 1,
                        groups: int = 1,
                        padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     dilation=padding,
                     groups=groups,
                     bias=False)


def initialize_conv_1x1(in_channels: int,
                        out_channels: int,
                        stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def create_sub_conv_block(
        conv_size: ConvSize,
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int = 1,
        add_activation_layer: bool = True
):
    model = nn.Sequential()

    if conv_size == ConvSize.Conv3x3:
        model.add_module(name='Conv3x3',
                         module=initialize_conv_3x3(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    stride=stride,
                                                    groups=1,
                                                    padding=padding,
                                                    )
                         )
    elif conv_size == ConvSize.Conv1x1:
        model.add_module(name='Conv1x1',
                         module=initialize_conv_1x1(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    stride=stride,
                                                    )
                         )

    model.add_module(name="Norm", module=nn.BatchNorm2d(out_channels))

    if add_activation_layer:
        model.add_module(name='Relu', module=nn.ReLU(inplace=True))

    return model


class Identity_Block(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int
                 ):
        super(Identity_Block, self).__init__()

        self.layer1 = create_sub_conv_block(conv_size=ConvSize.Conv1x1,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            stride=1,
                                            add_activation_layer=True)

        self.layer2 = create_sub_conv_block(conv_size=ConvSize.Conv3x3,
                                            in_channels=out_channels,
                                            out_channels=out_channels,
                                            stride=1,
                                            padding=1,
                                            add_activation_layer=True)

        self.layer3 = create_sub_conv_block(conv_size=ConvSize.Conv1x1,
                                            in_channels=out_channels,
                                            out_channels=out_channels,
                                            stride=1,
                                            padding=1,
                                            add_activation_layer=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = x

        # print(f'Identity Before Layer1 : {x.shape}')
        x = self.layer1(x)
        # print(f'Identity Before Layer1 : {x.shape}')
        x = self.layer2(x)
        # print(f'Identity Before Layer1 : {x.shape}')
        x = self.layer3(x)
        # print(f'After Before Layer1 : {x.shape}')
        x = x + skip_x

        return self.relu(x)


class Conv_Block(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int):
        super(Conv_Block, self).__init__()

        self.layer1 = create_sub_conv_block(conv_size=ConvSize.Conv3x3,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            stride=stride,
                                            padding=1,
                                            add_activation_layer=True)

        self.layer2 = create_sub_conv_block(conv_size=ConvSize.Conv3x3,
                                            in_channels=out_channels,
                                            out_channels=out_channels,
                                            stride=1,
                                            padding=1,
                                            add_activation_layer=True)

        self.layer3 = create_sub_conv_block(conv_size=ConvSize.Conv3x3,
                                            in_channels=out_channels,
                                            out_channels=out_channels,
                                            stride=1,
                                            padding=1,
                                            add_activation_layer=False)

        self.skip = create_sub_conv_block(conv_size=ConvSize.Conv1x1,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          stride=stride,
                                          add_activation_layer=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = self.skip(x)
        # print(f'Skip : {skip_x.shape}')

        # print(f'Conv Before Layer1 : {x.shape}')
        x = self.layer1(x)
        # print(f'Conv Before Layer2 : {x.shape}')
        x = self.layer2(x)
        # print(f'Conv After Layer2 : {x.shape}')
        x = self.layer3(x)
        # print(f'Conv After Layer 3 : {x.shape}')
        x = x + skip_x

        return self.relu(x)


class Residual_Block(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 num_identity_blocks: int = 1):
        super(Residual_Block, self).__init__()

        self.model = nn.Sequential()
        self.model.add_module('Conv_Block',
                              Conv_Block(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride)
                              )

        for n in range(num_identity_blocks):
            self.model.add_module(f'Identity_Block {n}',
                                  Identity_Block(in_channels=out_channels, out_channels=out_channels))

    def forward(self, x):
        x = self.model(x)

        return x


class CIFAR10ResnetModel(LightningModule):

    def __init__(self, num_channel: int, num_classes: int, learning_rate: float, epsilon: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0):
        super(CIFAR10ResnetModel, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.save_hyperparameters()

        self.starting_layer = nn.Sequential(
            # 32 x 32 x 3
            nn.Conv2d(num_channel, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            # 32 x 32 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 16 x 16 x 64
        )

        self.mid_layer = nn.Sequential(
            Residual_Block(in_channels=32, out_channels=64, stride=1, num_identity_blocks=3),
            Residual_Block(in_channels=64, out_channels=128, stride=2, num_identity_blocks=3),
            Residual_Block(in_channels=128, out_channels=256, stride=2, num_identity_blocks=5),
            Residual_Block(in_channels=256, out_channels=512, stride=2, num_identity_blocks=2)
        )

        self.final_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # print(f'Before Start : {x.shape}')
        x = self.starting_layer(x)
        # print(f'After Start : {x.shape}')

        x = self.mid_layer(x)
        # print(f'After Stage 1 : {x.shape}')

        x = self.final_layer(x)

        return x

    def initialize_weights(self):
        self.starting_layer.apply(self._initialize_weights)
        self.mid_layer.apply(self._initialize_weights)
        self.final_layer.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Conv2d):
            print("Assigning weights for Conv2d layer.")
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2, mode="fan_out", nonlinearity="relu")
            # module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Linear):
            print("Assigning weights for Linear layer.")
            module.weight = nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.BatchNorm2d):
            print("Assigning weights for BatchNorm")
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x = self.transforms(x)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self._metrics_log('train', y_hat, y, loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self._metrics_log('val', y_hat, y, loss, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self._metrics_log('test', y_hat, y, loss, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prediction = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        return prediction

    def _metrics_log(self, stage: str, y_hat, y, loss, on_step: bool = True):
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        value = {f"{stage}_loss": loss, f"{stage}_acc": acc, f"{stage}_f1": f1}
        self.log_dict(dictionary=value, prog_bar=True, on_step=on_step, on_epoch=True)

    def configure_optimizers(self):
        opt = Adam(self.parameters(),
                   lr=self.learning_rate,
                   eps=self.epsilon,
                   betas=self.betas,
                   weight_decay=self.weight_decay)

        lr_sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.005, total_iters=30)

        return [opt], [lr_sch]
