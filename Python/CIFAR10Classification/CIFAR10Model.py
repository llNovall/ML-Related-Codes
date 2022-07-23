import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Adam
import torchmetrics
import torchvision as tv


class CIFAR10Model(LightningModule):
    def __init__(self, num_channel: int, num_classes: int, learning_rate: float, epsilon: float = 1e-8,
                 betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0,
                 hidden_size: int = 64, dropout: float = 0.4):
        super().__init__()

        self.transforms = torch.nn.Sequential(
            tv.transforms.RandomHorizontalFlip(p=0.3),
            tv.transforms.RandomVerticalFlip(p=0.3),
        )

        self.sequential = nn.Sequential(
            nn.Conv2d(num_channel, hidden_size, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            #nn.Dropout(p=dropout),
            # 32 x 32 x 64

            #nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 x 16 x 64

            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            #nn.Dropout(p=dropout),
            # 16 x 16 x 64

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8 x 8 x 64

            # nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(hidden_size),
            # nn.GELU(),
            # nn.Dropout(p=0.4),
            # # 8 x 8 x 64
            #
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # # 4 x 4 x 64
            #
            # nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(hidden_size),
            # nn.GELU(),
            # nn.Dropout(p=0.4),
            # #  4 x 4 x 64
            #
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # # 2 x 2 x 64
            #
            # nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(hidden_size),
            # nn.GELU(),
            # nn.Dropout(p=0.4),
            # #  2 x 2 x 64
            #
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # 1 x 1 x 64

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 16 * 16, hidden_size),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )

        self.dropout = dropout
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.save_hyperparameters()

    def forward(self, x):

        output = self.sequential(x)
        output = self.classifier(output)

        return output

    def initialize_weights(self):
        self.sequential.apply(self._initialize_weights)
        self.classifier.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Conv2d):
            print("Assigning weights for Conv2d layer.")
            module.weight = nn.init.kaiming_uniform_(module.weight, a=1e-2, mode="fan_in", nonlinearity="leaky_relu")
        elif isinstance(module, nn.Linear):
            print("Assigning weights for Linear layer.")
            module.weight = nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")

    def training_step(self, batch, batch_idx):

        x, y = batch

        #x = self.transforms(x)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self._metrics_log('train', y_hat, y, loss)

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
