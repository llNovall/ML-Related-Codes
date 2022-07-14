import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Adam
import torchmetrics

class CIFAR10Model(LightningModule):
    def __init__(self, num_channel: int, num_classes: int, learning_rate: float):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(num_channel, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
        self.learning_rate = learning_rate
        self.num_classes = num_classes
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
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
        elif isinstance(module, nn.Linear):
            print("Assigning weights for Linear layer.")
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")

    def training_step(self, batch, batch_idx):

        x, y = batch
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

    def _metrics_log(self, stage:str, y_hat, y, loss, on_step: bool = True):
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        value = {f"{stage}_loss": loss, f"{stage}_acc": acc, f"{stage}_f1": f1}
        self.log_dict(dictionary=value, prog_bar=True, on_step=on_step, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

