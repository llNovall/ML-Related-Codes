import torchmetrics
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as f
import torchmetrics

class MNISTModel(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=-1)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=10)
        self.f1 = torchmetrics.F1Score(num_classes=10, average='macro')
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        return self.sequential(x)

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

        self._metrics_log('val', y_hat, y, loss)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self._metrics_log('test', y_hat, y, loss)

        return loss

    def _metrics_log(self, stage:str, y_hat, y, loss):
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        value = {f"{stage}_loss": loss, f"{stage}_acc": acc, f"{stage}_f1": f1}
        self.log_dict(dictionary=value, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

