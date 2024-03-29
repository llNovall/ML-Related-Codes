from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from torch import Generator
import os


class CIFAR10Dataset(LightningDataModule):

    def __init__(self, path: str = '',
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 predict_batch_size: int = 32,
                 train_val_split_size: int = 0.8,
                 manual_seed: int = 42
                 ):
        super().__init__()

        self.path = path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.predict_batch_size = predict_batch_size
        self.train_val_split_size = train_val_split_size
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize(mean=0.5,  std=0.5)
            ]
        )
        self.generator = Generator().manual_seed(manual_seed)

        self.save_hyperparameters()

    def prepare_data(self) -> None:

        CIFAR10(self.path, train=True, download=True)
        CIFAR10(self.path, train=False, download=True)

    def setup(self, stage=None) -> None:

        if stage == "fit" or stage is None:
            full_data = CIFAR10(self.path, train=True, transform=self.transforms)

            print(f'Image_Shape : {full_data[0][0].shape}')

            train_size = int(self.train_val_split_size * len(full_data))
            test_size = len(full_data) - train_size

            self.train_data, self.val_data = random_split(full_data,
                                                          [train_size, test_size],
                                                          self.generator)

        if stage == "test" or stage is None:
            self.test_data = CIFAR10(self.path, train=False, transform=self.transforms)

        if stage == "predict" or stage is None:
            self.predict_data = CIFAR10(self.path, train=False, transform=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          drop_last=True,
                          generator=self.generator,
                          persistent_workers=True,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data,
                          batch_size=self.val_batch_size,
                          num_workers=os.cpu_count(),
                          drop_last=True,
                          generator=self.generator,
                          persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data,
                          batch_size=self.test_batch_size,
                          num_workers=os.cpu_count(),
                          generator=self.generator)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.predict_data,
                          batch_size=self.test_batch_size,
                          num_workers=os.cpu_count(),
                          generator=self.generator)
