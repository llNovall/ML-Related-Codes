from pytorch_lightning import LightningDataModule
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms
import os
from pathlib import Path
from torch import Generator
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):

    def __init__(self, source_path, image_size: int = 256):
        super(ImageDataset, self).__init__()
        print(f'Loading images from path: {source_path}')

        self.transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize(mean=0.5, std=0.5)
            ]
        )

        self.source_path = source_path
        self.img_paths = os.listdir(source_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_image(path=str(Path(self.source_path) / self.img_paths[idx]), mode=ImageReadMode.RGB) / 255.0
        t_img = self.transforms(img)

        return t_img


class DCGANDataModule(LightningDataModule):

    def __init__(self, train_source_path: str, train_batch_size: int, image_size: int):
        super(DCGANDataModule, self).__init__()

        self.gen_manual_seed = 42
        self.train_source_path = train_source_path
        self.train_batch_size = train_batch_size

        self.image_size = image_size

        self.generator = Generator().manual_seed(self.gen_manual_seed)
        self.train_data = None

        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        self.train_data = ImageDataset(
            source_path=self.train_source_path,
            image_size=self.image_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            generator=self.generator,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            pin_memory=True
        )
