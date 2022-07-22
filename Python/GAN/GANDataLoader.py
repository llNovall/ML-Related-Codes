from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm


class ImageDataset(VisionDataset):
    def __init__(self, path, image_size):
        super(ImageDataset, self).__init__(root=path)

        self.path = path
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.image_size = image_size

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size)
            ]
        )

        self.images = []

        img_paths = os.listdir(path)

        for path in tqdm(img_paths, total=len(img_paths)):
            img = Image.open(fp=str(Path(self.path) / path))
            # img = read_image(path=str(Path(self.path) / path), mode=ImageReadMode.RGB)
            img = self.transforms(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class GANDataModule(LightningDataModule):

    def __init__(self, path: str = '',
                 image_size: int = 64,
                 train_batch_size: int = 32,
                 manual_seed: int = 42
                 ):
        super().__init__()

        self.path = path
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.generator = Generator().manual_seed(manual_seed)

        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        full_data = ImageDataset(self.path, self.image_size)

        self.train_data = full_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          drop_last=True,
                          generator=self.generator,
                          persistent_workers=True)
