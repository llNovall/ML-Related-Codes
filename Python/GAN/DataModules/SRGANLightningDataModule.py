from pytorch_lightning import LightningDataModule
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from pathlib import Path
from torch import Generator
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, source_path, hr_image_size: int = 256, lr_image_size: int = 64):
        super(ImageDataset, self).__init__()
        print(f'Loading images from path: {source_path}')

        self.lr_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=-1.0, std=2.0),
                transforms.Resize(lr_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )

        self.hr_transforms = transforms.Compose(
            [
                transforms.RandomCrop(hr_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=0.5, std=0.5)
            ]
        )

        #self.images = []
        self.source_path = source_path
        self.img_paths = os.listdir(source_path)

        # for path in tqdm(self.img_paths, total=len(self.img_paths)):
        #     hr_img = read_image(path=str(Path(source_path) / path), mode=ImageReadMode.RGB) / 255.0
        #
        #     self.images.append(hr_img)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img = read_image(path=str(Path(self.source_path) / self.img_paths[idx]), mode=ImageReadMode.RGB) / 255.0

        hr_image = self.hr_transforms(img)
        lr_image = self.lr_transforms(hr_image)
        return hr_image, lr_image


class SRGANDataModule(LightningDataModule):

    def __init__(self, train_source_path: str, train_batch_size: int,
                 val_source_path: str, val_batch_size: int,
                 hr_image_size: int, lr_image_size: int
                 ):
        super().__init__()

        self.gen_manual_seed = 42
        self.train_source_path = train_source_path
        self.train_batch_size = train_batch_size

        self.val_source_path = val_source_path
        self.val_batch_size = val_batch_size

        self.hr_image_size = hr_image_size
        self.lr_image_size = lr_image_size

        self.generator = Generator().manual_seed(self.gen_manual_seed)
        self.train_data = None
        self.val_data = None
        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        self.train_data = ImageDataset(source_path=self.train_source_path,
                                       hr_image_size=self.hr_image_size,
                                       lr_image_size=self.lr_image_size
                                       )

        self.val_data = ImageDataset(source_path=self.val_source_path,
                                     hr_image_size=self.hr_image_size,
                                     lr_image_size=self.lr_image_size
                                     )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          drop_last=True,
                          generator=self.generator,
                          num_workers=os.cpu_count(),
                          persistent_workers=True,
                          pin_memory=True
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=os.cpu_count(),
                          generator=self.generator,
                          persistent_workers=True,
                          pin_memory=True
                          )
