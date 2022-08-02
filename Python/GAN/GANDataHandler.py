from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path
import os
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from GANConfig import DataModuleParams


class ImageDataset:
    def __init__(self, params: DataModuleParams):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(params.ds_image_size, antialias=True)
            ]
        )

        self.images = []

        img_paths = os.listdir(params.ds_source_path)

        for path in tqdm(img_paths, total=len(img_paths)):
            img = read_image(path=str(Path(params.ds_source_path) / path), mode=ImageReadMode.RGB) / 255.0
            if params.ds_enable_preprocess_images:
                img = self.transforms(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class GANDataModule(LightningDataModule):

    def __init__(self, dm_params: DataModuleParams
                 ):
        super().__init__()

        self.dm_params = dm_params
        self.generator = Generator().manual_seed(dm_params.dl_manual_seed)
        self.train_data = None

        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        full_data = ImageDataset(self.dm_params)

        self.train_data = full_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data,
                          batch_size=self.dm_params.dl_train_batch_size,
                          shuffle=self.dm_params.dl_shuffle,
                          num_workers=os.cpu_count(),
                          drop_last=self.dm_params.dl_drop_last,
                          generator=self.generator,
                          persistent_workers=self.dm_params.dl_persistent_workers,
                          pin_memory=True)
