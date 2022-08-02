from torchvision.io import read_image, ImageReadMode
import os
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image


def preprocess_images(source: str, destination: str, image_size: int):

    transformations = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True)
        ]
    )

    img_paths = os.listdir(source)

    if not os.path.exists(destination):
        os.mkdir(destination)

    for img_path in tqdm(img_paths, total=len(img_paths)):
        img = read_image(path=str(Path(source) / img_path), mode=ImageReadMode.RGB) / 255.0
        t_img = transformations(img)
        save_image(t_img, fp=str(Path(destination) / img_path))

