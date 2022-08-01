from torchvision.io import read_image, ImageReadMode
import os
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image


def preprocess_images(source: str, destination: str):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    image_size = 128

    transformations = transforms.Compose(
        [
            # transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size)
        ]
    )

    img_paths = os.listdir(source)

    if not os.path.exists(destination):
        os.mkdir(destination)

    for img_path in tqdm(img_paths, total=len(img_paths)):
        # img = Image.open(fp=str(Path(source) / img_path))
        img = read_image(path=str(Path(source) / img_path), mode=ImageReadMode.RGB) / 255.0
        t_img = transformations(img)
        save_image(t_img, fp=str(Path(destination) / img_path))
        # img.close()
