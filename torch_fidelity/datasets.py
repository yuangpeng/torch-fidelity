import sys
import tarfile
from contextlib import redirect_stdout

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10

from torch_fidelity.helpers import vassert


class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, "Input is not a PIL.Image")
        return F.pil_to_tensor(img)


class ImagesPathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img


class TarsPathDataset(Dataset):
    def __init__(self, tar_files, transforms=None):
        self.tar_files = tar_files
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms
        self.image_paths = self._extract_image_paths()

    def _extract_image_paths(self):
        image_paths = []
        for tar_file in self.tar_files:
            with tarfile.open(tar_file, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.lower().endswith(("jpg", "jpeg", "png")):
                        image_paths.append((tar_file, member.name))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        tar_file, image_path = self.image_paths[index]
        with tarfile.open(tar_file, "r") as tar:
            img_data = tar.extractfile(image_path).read()
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img = self.transforms(img)
        return img


class Cifar10_RGB(CIFAR10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class Cifar100_RGB(CIFAR100):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class STL10_RGB(STL10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class RandomlyGeneratedDataset(Dataset):
    def __init__(self, num_samples, *dimensions, dtype=torch.uint8, seed=2021):
        vassert(dtype == torch.uint8, "Unsupported dtype")
        rng_stash = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.imgs = torch.randint(0, 255, (num_samples, *dimensions), dtype=dtype)
        finally:
            torch.set_rng_state(rng_stash)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i]
