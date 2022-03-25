import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, RandomResizedCrop, ToTensor, ColorJitter, Normalize

#from beit_mask_generator import BEiTMaskingGenerator
from data2vec.data2vec_datasets.simple_mask_generator import MaskingGenerator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Data2VecDataset(ImageFolder):
    def __init__(self, 
        root: str, 
        image_size=224,
        model_patch_size: int = 16, 
        mask_patch_size: int = 16, 
        mask_ratio: float = 0.6,
    ):
        assert isinstance(image_size, int)
        super().__init__(root)
        
        self.mask_generator = MaskingGenerator(
            input_size=image_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

        self.transform_image = Compose(
            [
                Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                RandomResizedCrop(image_size, scale=(0.67, 1.0), ratio=(3. / 4., 4. / 3.)),
                RandomHorizontalFlip(0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ToTensor(),
                Normalize(
                    mean=torch.tensor(IMAGENET_MEAN),
                    std=torch.tensor(IMAGENET_STD)
                )
            ]
        )

    def __getitem__(self, index: int):
        image, _ = super().__getitem__(index)
        return {"pixel_values": self.transform_image(image), "mask": self.mask_generator()}
