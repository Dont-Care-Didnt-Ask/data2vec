import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, RandomResizedCrop, ToTensor, ColorJitter, Normalize

from model_and_data_args import DataTrainingArguments, ModelArguments
from data_processing import BEiTMaskingGenerator, SimpleMaskGenerator, ImagenetIterableYTDataset

from typing import Any, Dict, Tuple, Iterable

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class DataAugmentationForData2Vec:
    def __init__(self, 
        data_args: DataTrainingArguments,
        model_args: ModelArguments,
    ):
        masking_strategy = data_args.masking_strategy
        mask_ratio = data_args.mask_ratio
        mask_patch_size = data_args.mask_patch_size

        image_size = model_args.image_size
        model_patch_size = model_args.patch_size

        if image_size % mask_patch_size != 0:
            raise ValueError("Image size must be divisible by mask patch size, "
                f"got image size {image_size} and mask patch size {mask_patch_size}")

        if masking_strategy == "beit":
            image_size_in_tokens = image_size // model_patch_size
            num_masking_patches = int(mask_ratio * image_size_in_tokens**2)
            
            self.mask_generator = BEiTMaskingGenerator(
                image_size_in_tokens,
                num_masking_patches,
                # the rest is kept default
            )
        elif masking_strategy == "random":
            self.mask_generator = SimpleMaskGenerator(
                input_size=image_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                mask_ratio=mask_ratio,
            )
        else:
            raise ValueError(f"Masking strategy {masking_strategy} is not supported, try 'beit' or 'random'.")

        self.transform_image = Compose(
            [
                Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                RandomResizedCrop(image_size, scale=(0.67, 1.0), ratio=(3. / 4., 4. / 3.)),
                RandomHorizontalFlip(0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ToTensor(),
                #Normalize(
                #    mean=torch.tensor(IMAGENET_MEAN),
                #    std=torch.tensor(IMAGENET_STD)
                #)
            ]
        )
    
    def  __call__(self, image):
        return {"pixel_values": self.transform_image(image), "mask": self.mask_generator()}


def build_data2vec_dataset(path: str, data_args: DataTrainingArguments, model_args: ModelArguments):
    transform = DataAugmentationForData2Vec(data_args, model_args)

    if data_args.dataset_class == "ImageFolder":
        return ImageFolder(path, transform=transform)
    elif data_args.dataset_class == "ImagenetIterableYTDataset":
        return ImagenetIterableYTDataset(path, transform=transform, num_readers=data_args.num_readers)
    else:
        raise ValueError(f"Dataset class {data_args.dataset_class} is not supported.")


AugmentOutput = Dict[str, torch.Tensor]
DatasetItem = Tuple[AugmentOutput, int]

def data2vec_collator(examples: Iterable[DatasetItem]):
    data, labels = zip(*examples)

    pixel_values = torch.stack([sample["pixel_values"] for sample in data])
    mask = torch.stack([sample["mask"] for sample in data])

    return {"pixel_values": pixel_values, "bool_masked_pos": mask}
