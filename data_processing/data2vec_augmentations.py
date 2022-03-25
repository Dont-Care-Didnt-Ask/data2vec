
class DataAugmentationForData2vec:
    def __init__(self, 
        image_size: int = 224,
        model_patch_size: int = 16, 
        mask_patch_size: int = 16, 
        mask_ratio: float = 0.6,
    ):

        self.mask_generator = BEiTMaskingGenerator(
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
    
    def  __call__(self, image):
        return {"pixel_values": self.transform_image(image), "mask": self.mask_generator()}
