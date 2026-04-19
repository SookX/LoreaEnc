import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import default_collate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train_transforms(image_size = (224, 224),
                     image_mean = IMAGENET_MEAN,
                     image_std = IMAGENET_STD,
                     hflip_probability = 0.5,
                     interpolation = InterpolationMode.BILINEAR,
                     random_aug_magnitude=9):
    
    transformation_chain = []
    transformation_chain.append(v2.RandomResizedCrop(image_size, interpolation=interpolation,antialias=True))
    
    if hflip_probability > 0:
        transformation_chain.append(v2.RandomHorizontalFlip(p=hflip_probability))

    if random_aug_magnitude > 0:
        transformation_chain.append(v2.RandAugment(magnitude=random_aug_magnitude, interpolation=interpolation))
    
    transformation_chain.append(v2.PILToTensor())

    transformation_chain.append(v2.ToDtype(torch.float32, scale=True))

    transformation_chain.append(v2.Normalize(mean=image_mean, std=image_std))

    return transforms.Compose(transformation_chain)

def eval_transforms(image_size=(224, 224),
                    resize_size=(256, 256),
                    image_mean = IMAGENET_MEAN,
                    image_std = IMAGENET_STD,
                    interpolation=InterpolationMode.BILINEAR):
    pass