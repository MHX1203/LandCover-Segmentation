import numpy as np
from PIL import Image
import PIL

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from torchvision import transforms

def get_augmentations(p=0.5):
    augmentation = Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

    return augmentation


class Augment:
    def __init__(self, augs=None):
        self.augs = augs

    def __call__(self, image, mask):
        if self.augs is None:
            return image, mask

        augmented = self.augs(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    
def preprocess(image, mask, target_size=256):
    image = image.resize((target_size, target_size), PIL.Image.BILINEAR)
    image = np.array(image, dtype=np.uint8)
    if mask is not None:
        mask = mask.resize((target_size, target_size), PIL.Image.BILINEAR)
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 255] = 7

    return image, mask