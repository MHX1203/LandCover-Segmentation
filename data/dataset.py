from torch.utils.data import Dataset
import numpy as np
# import cv2
from PIL import Image
import torch as t
from torchvision import transforms
import random
random.seed(0)

class SatImageDataset(Dataset):

    def __init__(self, image_list, mask_list=None, augs=None, preprocess=None, shuffle=False):
        """
        
        Params:
            image_list(list): image file list

            mask_list(list): mask file list

            augs(callable object): an augmentation collections

            preprocess(function): preprocess image and mask

        """
        super(SatImageDataset, self).__init__()

        if mask_list:
            assert len(image_list) == len(mask_list), "image count must equals with mask count"
            self.mode = "train"

            self.image_masks = list(zip(image_list, mask_list))
        else:
            self.mode = 'val'
            self.image_masks = image_list   

        if shuffle:
            random.shuffle(self.image_masks)

        self.preprocess = preprocess

        self.augs = augs

    def __getitem__(self, idx):
        if self.mode == 'train':
            image, mask = self.image_masks[idx]
            image = Image.open(image)
            mask = Image.open(mask)
        else:
            image = self.image_masks[idx]
            image = Image.open(image)
            mask = None

        
        if self.preprocess:
            image, mask = self.preprocess(image, mask)
        if self.augs:
            image, mask = self.augs(image, mask)

        image = image / 255.
        image = image.transpose(2, 0, 1)

        mean= [0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        image = (image - np.array(mean, dtype=np.float32)[:, None, None]) / np.array(std, dtype=np.float32)[:, None, None]


        if self.mode == 'train':
            return t.from_numpy(image).float(), t.from_numpy(mask).long()
        else:
            return self.image_masks[idx], t.from_numpy(image).float()

    def __len__(self):
        return len(self.image_masks)
