import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE
import torch 
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(), # p=1.0
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
def valid_transform():
    return A.Compose([
        ToTensorV2(), # p=1.0
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['image']
    })

def box_denormalize(x1, y1, x2, y2, width, height):
    x1 = (x1 / IMG_SIZE) * width
    y1 = (y1 / IMG_SIZE) * height 
    x2 = (x2 / IMG_SIZE) * width 
    y2 = (y2 / IMG_SIZE) * height 
    return x1.item(), y1.item(), x2.item(), y2.item()

