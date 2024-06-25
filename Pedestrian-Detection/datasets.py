import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image 
from collections import defaultdict
import cv2
import random
import glob
import os
import re 
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from custom_utils import collate_fn, train_transform, valid_transform
from config import BATCH_SIZE, SEED, IMG_SIZE, EPOCHS, NUM_CLASS

root = '/home/yoojinoh/Others/PR/PedDetect-Data/data' 
torch.manual_seed(SEED) 


class PedestrianDataset(Dataset):
    def __init__(self, root:str, train:bool, split:str = "train", transforms = None):
        super().__init__()
        
        self.root = root 
        self.train = train 
        self.transforms = transforms 

        split = split.capitalize() # train -> Train 
        
        annot_path = os.path.join(root, split, split, f'{split.lower()}_annotations.json') #/home/yoojinoh/Others/PR/PedDetect-Data/data/Train/Train/train_annotations.json

        with open(annot_path) as f:
            raw_annots: list[dict] = json.load(f)["annotations"]

        img_annots: dict[str, dict] = defaultdict(list)
        for ann in raw_annots:
            if 'image' in ann['image_id']:
                img_annots[ann['image_id'] + ".jpg"].append(ann) 
        
        image_list = os.listdir(os.path.join(root, split, split, "JPEGImages"))
        filtered_image_list = [img for img in image_list if img in img_annots.keys()] # Use images only in annotations file
        image_paths = sorted([os.path.join(root, split, split, 'JPEGImages', path) for path in filtered_image_list])

        assert len(img_annots) == len(image_paths), "Number of images and labels does not match"
        
        self.local_image_list = sorted(filtered_image_list)
        self.local_images = image_paths 
        self.local_annotations = img_annots 
    
    def __len__(self):
        return len(self.local_images)
    
    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        image_filename = os.path.basename(image_path)
        anns = self.local_annotations[image_filename]  
        # import IPython; IPython.embed()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # Resize images
        image_resized /= 255.0 
        width, height = image.shape[1], image.shape[0]

        boxes = []
        labels = []
        boxes_or = []
        for ann in anns:
            x1, y1, w, h = ann['bbox'] 
            x1 = int(x1)
            y1 = int(y1) 
            x2 = x1 + int(w) 
            y2 = y1 + int(h) 

            label = ann['category_id']
            labels.append(label)

            # Resize bboxes
            x1_r = (x1 / width)*IMG_SIZE
            y1_r = (y1 / height)*IMG_SIZE
            x2_r = (x2 / width)*IMG_SIZE
            y2_r = (y2 / height)*IMG_SIZE
            boxes_or.append([x1, y1, x2, y2])
            boxes.append([x1_r, y1_r, x2_r, y2_r])
        
        boxes = torch.as_tensor(boxes, dtype = torch.float32)    

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype = torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = int(re.findall(r'\d+', ann['image_id'])[0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area 
        target['iscrowd'] = iscrowd 
        target['image_id'] = torch.tensor([image_id])

        
        if self.train:
            if self.transforms is not None:
                sample = self.transforms(image = image_resized,
                                         bboxes = target['boxes'],
                                         labels = labels)
                image = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            return image, target 

        else:
            if self.transforms is not None: 
                sample = self.transforms(image=image_resized) # , bboxes=target['boxes'], labels=labels)
                image = sample['image']
            return image, target, width, height
                
        

def create_train_dataset():
    train_dataset = PedestrianDataset(root, train=True, split="train", transforms=train_transform())
    return train_dataset

def create_valid_dataset():
    val_dataset = PedestrianDataset(root, train= False, split="val", transforms=valid_transform())
    return val_dataset

def create_train_loader(train_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle= True,
                              collate_fn= collate_fn)
    return train_loader
def create_valid_loader(val_dataset):
    valid_loader = DataLoader(val_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle= False,collate_fn= collate_fn)
                              # collate_fn= collate_fn) 
    return valid_loader
