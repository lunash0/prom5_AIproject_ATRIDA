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

root = '/home/yoojinoh/Others/PR/PedDetect-Data/aihub/'  # change dir
torch.manual_seed(SEED) 

class PedestrianDataset(Dataset):
    def __init__(self, root: str, train: bool, split: str = "train", transforms=None):
        super().__init__()
        
        self.root = root
        self.train = train
        self.transforms = transforms

        annot_path = os.path.join(root, f'{split.lower()}_annotations.json')

        with open(annot_path) as f:
            raw_annots: list[dict] = json.load(f)["annotations"]

        with open(annot_path) as f:
            images_info: list[dict] = json.load(f)["images"]

        img_annots: dict[int, dict] = defaultdict(list)
        for ann in raw_annots:
            img_annots[ann['image_id']].append(ann)

        image_list = os.listdir(os.path.join(root, "Bbox_0250"))
        filtered_image_list = []
        images_info_dict = {}
        for image in images_info:
            if image['id'] in img_annots.keys():
                filtered_image_list.append(image['file_name'])
                images_info_dict[image['file_name']] = image['id']

        image_paths = sorted([os.path.join(root, "Bbox_0250", path) for path in filtered_image_list])

        assert len(img_annots) == len(image_paths), "Number of images and labels does not match"

        self.local_image_list = filtered_image_list
        self.local_images = image_paths
        self.local_annotations = img_annots
        self.images_info_dict = images_info_dict

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        image_filename = os.path.basename(image_path)
        anns = self.local_annotations[self.images_info_dict[image_filename]]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize images
        image_resized /= 255.0
        image_resized = image_resized.copy()  # Ensure the array is contiguous in memory

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
            x1_r = (x1 / width) * IMG_SIZE
            y1_r = (y1 / height) * IMG_SIZE
            x2_r = (x2 / width) * IMG_SIZE
            y2_r = (y2 / height) * IMG_SIZE
            boxes_or.append([x1, y1, x2, y2])
            boxes.append([x1_r, y1_r, x2_r, y2_r])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = ann['image_id']

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = torch.tensor([image_id])

        if self.train:
            if self.transforms is not None:
                image_resized = torch.tensor(image_resized).permute(2, 0, 1)
                _image, _target = self.transforms(image_resized, target)

                # Ensure boxes are valid after transformation
                valid_boxes = []
                for box in _target['boxes']:
                    xmin, ymin, xmax, ymax = box
                    if xmax > xmin and ymax > ymin:
                        valid_boxes.append([xmin, ymin, xmax, ymax])
                
                _target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32)
                
                image = torch.Tensor(_image)
                target['boxes'] = torch.Tensor(_target['boxes'])
            return image, target, image_filename

        else:
            if self.transforms is not None:
                image_resized = torch.tensor(image_resized).permute(2, 0, 1)
                _image, _target = self.transforms(image_resized, target)
                
                image = torch.Tensor(_image)
                target['boxes'] = torch.Tensor(_target['boxes'])
            return image, target, width, height, image_filename


# class PedestrianDataset(Dataset):
#     def __init__(self, root:str, train:bool, split:str = "train", transforms = None):
#         super().__init__()
        
#         self.root = root 
#         self.train = train 
#         self.transforms = transforms 

        
#         annot_path = os.path.join(root, f'{split.lower()}_annotations.json')

#         with open(annot_path) as f:   
#             raw_annots: list[dict] = json.load(f)["annotations"]

#         with open(annot_path) as f:
#             images_info: list[dict] = json.load(f)["images"]
 
#         img_annots: dict[int, dict] = defaultdict(list)
#         for ann in raw_annots:
#             img_annots[ann['image_id']].append(ann) 
        
#         image_list = os.listdir(os.path.join(root, "Bbox_0250")) 
#         filtered_image_list = []
#         images_info_dict = {}
#         for image in images_info:
#             if image['id'] in img_annots.keys():
#                 filtered_image_list.append(image['file_name']) 
#                 images_info_dict[image['file_name']] = image['id']
                
#         image_paths = sorted([os.path.join(root, "Bbox_0250", path)for path in filtered_image_list])

#         assert len(img_annots) == len(image_paths), "Number of images and labels does not match"
        
#         self.local_image_list = filtered_image_list 
#         self.local_images = image_paths 
#         self.local_annotations = img_annots 
#         self.images_info_dict = images_info_dict # {filename : image_id} ex. 'MP_SEL_044051.jpg' : 0
    
#     def __len__(self):
#         return len(self.local_images)
    
#     def __getitem__(self, idx):
#         image_path = self.local_images[idx]
#         image_filename = os.path.basename(image_path)
#         anns = self.local_annotations[self.images_info_dict[image_filename]]  

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
#         image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # Resize images
#         image_resized /= 255.0 
#         image_resized = image_resized.copy()  # Ensure the array is contiguous in memory

#         width, height = image.shape[1], image.shape[0]

#         boxes = []
#         labels = []
#         boxes_or = []
#         for ann in anns:
#             x1, y1, w, h = ann['bbox'] 
#             x1 = int(x1)
#             y1 = int(y1) 
#             x2 = x1 + int(w) 
#             y2 = y1 + int(h) 

#             label = ann['category_id']
#             labels.append(label)

#             # Resize bboxes
#             x1_r = (x1 / width)*IMG_SIZE
#             y1_r = (y1 / height)*IMG_SIZE
#             x2_r = (x2 / width)*IMG_SIZE
#             y2_r = (y2 / height)*IMG_SIZE
#             boxes_or.append([x1, y1, x2, y2])
#             boxes.append([x1_r, y1_r, x2_r, y2_r])
        
#         boxes = torch.as_tensor(boxes, dtype = torch.float32)    

#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         iscrowd = torch.zeros((boxes.shape[0],), dtype = torch.int64)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         image_id = ann['image_id'] 

#         target = {}
#         target['boxes'] = boxes ##
#         target['labels'] = labels
#         target['area'] = area 
#         target['iscrowd'] = iscrowd 
#         target['image_id'] = torch.tensor([image_id])

        
#         if self.train:
#             if self.transforms is not None:
#                 _image, _target = self.transforms(image_resized, target)
                
#                 # Ensure boxes are valid after transformation
#                 valid_boxes = []
#                 for box in _target['boxes']:
#                     xmin, ymin, xmax, ymax = box
#                     if xmax > xmin and ymax > ymin:
#                         valid_boxes.append([xmin, ymin, xmax, ymax])
#                 device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

#                 _target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32, device=device)
                
#                 image = torch.Tensor(_image) 
#                 target['boxes'] = torch.Tensor(_target['boxes']) 
#             return image, target 

#         else:
#             if self.transforms is not None: 
#                 sample = self.transforms(image=image_resized.copy()) 
#                 image = sample['image']
#             return image, target, width, height, image_filename
        





# class PedestrianDataset(Dataset):
#     def __init__(self, root:str, train:bool, split:str = "train", transforms = None):
#         super().__init__()
        
#         self.root = root 
#         self.train = train 
#         self.transforms = transforms 

        
#         annot_path = os.path.join(root, f'{split.lower()}_annotations.json') #/home/yoojinoh/Others/PR/PedDetect-Data/data/Train/Train/train_annotations.json # change dir

#         with open(annot_path) as f:   
#             raw_annots: list[dict] = json.load(f)["annotations"]

#         with open(annot_path) as f:
#             images_info: list[dict] = json.load(f)["images"]
 
#         img_annots: dict[int, dict] = defaultdict(list)
#         for ann in raw_annots:
#             img_annots[ann['image_id']].append(ann) 
        
#         image_list = os.listdir(os.path.join(root, "Bbox_0250")) 
#         filtered_image_list = [img for img in image_list if img in img_annots.keys()] 
#         filtered_image_list = []
#         images_info_dict = {}
#         for image in images_info:
#             if image['id'] in img_annots.keys():
#                 filtered_image_list.append(image['file_name']) 
#                 images_info_dict[image['file_name']] = image['id']
                
#         image_paths = sorted([os.path.join(root, "Bbox_0250", path)for path in filtered_image_list])

#         assert len(img_annots) == len(image_paths), "Number of images and labels does not match"
        
#         self.local_image_list = sorted(filtered_image_list)
#         self.local_images = image_paths 
#         self.local_annotations = img_annots 
#         self.images_info_dict = images_info_dict
    
#     def __len__(self):
#         return len(self.local_images)
    
#     def __getitem__(self, idx):
#         image_path = self.local_images[idx]
#         image_filename = os.path.basename(image_path)
#         anns = self.local_annotations[self.images_info_dict[image_filename]]  

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
#         image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # Resize images
#         image_resized /= 255.0 
#         image_resized = image_resized.copy()  # Ensure the array is contiguous in memory

#         width, height = image.shape[1], image.shape[0]

#         boxes = []
#         labels = []
#         boxes_or = []
#         for ann in anns:
#             x1, y1, w, h = ann['bbox'] 
#             x1 = int(x1)
#             y1 = int(y1) 
#             x2 = x1 + int(w) 
#             y2 = y1 + int(h) 

#             label = ann['category_id']
#             labels.append(label)

#             # Resize bboxes
#             x1_r = (x1 / width)*IMG_SIZE
#             y1_r = (y1 / height)*IMG_SIZE
#             x2_r = (x2 / width)*IMG_SIZE
#             y2_r = (y2 / height)*IMG_SIZE
#             boxes_or.append([x1, y1, x2, y2])
#             boxes.append([x1_r, y1_r, x2_r, y2_r])
        
#         boxes = torch.as_tensor(boxes, dtype = torch.float32)    

#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         iscrowd = torch.zeros((boxes.shape[0],), dtype = torch.int64)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         image_id = ann['image_id'] 

#         target = {}
#         target['boxes'] = boxes
#         target['labels'] = labels
#         target['area'] = area 
#         target['iscrowd'] = iscrowd 
#         target['image_id'] = torch.tensor([image_id])

        
#         if self.train:
#             if self.transforms is not None:
#                 # sample = self.transforms(image = image_resized.copy(),
#                 #                          bboxes = target['boxes'].cpu().numpy().copy(),
#                 #                          labels = labels.cpu().numpy().copy())
#                 if self.transforms is not None:
#                     _image, _target = self.transforms(image_resized, target)
#                 image = torch.Tensor(_image) # sample['image']
#                 target['boxes'] = torch.Tensor(_target['boxes']) # torch.Tensor(sample['bboxes'])
#             return image, target # , image_filename

#         else:
#             if self.transforms is not None: 
#                 sample = self.transforms(image=image_resized.copy()) # , bboxes=target['boxes'], labels=labels)
#                 image = sample['image']
#             return image, target, width, height, image_filename
                
        

def create_train_dataset():
    train_dataset = PedestrianDataset(root='/home/yoojinoh/Others/PR/PedDetect-Data/aihub/', train=True, split="train", transforms=train_transform(train=True))
    return train_dataset

def create_valid_dataset():
    val_dataset = PedestrianDataset(root='/home/yoojinoh/Others/PR/PedDetect-Data/aihub/', train= False, split="val", transforms=train_transform(train=False))
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
    return valid_loader
