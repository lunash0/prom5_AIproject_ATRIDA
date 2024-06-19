"""
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

# import utils 
import re 
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

CFG = {
    'NUM_CLASS':2,
    'IMG_SIZE':512,
    'EPOCHS':10, # 20
    'LR':3e-4,
    'BATCH_SIZE':4,
    'SEED':41,
}

#TODO(Yoojin) : Skip empty annotation files
def collate_fn(batch):
    images, annots = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(annots)):
        if len(annots[i]['boxes']) > 0:
            target = {
                "boxes": annots[i]['boxes'],
                "labels": annots[i]['labels']
            }
            targets.append(target)

    return images, targets 

class PedestrianDataset(Dataset):
    def __init__(self,
                 root: str, 
                 train: bool,
                 split: str = "train",
                 transforms= None):
        super().__init__()
        split = split.capitalize() # train -> Train 
        self.root = root        
        self.train = train 

        annotation_path = os.path.join(root, split, split, f'{split.lower()}_annotations.json')
        with open(annotation_path) as f:
            raw_annotations: list[dict] = json.load(f)["annotations"]
        fname_to_annotation: dict[str, dict] = defaultdict(list)
        for ann in raw_annotations:
            fname_to_annotation[(ann['image_id'])+ ".jpg"].append(ann)
        
        image_paths = os.listdir(os.path.join(root, split, split, 'JPEGImages'))
        image_paths = sorted([os.path.join(root, split, split, 'JPEGImages', path) for path in image_paths])
        image_paths = sorted([path for path in image_paths if os.path.basename(path) in fname_to_annotation])

        # Filter out images with empty annotations
        filtered_image_paths = []
        filtered_annotations = []
        for path in image_paths:
            base_name = os.path.basename(path)
            if base_name in fname_to_annotation:
                annotations = fname_to_annotation[base_name]
                if any(ann['bbox'] for ann in annotations):  # Check if any annotation has a non-empty bbox
                    filtered_image_paths.append(path)
                    filtered_annotations.append(annotations)

        assert len(filtered_image_paths) == len(filtered_annotations), "Number of images and labels does not match"

        self.local_images = filtered_image_paths
        self.local_annotations = filtered_annotations

        self.transforms = transforms
    def __len__(self):
        return len(self.local_images)
    
    def __getitem__(self, idx):
        path = self.local_images[idx]
        anns: list[dict] = self.local_annotations[idx] 
        labels: list = []
        boxes: list = []

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0 
        width, height = image.shape[0], image.shape[1]

        for ann in anns:
            x1, y1, w, h = ann['bbox'] 
            x1 = int(x1)
            y1 = int(y1) 
            x2 = x1 + int(w) 
            y2 = y1 + int(h) 

            label = ann['category_id']
            labels.append(label)
            boxes.append([x1, y1, x2, y2])
        
        # convert boxes and labels into tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype = torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = int(re.findall(r'\d+', ann['image_id'])[0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([image_id])

        
        if self.train:     
            if self.transforms is not None:
                transformed = self.transforms(image = image,bboxes = boxes,labels = labels)
                
                image = transformed['image']
                target['boxes'] = torch.Tensor(transformed['bboxes'])
            return image, target 
        else:
            if self.transforms is not None:
                transformed = self.transforms(image = image)
                image = transformed["image"]
            return image_id, image, width, height      

def train_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),  # RandomHorizontalFlip
        A.RandomBrightnessContrast(p=0.2),  # RandomAffine with brightness and contrast
        A.RandomRotate90(p=0.5),  # RandomAffine with 90 degrees rotation
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=10, p=0.5),  # RandomAffine with shift, scale, and rotate
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def test_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        ToTensorV2(),
    ])


def box_denormalize(x1, y1, x2, y2, width, height):
    x1 = (x1 / CFG['IMG_SIZE']) * width
    y1 = (y1 / CFG['IMG_SIZE']) * height
    x2 = (x2 / CFG['IMG_SIZE']) * width
    y2 = (y2 / CFG['IMG_SIZE']) * height
    return x1.item(), y1.item(), x2.item(), y2.item()

                  
def train(model, device, train_loader, val_loader, optimizer, scheduler):
    epochs = CFG['EPOCHS']
    train_losses = []
    validation_scores = []
    best_loss = 9999999
    best_model = None  
    results = []
    for epoch in range(epochs):
        train_losses = []
        running_loss = 0.0
        val_running_loss = 0.0

        model.train()  # Set the model to training mode

        for images, targets in tqdm(train_loader):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

                if scheduler is not None:
                    scheduler.step()
            except IndexError as e:
                print(f'Index error: {e}')
                import IPython; IPython.embed()

        model.eval()  # Set the model to evaluation mode
        for image_ids, image, width, height in tqdm(iter(val_loader)):
            images = [img.to(device) for img in images]

            with torch.no_grad():
                outputs = model(images)

            for idx, output in enumerate(outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = box_denormalize(x1, y1, x2, y2, width[idx], height[idx])
                    result_dict = {
                        "file_name": image_ids[idx],
                        "class_id": label-1,
                        "confidence": score,
                        "point1_x": x1, "point1_y": y1,
                        "point2_x": x2, "point2_y": y1,
                        "point3_x": x2, "point3_y": y2,
                        "point4_x": x1, "point4_y": y2
                    }
                    results.append(result_dict)
                    
        train_losses.append(running_loss / len(train_loader))
        validation_scores.append(scores) 

        print(f"Epoch: {epoch+1}/{epochs}.. ",
              f"Training Loss: {running_loss / len(train_loader):.3f}.. ")

        if epoch > 2 and best_loss > (running_loss / len(train_loader)):
            best_loss = running_loss / len(train_loader) 
            best_model = model
    return best_model, train_losses, validation_scores, results 

def build_model(num_classes=CFG['NUM_CLASS']):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model 

def main():
    torch.manual_seed(1)
    root = '/home/yoojinoh/Others/PR/PedDetect-Data/data'

    train_dataset = PedestrianDataset(root, train=True, split="train", transforms=train_transforms())
    val_dataset = PedestrianDataset(root, train= False, split="val", transforms=test_transforms())

    print('Load dataloader')
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
    
    print('Define model')
    model = build_model().to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])

    # TODO(Yoojin) : Tune the model (https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    print('Train model')
    infer_model, tr_loss, val_loss, results =  train(model, device, train_loader, val_loader, optimizer, lr_scheduler)
    results.to_csv('baseline_submit.csv', index=False)

if __name__ == "__main__":
    main()
"""