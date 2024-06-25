import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE
import torch 
import numpy as np
import json
import cv2 
import matplotlib.pyplot as plt
import re 
import os 
from torchvision.ops import box_iou, nms 

def collate_fn(batch):
    return tuple(zip(*batch))

def train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
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

def calculate_IoU(box1, box2): # https://minimin2.tistory.com/144
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {k: v[keep] for k, v in orig_prediction.items()}
    return final_prediction

def filter_boxes_by_score(output, threshold):
    keep = output['scores'] > threshold
    filtered_output = {k: v[keep] for k, v in output.items()}
    return filtered_output


def get_image_path(image_id:int, root:str = "/home/yoojinoh/Others/PR/PedDetect-Data/data/Val/Val/JPEGImages")->str:
    return os.path.join(root, f'image ({image_id}).jpg')


def draw_boxes_on_image(image_path:str, boxes, labels, annot_path:str, save_path = None):
    image = cv2.imread(image_path) 
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #TODO (Yoojin) : Generalize codes for other format of datasets
    # image_id : image (1) 
    match = re.search(r'image \(\d+\)', image_path)
    if match:
        image_id = match.group()
    # print(image_id)
    with open(annot_path, 'r') as f:
        annotations: list[dict] = json.load(f)["annotations"]

    # for ann in annotations:
    #     if ann['image_id'] == image_id:
    #         xmin, ymin = ann['bbox'][0], ann['bbox'][1]
    #         xmax, ymax = (xmin + ann['bbox'][2]), (ymin + ann['bbox'][3])
    #         class_id = ann['category_id']
    #         class_name = 'person' if 1 else ''
    #         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    #         cv2.putText(image, class_name, (xmin, ymin- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # boxes = boxes.cpu().numpy()
    # labels = labels.cpu().numpy()

    for idx, box in enumerate(boxes):
        label = labels[idx]
        x1, y1, x2, y2 = map(int, box) # box.astype(int)
        if label == 1:
            class_name = 'person'
        else:
            class_name = ''
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, class_name, (x1, y1- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image

