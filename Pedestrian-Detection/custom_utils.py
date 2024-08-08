import torch 
import numpy as np
import json
import cv2 
import matplotlib.pyplot as plt
import re 
import os 
from torchvision.ops import nms 
from torchvision.transforms import v2 as T
import yaml 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb 

def collate_fn(batch):
    return tuple(zip(*batch))

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_wh(image_size):
    w = image_size[0]
    h = image_size[1]
    return w, h # width, height

def train_transform():
    WIDTH, HEIGHT = get_wh(load_yaml('data/configs.yaml')['train']['image_size'])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # A.HueSaturationValue(p=0.4),
        A.Resize(height=HEIGHT, width=WIDTH), 
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()  
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # pascal_voc [x_min, x_max, x_max, y_max]

def valid_transform():
    WIDTH, HEIGHT = get_wh(load_yaml('data/configs.yaml')['train']['image_size'])
    return A.Compose([
        A.Resize(height=HEIGHT, width=WIDTH),  
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def box_denormalize(x1, y1, x2, y2, width, height):
    WIDTH, HEIGHT = get_wh(load_yaml('data/configs.yaml')['train']['image_size'])
    x1 = (x1 / WIDTH) * width
    y1 = (y1 / HEIGHT) * height 
    x2 = (x2 / WIDTH) * width 
    y2 = (y2 / HEIGHT) * height 
    return x1.item(), y1.item(), x2.item(), y2.item()

def normalize_bbox(bboxes, width, height):
    return [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in bboxes]


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


# def get_image_path(image_id:int, root:str)->str:
#     return os.path.join(root, f'MP_SEL_{str(image_id).zfill(6)}.jpg')  # NOTE: Hardcoded

def draw_boxes_on_image_val(image_path, pred_boxes, gt_boxes, pred_labels, gt_labels, save_path):
    image = cv2.imread(image_path)
    
    for box, label in zip(pred_boxes, pred_labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Pred: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'GT: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imwrite(save_path, image)
    print(f'Saved image with bounding boxes to {save_path}')

def draw_boxes_on_image(image, boxes, labels, save_path):
    """
    for sanity check
    
    e.g.
        image_path = find_root('/data/tmp/data', image_filename[0]) # change dir
        img = draw_boxes_on_image_val(image_path = image_path, 
            pred_boxes= pred_boxes_norm, 
            gt_boxes= gt_boxes,
            pred_labels= labels, 
            gt_labels= gt_labels,
            save_path=f'/home/yoojinoh/Others/PR/prom5_AIproject_ATRIDA/Pedestrian-Detection/outputs/valid_{image_filename[0]}') # change dir
        
    """
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)

def visualize_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.show()


 
def get_log(configs):
    wandb.login()
    wandb.init(project=configs['project'],
               entity=configs['entity'],
               name =configs['name'])