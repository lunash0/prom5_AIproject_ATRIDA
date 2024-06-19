import numpy as np
import pandas as pd 
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from xml.etree import ElementTree
import numpy as np
from tqdm import tqdm  
import json 

#TODO (Yoojin): Change to args parser
split = 'Val'

data_path = '/home/yoojinoh/Others/PR/PedDetect/data'
annot_paths = os.path.join(data_path, f'{split}/{split}', 'Annotations')
annot_files = os.listdir(annot_paths)   
annot_files = [os.path.join(annot_paths, annot_file) for annot_file in annot_files] # annot_files[0]='/home/yoojinoh/Others/PR/PedDetect/data/Train/Train/Annotations/image (718).xml'

def dic_per_image(annot_path):
    tot_anns = []
    image_id = ElementTree.parse(annot_path).findall('filename')[0].text
    anns = ElementTree.parse(annot_path).findall('object')
    
    if len(anns) == 0:
        return [{'image_id': image_id, 'category_id': None, 'bbox': None}]
    
    for ann in anns: # multiple objects
        image_id = image_id.split('.')[0] # + ".jpg"
        # "person" : 1, "person-like" : 0
        label = 1 if ann.find('name').text == "person" else 0

        bbox = ann.find('bndbox')
        xmin = int(bbox.find('xmin').text) 
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text) 
        w = xmax - xmin 
        h = ymax - ymin 


        tot_anns.append({'image_id': image_id,
                        'category_id' : label,
                        'bbox' : [xmin, ymin, w, h]})
    return tot_anns 

def create_annotation_dict(annot_paths):
    tot = {'annotations': []}
    for annot_path in tqdm(annot_paths):
        tot['annotations'].extend(dic_per_image(annot_path))
    return tot

def main():
    # Create the dictionary of annotations
    base_path = '/home/yoojinoh/Others/PR/PedDetect/data'
    save_path = os.path.join(base_path, f"{split}/{split}/{split.lower()}_annotations_2.json")
    tot = create_annotation_dict(annot_files)
    with open(save_path, "w") as f:
        json.dump(tot, f, indent=4)

if __name__ == "__main__":
    main()