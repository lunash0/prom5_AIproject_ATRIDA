import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import progressbar
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

def parse_xml_for_person(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []
    annotations = []
    annotation_id = 0

    # Iterate over all images
    for image in root.findall('image'):
        image_id = int(image.get('id'))
        file_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Iterate over all boxes in the image
        for box in image.findall('box'):
            label = box.get('label')

            if label in ["person", "dog", "cat"]:
                category_id = 0 if label == "person" else 1

                occluded = int(box.get('occluded')) if box.get('occluded') is not None else 0
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                bbox_width = xbr - xtl
                bbox_height = ybr - ytl

                z_order = int(box.get('z_order')) if box.get('z_order') is not None else 0

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xtl, ytl, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "occluded": occluded,
                    "z_order": z_order
                })
                annotation_id += 1

    return images, annotations

def process_and_save(xml_files, output_file_path):
    cumulative_images = []
    cumulative_annotations = []
    cumulative_categories = [{"id": 0, "name": "person"}, {"id": 1, "name": "objects"}]

    image_id_counter = 0
    annotation_id_counter = 0

    for xml_file_path in xml_files:
        images, annotations = parse_xml_for_person(xml_file_path)

        for img in images:
            img['id'] = image_id_counter
            cumulative_images.append(img)
            image_id_counter += 1

        for ann in annotations:
            ann['id'] = annotation_id_counter
            ann['image_id'] = ann['image_id'] + image_id_counter - len(images)  # Adjusting image_id
            cumulative_annotations.append(ann)
            annotation_id_counter += 1

        print(f'Processed {xml_file_path}')

    coco_format_data = {
        "images": cumulative_images,
        "annotations": cumulative_annotations,
        "categories": cumulative_categories
    }

    with open(output_file_path, 'w') as json_file:
        json.dump(coco_format_data, json_file, indent=4)

    print(f'Combined annotations saved to {output_file_path}')




def main():
    dataset_base_folder = '/home/yoojinoh/Others/PR/PedDetect-Data/aihub/Bbox_0250'
    xml_files = []
    for subdir, _, files in (os.walk(dataset_base_folder)):
        for file in files:
            if file.endswith('.xml'):
                xml_file_path = os.path.join(subdir, file)
                xml_files.append(xml_file_path)
    
    train_files, val_files = train_test_split(xml_files, test_size=0, random_state=42)

    train_output_path = '/home/yoojinoh/Others/PR/PedDetect-Data/aihub/train_annotations.json' #train_annotations_1=유진 only 데이터셋
    process_and_save(train_files, train_output_path)

    val_output_path = '/home/yoojinoh/Others/PR/PedDetect-Data/aihub/val_annotations.json'
    process_and_save(val_files, val_output_path)


if __name__ == "__main__":
    main()