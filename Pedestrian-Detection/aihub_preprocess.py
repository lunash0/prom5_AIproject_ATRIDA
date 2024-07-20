import os
import xml.etree.ElementTree as ET
import json
from sklearn.model_selection import train_test_split

def parse_xml_for_person(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []
    annotations = []
    category_id = 1  # We only have one category "person"
    annotation_id = 1

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

            # We only process "person" labels
            if label == "person":
                occluded = int(box.get('occluded')) if box.get('occluded') is not None else 0
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                # Calculate width and height of the bounding box
                bbox_width = xbr - xtl
                bbox_height = ybr - ytl

                z_order = int(box.get('z_order')) if box.get('z_order') is not None else 0

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # "person" category
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
    cumulative_categories = [{"id": 1, "name": "person"}]  # Static category

    image_id_counter = 1
    annotation_id_counter = 1

    for xml_file_path in xml_files:
        images, annotations = parse_xml_for_person(xml_file_path)

        # Adjust image and annotation IDs to be cumulative and unique
        for img in images:
            img['id'] = image_id_counter
            cumulative_images.append(img)
            image_id_counter += 1

        for ann in annotations:
            ann['id'] = annotation_id_counter
            ann['image_id'] = ann['image_id'] + image_id_counter - len(images) - 1  # Adjusting image_id
            cumulative_annotations.append(ann)
            annotation_id_counter += 1

        print(f'Processed {xml_file_path}')

    # Combine everything into the COCO format
    coco_format_data = {
        "images": cumulative_images,
        "annotations": cumulative_annotations,
        "categories": cumulative_categories
    }

    # Save the combined data to a single JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(coco_format_data, json_file, indent=4)

    print(f'Combined annotations saved to {output_file_path}')

# Set the dataset path
dataset_base_folder = '/content/drive/My Drive/dataset'  # Path to the base folder containing subfolders with XML files

# Get list of all XML files
xml_files = []
for subdir, _, files in os.walk(dataset_base_folder):
    for file in files:
        if file.endswith('.xml'):
            xml_file_path = os.path.join(subdir, file)
            xml_files.append(xml_file_path)

# Split the dataset into train and val with 80:20 ratio
train_files, val_files = train_test_split(xml_files, test_size=0.2, random_state=42)

# Process and save train dataset
train_output_path = '/content/drive/My Drive/dataset/train_annotations.json'
process_and_save(train_files, train_output_path)

# Process and save val dataset
val_output_path = '/content/drive/My Drive/dataset/val_annotations.json'
process_and_save(val_files, val_output_path)
