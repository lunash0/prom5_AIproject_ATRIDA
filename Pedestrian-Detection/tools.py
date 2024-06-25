import json
import cv2 
import matplotlib.pyplot as plt
import re 

def draw_boxes_on_image(image_path:str, annot_path:str)->None:
    image = cv2.imread(image_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #TODO (Yoojin) : Generalize codes for other format of datasets
    # image_id : image (1) 
    match = re.search(r'image \(\d+\)', image_path)
    if match:
        image_id = match.group()
    print(image_id)
    with open(annot_path, 'r') as f:
        annotations: list[dict] = json.load(f)["annotations"]

    for ann in annotations:
        if ann['image_id'] == image_id:
            xmin, ymin = ann['bbox'][0], ann['bbox'][1]
            xmax, ymax = (xmin + ann['bbox'][2]), (ymin + ann['bbox'][3])
            class_id = ann['category_id']
            class_name = 'person' if 1 else ''
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image, class_name, (xmin, ymin- 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.show()