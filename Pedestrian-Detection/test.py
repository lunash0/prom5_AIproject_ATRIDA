import torch
import cv2
import time
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
import cv2 
import time
import torch
import tqdm 
from custom_utils import * 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

def build_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(checkpoint_path, num_classes, device):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def filter_boxes_by_score(output, threshold):
    keep = output['scores'] > threshold
    filtered_output = {k: v[keep] for k, v in output.items()}
    return filtered_output

def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {k: v[keep] for k, v in orig_prediction.items()}
    return final_prediction

def process_frame(frame, model, device, iou_thresh=0.3, confidence_threshold=0.5):
    image = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0]

    output = apply_nms(output, iou_thresh)
    output = filter_boxes_by_score(output, confidence_threshold)

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    return boxes, labels, scores

def draw_boxes_on_frame(frame, boxes, labels, scores, threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def detect_frame(model, img_frame, device, thr):
    boxes, labels, scores = process_frame(img_frame, model, device, confidence_threshold=thr)
    img_frame = draw_boxes_on_frame(img_frame, boxes, labels, scores, threshold=thr)
    return img_frame

def detect_video(model, input_path, output_path, device, thr=0.5):
    cap = cv2.VideoCapture(input_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(output_path, codec, video_fps, video_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames: {frame_cnt}')

    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print(f'Processed all frames')
            break

        img_frame = detect_frame(model, img_frame, device, thr)
        video_writer.write(img_frame)

    video_writer.release()
    cap.release()

if __name__ == "__main__":
    video_name = 'after_detection'
    input_video = '/home/yoojinoh/Others/PR/PedDetect-Data/2954065-hd_1920_1080_30fps.mp4'
    output_video = f'/home/yoojinoh/Others/PR/PedDetect-Data/{video_name}.mp4'
    model_path = '/home/yoojinoh/Others/PR/ATRIDA_prom5_AIproject/Pedestrian-Detection/outputs/best_fasterrcnn_e6s0.7686803706057437l0.10222071961704958.pth'
    num_classes = 2 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, num_classes, device)
    detect_video(model, input_video, output_video, device)