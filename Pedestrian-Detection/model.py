import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead

def build_model(num_classes):
    num_classes = 2
    # pretrained=True 옵션을 사용하여 미리 학습된 모델을 불러옵니다.
    model = retinanet_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    # Replace the classification head with a new one for our dataset
    model.head.classification_head = RetinaNetClassificationHead(in_features, num_anchors, num_classes)
    
    return model

# def build_model(num_classes):
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
#     return model