from config import BATCH_SIZE, SEED, IMG_SIZE, EPOCHS, NUM_CLASS, LR
from model import build_model
from custom_utils import *
from tqdm.auto import tqdm 
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
import matplotlib.pyplot as plt
import time

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

def train(device, model, train_loader, optimizer, scheduler):
    model.train()

    train_losses = []
    running_loss = .0
    progress_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(progress_bar):
        images, targets = data

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)


        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value 
        train_losses.append(loss_value)

        if torch.isnan(losses).any():
            import IPython; IPython.embed()

        progress_bar.set_description(desc=f"Training Loss: {loss_value:.4f}")

    tot_loss = running_loss / len(train_loader)
    
    return train_losses, tot_loss

from torchvision.ops import box_iou

def valid(device, model, valid_loader):
    progress_bar = tqdm(valid_loader, total=len(valid_loader))
    model.eval()

    iou_scores = []

    for i, data in enumerate(progress_bar):
        images, targets, width, height = data
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        image_ids = [int(t['image_id']) for t in targets]

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu()
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()

            gt_boxes = targets[idx]['boxes'].cpu()
            gt_labels = targets[idx]['labels'].cpu()
            
            # Calculate IoU between predicted boxes and ground truth boxes
            ious = box_iou(boxes, gt_boxes)

            # For simplicity, let's assume you want to calculate the mean IoU for this batch
            if ious.numel() > 0:
                mean_iou = ious.max(dim=1)[0].mean().item()
            else:
                mean_iou = 0.0

            iou_scores.append(mean_iou)
            
            # TODO(Yoojin): sanity check as it seems that the predicted annotations is not matching with the targets.
            image_path = get_image_path(image_ids[idx])
            img = draw_boxes_on_image(image_path, boxes, labels, annot_path='/home/yoojinoh/Others/PR/PedDetect-Data/data/Val/Val/val_annotations.json')

            # Log some info (optional)
            progress_bar.set_description(desc=f"Validation IoU: {mean_iou:.4f}")

    final_avg_iou = sum(iou_scores) / len(iou_scores) if len(iou_scores) > 0 else 0.0
    return iou_scores, final_avg_iou


def main():
      best_train_loss = 99999
      best_valid_score = -1 

      train_dataset = create_train_dataset()
      valid_dataset = create_valid_dataset()
      train_loader = create_train_loader(train_dataset)
      valid_loader = create_valid_loader(valid_dataset)

      print(f"# of training samples : {len(train_dataset)}")
      print(f'# of validation samples : {len(valid_dataset)}')   

      model = build_model(NUM_CLASS).to(device)    

      params = [p for p in model.parameters() if p.requires_grad]
      optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

      train_loss_list = []
      val_loss_list = []

      # name to save the trained model with
      MODEL_NAME = 'model'

      #TODO(Yoojin): Save best model
      for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}...")
            start = time.time()

            train_loss, train_tot_loss = train(device, model, train_loader, optimizer, lr_scheduler)
            iou_scores, final_avg_iou = valid(device, model, valid_loader)
            
            print(f"Epoch [{epoch+1}] train loss: {train_tot_loss:.3f}")   
            print(f"Epoch [{epoch+1}] validation IoU score: {final_avg_iou:.3f}")   
            
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

            if best_valid_score < final_avg_iou and final_avg_iou > 0.6:
                  best_valid_score = final_avg_iou 
                  best_model = model 

                  print(f"\nBest validation score: {final_avg_iou}")
                  print(f"\nSaving best model for epoch: {epoch+1}\n")
                  torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, f'./outputs/best_{MODEL_NAME}_e{epoch}s{iou_scores}l{train_tot_loss}.pth')
            
            time.sleep(2)

if __name__ == "__main__":
    main()


    
# def valid(device, model, valid_loader):
#     val_scores = []
#     val_running_scores = .0
#     progress_bar = tqdm(valid_loader, total = len(valid_loader))
#     results = [] # records the results
#     model.eval()  

#     for i, data in enumerate(progress_bar):
#           val_avg_score = .0

#           images, targets, width, height = data 
#           images = [img.to(device) for img in images]
#           targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#           with torch.no_grad():
#                 outputs = model(images, targets) 

#           #TODO(Yoojin) : Change the metrics for validation
#           for idx, output in enumerate(outputs):
#                 boxes = output["boxes"].cpu().numpy()
#                 labels = output["labels"].cpu().numpy()
#                 scores = output["scores"].cpu().numpy() 
                
#                 # Ground truth
#                 image_id = targets[idx]['image_id']

#                 # if len(scores) > 0:
#                 #     avg_score = sum(scores) / len(scores) 
#                 # else:
#                 #      avg_score = sum(scores)
                
#                 ## ADDED BELOW ##
#                 for box, label, score in zip(boxes, labels, scores):
#                      x1, y1, x2, y2 = box
#                      x1, y1, x2, y2 = box_denormalize(y1, y1, x2, y2, width, height)
#                      result_dict = {
#                         "file_name": targets['image_id'][idx],
#                         "class_id": label-1,
#                         "confidence": score,
#                         "point1_x": x1, "point1_y": y1,
#                         "point2_x": x2, "point2_y": y1,
#                         "point3_x": x2, "point3_y": y2,
#                         "point4_x": x1, "point4_y": y2
#                      }
#                      results.append(result_dict)
#                 # val_avg_score += avg_score

#           # val_avg_score /= len(outputs)
#           # val_scores.append(val_avg_score)
#           # val_running_scores += val_avg_score 
#           progress_bar.set_description(desc=f"Validation score: {val_avg_score:.4f}")

#     final_avg_score = val_running_scores / len(valid_loader)
#     return val_scores , final_avg_score

# def valid(device, model, valid_loader):
#     progress_bar = tqdm(valid_loader, total = len(valid_loader))
#     model.eval()  

#     for i, data in enumerate(progress_bar):
#           val_avg_score = .0

#           images, targets, width, height = data 
#           images = [img.to(device) for img in images]
#           targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#           with torch.no_grad():
#                 outputs = model(images, targets) 
          
#           image_ids = [t['image_id'] for t in targets]

#           #TODO(Yoojin) : Change the metrics for validation
#           for idx, output in enumerate(outputs):
#                 boxes = output["boxes"].cpu().numpy()
#                 labels = output["labels"].cpu().numpy()
#                 scores = output["scores"].cpu().numpy() 
                
#                 image_id = image_ids[idx]                
#                 gt_labels = targets[idx]['labels']
#                 assert len(gt_labels) == len(labels), f"Mismatch: {len(gt_labels)} ground truth labels vs {len(labels)} predicted labels"


#                 for box, label, score in zip(boxes, labels, scores):
#                      x1, y1, x2, y2 = box
#                      x1, y1, x2, y2 = box_denormalize(y1, y1, x2, y2, width, height)
#                      result_dict = {
#                         "image_id": image_id,
#                         "class_id": label-1,
#                         "confidence": score,
#                         "x1": x1, "y1": y1,
#                         "x2": x2, "y2": y1,
#                         "point3_x": x2, "point3_y": y2,
#                         "point4_x": x1, "point4_y": y2
#                      }
#                      #results.append(result_dict)

#           progress_bar.set_description(desc=f"Validation score: {val_avg_score:.4f}")

#     final_avg_score = val_running_scores / len(valid_loader)
#     return val_scores , final_avg_score
