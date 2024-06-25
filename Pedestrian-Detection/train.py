from config import BATCH_SIZE, SEED, IMG_SIZE, EPOCHS, NUM_CLASS, LR, MODEL_NAME
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
from torchvision.ops import box_iou 

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

def train(device, model, train_loader, optimizer, scheduler):
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) # Gradient clipping
        
        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value 
        train_losses.append(loss_value)

        if torch.isnan(losses).any():
            import IPython; IPython.embed()

        progress_bar.set_description(desc=f"Training Loss: {loss_value:.4f}")

    tot_loss = running_loss / len(train_loader)
    
    return train_losses, tot_loss

def valid(device, model, valid_loader, iou_thresh=0.3, confidence_threshold=0.5):
    progress_bar = tqdm(valid_loader, total=len(valid_loader))

    iou_scores = []

    for i, data in enumerate(progress_bar):
        images, targets, width, height = data
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        # Apply NMS and Confidence Score Threshold
        outputs = [apply_nms(output, iou_thresh) for output in outputs]
        outputs = [filter_boxes_by_score(output, confidence_threshold) for output in outputs]

        image_ids = [int(t['image_id']) for t in targets]

        for idx, output in enumerate(outputs):
            if len(output['boxes']) > 0: # pass if it predicted as blank
                pred_boxes = []
                pred_boxes_norm = []
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                gt_boxes = targets[idx]['boxes'].cpu() #.numpy()
                gt_labels = targets[idx]['labels'].cpu() # .numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box
                        x1_norm, y1_norm, x2_norm, y2_norm = box_denormalize(x1, y1, x2, y2, width[idx], height[idx])
                        pred_boxes.append([x1, y1, x2, y2])
                        pred_boxes_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])

                # Calculate IoU between predicted boxes and ground truth boxes
                pred_boxes = torch.tensor(pred_boxes)
                ious = box_iou(pred_boxes, gt_boxes)

                if ious.numel() > 0:
                    mean_iou = ious.max(dim=1)[0].mean().item()
                else:
                    mean_iou = 0.0

                iou_scores.append(mean_iou)

                
                if mean_iou > 0.5:
                    # TODO(Yoojin): sanity check as it seems that the predicted annotations is not matching with the targets.
                    image_path = get_image_path(image_ids[idx])
                    img = draw_boxes_on_image(image_path, pred_boxes_norm, labels, annot_path='/home/yoojinoh/Others/PR/PedDetect-Data/data/Val/Val/val_annotations.json', save_path='/home/yoojinoh/Others/PR/ATRIDA_prom5_AIproject/Pedestrian-Detection/outputs/test.png')
                
                # Log some info (optional)
                progress_bar.set_description(desc=f"Validation IoU: {mean_iou:.4f}")

    final_avg_iou = sum(iou_scores) / len(iou_scores) if len(iou_scores) > 0 else 0.0
    print(f"> Final Average IoU : {final_avg_iou} = {sum(iou_scores)} / {len(iou_scores)}")
    return iou_scores, final_avg_iou


def main():
      base_root = '/home/yoojinoh/Others/PR/ATRIDA_prom5_AIproject/Pedestrian-Detection'
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

      
      for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}...")
            start = time.time()
            
            model.train()
            train_loss, train_tot_loss = train(device, model, train_loader, optimizer, lr_scheduler)
            model.eval()
            iou_scores, final_avg_iou = valid(device, model, valid_loader)
            
            print(f"Epoch [{epoch+1}] train loss: {train_tot_loss:.3f}")   
            print(f"Epoch [{epoch+1}] validation IoU score: {final_avg_iou:.3f}")   
            
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

            if best_valid_score < final_avg_iou and final_avg_iou > 0.6:
                  best_valid_score = final_avg_iou 
                  best_model = model 

                  print(f"\nBest validation score(IoU): {final_avg_iou}")
                  print(f"\nSaving best model for epoch: {epoch+1}\n")
                  torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, f'{base_root}/outputs/best_{MODEL_NAME}_e{epoch}s{final_avg_iou}l{train_tot_loss}.pth')
            
            time.sleep(2)

if __name__ == "__main__":
    main()

