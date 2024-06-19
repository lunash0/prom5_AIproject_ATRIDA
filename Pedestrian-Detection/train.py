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
        
        # Diagnostic prints
        if torch.isnan(losses).any():
            print("NaN detected in losses")
            for key, loss in loss_dict.items():
                print(f"{key}: {loss}")

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

def valid(device, model, valid_loader):
    val_scores = []
    val_running_scores = .0
    progress_bar = tqdm(valid_loader, total = len(valid_loader))
    
    model.eval()  

    for i, data in enumerate(progress_bar):
          val_avg_score = .0

          images, targets = data 
          images = [img.to(device) for img in images]
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

          with torch.no_grad():
                outputs = model(images, targets) 

          for idx, output in enumerate(outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy() # 한개의 이미지에 대한 scores
                
                if len(scores) > 0:
                    avg_score = sum(scores) / len(scores) # 한개의 이미지에 대한 평균 score
                else:
                     avg_score = sum(scores)
                val_avg_score += avg_score
         # val_avg_score :  # batch개의 이미지들에 대한 score합
          val_avg_score /= len(outputs)
          val_scores.append(val_avg_score)
          val_running_scores += val_avg_score 
          progress_bar.set_description(desc=f"Validation score: {val_avg_score:.4f}")

    final_avg_score = val_running_scores / len(valid_loader)
    return val_scores , final_avg_score

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
            print(f"\nEPOCH {epoch+1} of {EPOCHS}")
            start = time.time()

            train_loss, train_tot_loss = train(device, model, train_loader, optimizer, lr_scheduler)
            # import IPython; IPython.embed()
            val_score, val_tot_score = valid(device, model, valid_loader)
            
            print(f"Epoch #{epoch+1} train loss: {train_tot_loss:.3f}")   
            print(f"Epoch #{epoch+1} validation score: {val_tot_score:.3f}")   
            
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

            if best_valid_score < val_tot_score:
                  best_valid_score = val_tot_score 
                  best_model = model 

                  print(f"\nBest validation score: {best_valid_score}")
                  print(f"\nSaving best model for epoch: {epoch+1}\n")
                  torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, './outputs/best_model.pth')
            
            time.sleep(2)

if __name__ == "__main__":
    main()