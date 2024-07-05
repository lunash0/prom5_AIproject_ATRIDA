import wandb 

BATCH_SIZE = 4
SEED = 41
IMG_SIZE = 512
EPOCHS = 10
NUM_CLASS = 2 
LR = 0.005
MODEL_NAME = 'fasterrcnn'

API_KEY = None 
RUN = 0 

def get_log():
    wandb.login()
    wandb.init(project='pedestrian_detection',
               entity='finally_upper',
               name = f'exp_{RUN}',
               config={
                    "batch_size": BATCH_SIZE,
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "num_classes": NUM_CLASS,
                    "learning_rate": LR,
                    "model_name": MODEL_NAME
               })