import wandb 

BATCH_SIZE = 1
SEED = 41
IMG_SIZE = [640, 360] # [1920, 1080]=[width, height]
EPOCHS = 10
NUM_CLASS = 3 # background, person, object(dog, cat)
LR = 0.005
MODEL_NAME = 'fasterrcnn'

API_KEY = None 
RUN = 1

def get_log():
    wandb.login()
    wandb.init(project='pedestrian_detection',
               entity='finally_upper',
               name = f'toy-detection-server',
               config={
                    "batch_size": BATCH_SIZE,
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "num_classes": NUM_CLASS,
                    "learning_rate": LR,
                    "model_name": MODEL_NAME
               })