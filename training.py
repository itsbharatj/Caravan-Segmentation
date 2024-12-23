# %%
!pip install -U albumentations

# %%
import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from model import UNET
from utils import  (
    load_checkpoint,
    save_checkpoint, 
    get_loaders,
    check_accuracy, 
    save_predicitions_as_imgs
)

# %%
## Hyperparameters: 

LEARNING_RATE = 1e-4
DEVICE = "cuda" 
BATCH_SIZE = 32 
NUM_EPOCHS = 100 
NUM_WORKERS = 2 
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240 
PIN_MEMORY = True 
LOAD_MODEL = False
TRAIN_IMG_DIR = "/Users/bharatjain/Desktop/Deep Learning/UNet/dataset/train"
TRAIN_MASK_DIR = "/Users/bharatjain/Desktop/Deep Learning/UNet/dataset/train_masks"
VAL_IMG_DIR = "/Users/bharatjain/Desktop/Deep Learning/UNet/dataset/valid"
VAL_MASK_DIR = "/Users/bharatjain/Desktop/Deep Learning/UNet/dataset/valid_masks"
# %%
# ## It will do one epoch of training: 

# def train_fn(loader,model,optimizer,loss_fn,scaler):
#     loop = tqdm(loader)

#     for batch_idx,(data,targets) in enumerate(loop): 
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)

#         #forward: 
#         with torch.autocast(device_type='mps'):
#             predictions = model(data)
#             loss = loss_fn(predictions,targets)
        

#         #backward: 
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         ##Update tqdm loop: 
#         loop.set_postfix(loss=loss.item())

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # Forward pass (without autocast for MPS)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

# %%
def main(): 
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.1), 
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_values=255.0,  # Fixed typo in parameter name
        ), 
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_values=255.0,  # Fixed typo in parameter name
        ), 
        ToTensorV2(),
    ])

    model = UNET(in_channels=3,out_channels=1).to(DEVICE)   
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms
    )

    # Remove the scaler initialization
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, None)  # Pass None instead of scaler
        
        # Save checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        
        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # Save predictions
        save_predicitions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


# %%
if __name__ == "__main__":
    main()

# %%


# %%



