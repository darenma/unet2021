
import torchio as tio
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics.functional import dice_score
from pytorch_lightning.loggers import NeptuneLogger
import matplotlib.pyplot as plt
from neptune.new.types import File
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from model import UNet

seed_everything(seed=14)


PARAMS = {
    'min_epochs': 30,
    'max_epochs': 50,
    'learning_rate': 1e-4,
    'batch_size': 1,
    'weight_decay': 1e-6
}

# Loading Data Into the file paths
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
trainlist = tuple(
    zip(train['input_file_path'].values, train['Parcellation'].values))
vallist = tuple(zip(val['input_file_path'].values, val['Parcellation'].values))
train_subjects = []
val_subjects = []

# Scalar Image


for subject_path in trainlist:
    subject = tio.Subject({"MRI": tio.ScalarImage(
        subject_path[0]), "Label": tio.LabelMap(subject_path[1])})
    train_subjects.append(subject)

for subject_path in vallist:
    subject = tio.Subject({"MRI": tio.ScalarImage(
        subject_path[0]), "Label": tio.LabelMap(subject_path[1])})
    val_subjects.append(subject)


# Preprocessing

remapp = pd.read_pickle('mapping.pkl')

process = tio.Compose([
    tio.CropOrPad((192, 192, 192)),  # This crops it to the desired metirc
    tio.RescaleIntensity((-1, 1)),  # This normalizes the data
    tio.RemapLabels(remapp)
])

spatial_transforms = {
    tio.RandomElasticDeformation(): 0.2,
    tio.RandomAffine(scales=(0.9, 1.1), degrees=(-20, 20)): 0.8,
}
augmentation = tio.Compose([
    tio.OneOf(spatial_transforms, p=0.5),
])

val_transform = process
train_transform = tio.Compose([process, augmentation])

# Preprocessing end

# Train test split
train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
# Test Train split end


# DATA LOADER
batch_size = PARAMS['batch_size']  # Hyper parameter
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=64)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=64)


class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4,weight_decay=1e-6)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        img = batch["MRI"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        dice_sc = dice_score(pred,mask)
        print(f'Dice Score: {dice_sc}')
        print(f'train loss: {loss.item()}')
        self.log('Dice loss',dice_sc)
        # Logs
        self.log("Train Loss", loss.item())
        if batch_idx % 50 == 0:
            names=[i.split('/')[-1] for i in batch['MRI']['path']]
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train",names)
            plt.close('all')
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        img = batch["MRI"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss.item())
        names=[i.split('/')[-1] for i in batch['MRI']['path']]
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val",names)
        plt.close('all')
        print(f'val loss: {loss.item()}')
        
        return loss

    
    def log_images(self, img, pred, mask, name, file_names):
        
        pred = torch.argmax(pred, 1) # Take the output with the highest value
        axial_slice = 100  # Always plot slice 50 of the 96 slices
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth \n"+file_names[0],wrap=True)
        
        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Prediction")
        self.logger.experiment[f"{name} Prediction vs Label"].log(File.as_image(fig))
        plt.close('all')
    
    
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]


model = Segmenter()

checkpoint_callback = ModelCheckpoint(
    monitor='Val Loss',
    dirpath="/home2/wamri/brainlabs/Brainwork/workfiles/models",
    save_top_k=5,
    mode='min')

neptune_logger = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDVlNmRiMi1iNGUwLTQzODUtYmI4Yi0xZDI3N2VlZmJiN2UifQ==",  # replace with your own
                               project="otter/Parcel",  # format "<WORKSPACE/PROJECT>"
                               tags=["training", "parcellation"],  # optional
                               )

trainer = pl.Trainer(
    gpus=[1, 2, 3],
    auto_select_gpus=True,
    #strategy='ddp',
    precision=16,
    accelerator="auto",
    plugins=DDPPlugin(find_unused_parameters=False),
    callbacks=checkpoint_callback,
    min_epochs=PARAMS['min_epochs'],  # Hyper parameter
    max_epochs=PARAMS['max_epochs'],  # Hyper parameter
    logger=neptune_logger
)
# trainer.tune(model)
trainer.fit(model, train_loader, val_loader)
