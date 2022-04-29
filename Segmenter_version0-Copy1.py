from pathlib import Path

import torchio as tio
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import dice_score 
from neptune.new.types import File
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import os
from pytorch_lightning.loggers import NeptuneLogger

from model import UNet

#Loading Data Into the file paths  
curfiles=["_".join(f.split('_')[:-1]) for f in os.listdir('../parcellation/') if f.split('.')[-1]=='nii']
x_data=pd.read_csv("Data_holder.csv")
train_final=x_data[x_data['filenames'].isin(curfiles)]
tr=train_final[['filenames','input_file_path']]
test=[]
for i in tr['filenames']:
    tfile=f'/home2/wamri/brainlabs/Brainwork/parcellation/{i}_parcel.nii'
    test.append(tfile)
tr['parcellation']=test
tr=tr.reset_index(drop=True)
sublist=tuple(zip(tr['input_file_path'].values,tr['parcellation'].values))

#Preprocessing

subjects=[]
for subject_path in sublist:
    subject = tio.Subject({"MRI":tio.ScalarImage(subject_path[0]), "Label":tio.LabelMap(subject_path[1])})
    subjects.append(subject)
    
remapp={}
for i,k in enumerate(np.unique(nib.load(sublist[0][1]).get_fdata())):
    remapp[torch.tensor(k,dtype=torch.int16)]=i

process = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad((192, 192, 192)), #This crops it to the desired metirc
            tio.RescaleIntensity((-1, 1)), # This normalizes the data
            tio.RemapLabels(remapp) #This remaps the labels 
            ])

augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))

val_transform = process
train_transform = tio.Compose([process, augmentation])

#Preprocessing end

#Train test split
train_dataset = tio.SubjectsDataset(subjects[:1000], transform=train_transform)
val_dataset = tio.SubjectsDataset(subjects[1000:], transform=val_transform)


batch_size =1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=8)

train_losses=[]
val_losses=[]


def dice_loss(inputs, target):
    smooth = 1.

    iflat = inputs.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        #print('Model Initilized')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, data):
        #print('we are moving forward?')
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        #print('we are training?')
        img = batch["MRI"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        dice_sc = dice_score(pred,mask)
       # print(f'Dice Score: {dice_sc}')
       # print(f'train loss: {loss}')
        self.log('Dice loss',dice_sc)
        # Logs
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img = batch["MRI"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss)
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")
       # print(f'val loss: {loss}')
        #val_losses.append(losses)
        
        return loss

    
    def log_images(self, img, pred, mask, name):
        
        results = []
        pred = torch.argmax(pred, 1) # Take the output with the highest value
        axial_slice = 100  # Always plot slice 50 of the 96 slices
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        
        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")

        self.logger.experiment[f"{name} Prediction vs Label"].log(File.as_image(fig))

            
    
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]


model = Segmenter()

checkpoint_callback = ModelCheckpoint(
    monitor='Val Loss',
    dirpath="/home2/wamri/brainlabs/Brainwork/workfiles/models",
    save_top_k=5,
    mode='min')

#logger = CSVLogger("logs", name="my_exp_name")
#logger = TensorBoardLogger("tb_logs", name="my_model")

neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDVlNmRiMi1iNGUwLTQzODUtYmI4Yi0xZDI3N2VlZmJiN2UifQ==",  # replace with your own
    project="otter/Parcel",  # format "<WORKSPACE/PROJECT>"
    tags=["training", "resnet"],  # optional
)
                   
trainer = pl.Trainer(
 #   fast_dev_run = True,
    
    gpus=[1,2,3], 
    auto_select_gpus=True,
    #auto_lr_find=True,
    strategy='ddp',
    precision=16,
    accelerator="auto",
    #deterministic=True,
    #plugins=DDPPlugin(find_unused_parameters=False),
    callbacks=checkpoint_callback,
    min_epochs= 50,
    max_epochs = 100,
    logger=neptune_logger
)
#trainer.tune(model)
trainer.fit(model, train_loader, val_loader)