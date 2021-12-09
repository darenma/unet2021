import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import itertools
from skimage import io
import random
from pathlib import Path
from random import randint
from volumentations import *
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from imports import *
import pandas as pd

class CogDataset3d(torch.utils.data.Dataset):
   
    """
    Class for getting individual transformations and data
    Args:
        input_dir = path of input images
        target_dir = path of target images
        input = list of filenames for input
        target = list of filenames for target
        transform = Images transformation (default: False)
        crop = crop size
        df = dataframe for cognitive scores
    Output:
        Transformed input
        Transformed image target
        ADAS11 score
        MMSE score
        filename
        
    """
    
    def __init__(self, input_dir, target_dir, input_files, df, transform=False, crop = (128,128,128)):
        self.input_dir = input_dir 
        self.target_dir = target_dir
        self.input = sorted(input_files)   
        self.transform = transform
        self.crop = crop
        self.df = df
        
        
        patient_files = pickle.load(open( "/home/cguo18/brain/data/patient_files.data", "rb" ))
        self.patient_files = list(set(map(lambda x: x[0], patient_files)))
        
        self.train_transforms = Compose([RandomCrop(shape = (128,128,128), always_apply=True),
                                        ElasticTransform((0, 0.20), interpolation=4, p=1),
                                         RandomRotate90((0,1), p=0.5),
                                        #RandomGamma(gamma_limit=(0.5, 1.5), p=0.8),
                                         Normalize(always_apply=True)], p=1.0)

        self.val_transforms = Compose([CenterCrop(shape = (128,128,128), always_apply=True),
                                       Normalize(always_apply=True)], p=1.0)

    def __len__(self):
        return len(self.input)
    
        
    def __getitem__(self, i):
        
        # grab the baseline images
        X_tr_pid = list(map(lambda x: x[8:16], [self.input[i]]))
        new_input = list(map(self.get_baseline_file, X_tr_pid))[0]
        new_target = new_input.split('.nii')[0]+'_seg.nii'
        
        inp = nib.load(self.input_dir + new_input).get_fdata()
        target = nib.load(self.target_dir + new_target).get_fdata()
        
        data = {'image': inp, 'mask': target}
        
        if self.transform == True:
            aug_data = self.train_transforms(**data)
            filename_df = self.input[i].split('.nii')[0]
        else:
            aug_data = self.val_transforms(**data)
            filename_df = self.input[i].split('.nii')[0]

        #checking if image has an associated cognitive score 
        files_have_cog = self.df['filenames'].values.tolist()
        a_score = filename_df in files_have_cog
        
        #returning the cognitive score if true
        y_adas_score = None
        if a_score == True:
            y_adas_score = self.df[self.df['filenames'] == filename_df]['ADAS11'].values[0]
            
        x, y_img = aug_data['image'], aug_data['mask']
        
        return x[None,], y_img, y_adas_score, self.input[i].split('.nii')[0]
    
    def get_baseline_file(self, current_file):
        for s in filter(lambda x: current_file in x, self.patient_files):
            return s

            
def visualize_slices(brain, start, stop, target=False, slice_type='sagittal'):
    """
    brain: instance of the dataset
    start: starting slice
    stop: ending slice
    target: return input or target
    slice_type: sagittal, coronal, or horizontal slices
    """
    rang = stop-start
    cols = int(rang/5)
    
    fig, ax = plt.subplots(cols, 5, figsize = (int(25),int(rang/(1.5))))
    fig.set_facecolor("black")
    ax = ax.flatten()
    start_idx = start

    for i in range(0,rang, 1):
        if slice_type == 'sagittal':            
            brain_in = brain[0][:,start+i,:,:]
            brain_out= brain[1][start+i,:,:]
        elif slice_type == 'coronal':
            brain_in = brain[0][:,:,start+i,:]
            brain_out= brain[1][:,start+i,:]
        elif slice_type == 'horizontal':
            brain_in = brain[0][:,:,:,start+i]
            brain_out= brain[1][:,:,start+i]

        shape_img = np.shape(brain_in)
        if target == False:
            ax[i].set_facecolor('black')
            ax[i].set_title(f'slice: {start_idx}')
            ax[i].imshow(brain_in.reshape(shape_img[1],shape_img[2]))
        if target == True:
            ax[i].set_facecolor('black')
            ax[i].set_title(f'slice: {start_idx}')
            ax[i].imshow(brain_out)
        start_idx+=1
    plt.tight_layout()
    
    
def split_train_val(X_tr, X_v, df):
    X_train_files = [f.split('.nii')[0] for f in X_tr]
    X_val_files = [f.split('.nii')[0] for f in X_v]

    
    X_train = df[df['filenames'].isin(X_train_files)]
    X_val = df[df['filenames'].isin(X_val_files)]

    y_adas_train = X_train['ADAS11'].values
    y_adas_val = X_val['ADAS11'].values
    y_mmse_train = X_train['MMSE'].values
    y_mmse_val = X_val['MMSE'].values

    X_train = X_train.drop(columns=['filenames', 'ADAS11', 'MMSE'])
    X_val = X_val.drop(columns=['filenames', 'ADAS11', 'MMSE'])
    
    
    return X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val

def initialize_data():
    input_path = '/home/cpabalan/brainlabs_cp/final_brainlabs/data/input_files/'
    target_path = '/home/cpabalan/brainlabs_cp/final_brainlabs/data/target_files/'
    csv_path = 'cleaned_df_5_31.csv'
    df = pd.read_csv(csv_path)
    X_tr, X_v = get_file_splits()
    print(f'len X_v: {len(X_v)}')
    X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val = split_train_val(X_tr, X_v, df)
    
    return X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val, input_path, target_path, csv_path, df
    
    
def get_file_splits(subset='all'):

    if subset == 'all':
        paths = ['/home/cguo18/brain/data/train_files5.data', 
                 '/home/cguo18/brain/data/val_files5.data'] 

    with open(paths[0], 'rb') as filehandle:
        X_tr = pickle.load(filehandle)
    with open(paths[1], 'rb') as filehandle:
        X_v = pickle.load(filehandle)

    return X_tr, X_v
    


def get_ds_dl(subset='all', batch_size=10, num_workers=16):
    
    _, _, _, _, _, _, input_path, target_path, csv_path, df = initialize_data()
    X_tr, X_v = get_file_splits(subset=subset) 
    ds_train = CogDataset3d(input_path, target_path, X_tr, df, transform=True, crop = (128,128))
    ds_val = CogDataset3d(input_path, target_path, X_v, df, transform=False, crop = (128,128))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return ds_train, ds_val, dl_train, dl_val

def tab_predict(pipe, X_train, y_train, X_val, y_val, name = 'Model'):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    train_preds = pipe.predict(X_train)

    print(f"{f'{name} Train Loss'}: {round(mean_squared_error(y_train, train_preds),3)}")
    print(f"{f'{name}  Train R2  '}: {round(r2_score(y_train, train_preds),3)}\n")
    print(f"{f'{name}  Valid Loss'}: {round(mean_squared_error(y_val, preds),3)}")
    print(f"{f'{name}  Valid R2  '}: {round(r2_score(y_val, preds),3)}\n")
    
    return train_preds, preds


def show_test_accuracy(nums, model, dl_test, batch_size=10, 
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    use_amp = True
    model.eval()
    batch_losses = []
    total = 0
    correct = 0
    total_loss = 0
    i=0
    nums=1
    for x, y, y_score, filenames in dl_test:
        with torch.no_grad(): 
            
            y = y.squeeze(1).long().cuda()
            dim1,dim2,dim3,dim4 = y.size() #CHANGED
            x = x.view(dim1,1,dim2,dim3,dim4).cuda()
            total += dim1*dim2*dim3*dim4  


            with torch.cuda.amp.autocast(enabled=use_amp): 

                total += y.shape[0]
                reg_out, y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                batch_losses.append(loss.item())
                pred = torch.max(y_hat, 1)[1]
                correct += (pred == y).float().sum().item()   

                if i < nums:
                    slice_idx = random.randint(40,100)
                    fig, ax = plt.subplots(3,3, figsize=(10,10))
                    fig.set_facecolor("black")
                    ax=ax.flatten()
                    sag_record = [x[i][0,:,:,slice_idx], y[i][:,:,slice_idx], pred[i][:,:,slice_idx]]
                    hor_record = [x[i][0,:,slice_idx,:], y[i][:,slice_idx,:], pred[i][:,slice_idx,:]]
                    cor_record = [x[i][0,slice_idx,:,:], y[i][slice_idx,:,:], pred[i][slice_idx,:,:]]

                    for idx in range(0,3):
                        ax[idx].set_facecolor('black')
                        ax[idx].imshow((sag_record[idx]).cpu().numpy().reshape(128,128))
                        ax[idx+3].set_facecolor('black')
                        ax[idx+3].imshow((hor_record[idx]).cpu().numpy().reshape(128,128))
                        ax[idx+6].set_facecolor('black')
                        ax[idx+6].imshow((cor_record[idx]).cpu().numpy().reshape(128,128))
                        
                    i += 1
    print(f'\nCorrect predictions percentage is: {np.round((correct*100/total), 4)}')
    