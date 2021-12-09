import sys
import pickle
import pandas as pd
import os
import gzip
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from   category_encoders             import *
from   sklearn.compose               import *
from   sklearn.ensemble              import *
from   sklearn.linear_model          import *
from   sklearn.impute                import *
from   sklearn.metrics               import *
from   sklearn.pipeline              import *
from   sklearn.preprocessing         import *
from   sklearn.model_selection       import *
from   sklearn.feature_selection     import *
from   imblearn.over_sampling        import *
from   sklearn.neural_network        import MLPClassifier
from   sklearn.gaussian_process      import GaussianProcessClassifier
from   sklearn.neighbors             import *
from   sklearn.naive_bayes           import *
import pickle


class FilesWithMRI():
    """Returns dataframe that correspond with the MRI
       images in the list of input directories """
   
    def __init__(self, files):
        self.files = files

    def transform(self, X, **transform_params): 
        all_files = [f.split('.nii')[0] for f in self.files]
        
        # Add in filenames column in the dataframe using 'PTID' and 'VISCODE'
        X['filenames'] = 'sub-ADNI' + X['PTID'].str[:3] + 'S' + X['PTID'].str[6:] \
                                     + '_ses-' + X['VISCODE'].str.upper() + '_T1w'

        # only return df rows with corresponding MRI files
        X = X[X.filenames.isin(all_files)] 
        
        return X.reset_index(drop=True)

    def fit(self, X, y=None, **fit_params):
        return self

class RemoveNanRows():
    """Returns rows that are not None within the
       columns in the input list of columns"""
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, **transform_params): 
        for col in self.columns:
            X = X[X[col].notna()]
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    

class DropRowsCog():
    def __init__(self):
        ...
        
    def transform(self, X): 
        cog_cols = ['ADAS11_bl','ADAS13_bl','ADASQ4_bl','MMSE_bl','RAVLT_immediate_bl',
                    'RAVLT_learning_bl','RAVLT_forgetting_bl','RAVLT_perc_forgetting_bl',
                    'DIGITSCOR_bl','TRABSCOR_bl','FAQ_bl', 'mPACCdigit_bl',
                    'mPACCtrailsB_bl','ADAS13', 'ADASQ4','RAVLT_immediate', 
                    'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 
                    'DIGITSCOR','TRABSCOR', 'FAQ', 'mPACCdigit', 'mPACCtrailsB', 'MOCA', 'EcogPtMem',
                    'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
                    'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang',
                    'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt',
                    'EcogSPTotal', 'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl',
                    'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl',
                    'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl',
                    'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl',
                    'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl',
                    'EcogSPDivatt_bl', 'EcogSPTotal_bl','FDG_bl', 'PIB_bl', 'AV45_bl']
        X = X.drop(columns = cog_cols)
        return X

    def fit(self, X, y=None):
        return self

class DropNotBaseline():
    def __init__(self):
        ...

    def transform(self, X): 
        keep = ['AGE', 'PTGENDER', 'PTEDUCAT', 'PETETHCAT', 'PTRACCACT', 
                'PTMARRY', 'APOE4', 'ADAS11', 'M', 'MMSE', 'filenames']
        X = X.drop(columns = [c for c in X.columns if ('_bl' not in c) and c not in keep])
        remove = ['EXAMDATE_bl', 'CDRSB_bl', 'FLDSTRENG_bl','FSVERSION_bl', 'Years_bl','Month_bl']
        X = X.drop(columns = remove)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class CorrectUniqueCat():
    def __init__(self):
        ...

    def transform(self, X): 
        X['ABETA_bl'] = X['ABETA_bl'].replace('>1700', 1700)
        X['ABETA_bl'] = X['ABETA_bl'].replace('<200', 200)
        X = X.astype({'ABETA_bl': 'float'})

        #df['ABETA_bl'] = df['ABETA_bl'].replace('>1700', 1700)
        X['TAU_bl'] = X['TAU_bl'].replace('<80', 80)
        X = X.astype({'TAU_bl': 'float'})
        X['PTAU_bl'] = X['PTAU_bl'].replace('<8', 8)
        X = X.astype({'PTAU_bl': 'float'})
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class CorrectNaNs():
    def __init__(self):
        ...

    def transform(self, X): 
      # filter out too many nans
        X = X.loc[:, X.isnull().mean() < .70]
        for col in X.select_dtypes([np.number]).columns:
            X[col].fillna(X[col].mean(), inplace = True)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class ChangeDTypes():
    def __init__(self):
        ...

    def transform(self, X): 
      # filter out too many nans
        X = X.astype({'MMSE': int,
                'M': int})
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class OneHotEncode():
    def __init__(self):
        ...

    def transform(self, X): 
        df = X.copy()
        for col in df.select_dtypes('object').columns:
                if col != 'filenames':
                    df = pd.concat([df,pd.get_dummies(df[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)

        return df

    def fit(self, X, y=None, **fit_params):
        return self

class ChangeBaseline():
    def __init__(self, original_df, files):
        self.original_df = original_df
        self.original_df = self.original_df[['PTID', 'VISCODE',  'Ventricles',
                         'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
                         'MidTemp', 'ICV', 'MOCA', 'EcogPtMem', 'EcogPtLang', 
                         'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
                         'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
                         'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
                         'ABETA', 'TAU', 'PTAU', 'FDG', 'M','AV45']]
        self.files = files

    def transform(self, X): 
        df = X.copy()
        pp = Pipeline([('FilesWithMRI', FilesWithMRI(self.files))])
        new_df = pp.fit_transform(self.original_df.copy())
        
        df = X.copy()
        with open('data/patient_files.data', "rb") as f:
                    patient_files = pickle.load(f)

        for patient, month in patient_files:
                try:
                    for c in new_df.columns:
                        if c not in ['PTID', 'VISCODE','M','DX_bl','filenames']:
                            current_patient =new_df[new_df['filenames'].str[4:16].isin([patient[4:16]])]
                            min_patient = current_patient[current_patient.M == month]
                            actual_curr_patient = df[df['filenames'].str[4:16].isin([patient[4:16]])]
                            df.loc[actual_curr_patient.index, c+'_bl'] = [min_patient[c]]*len(actual_curr_patient)
                            df.loc[actual_curr_patient.index, 'months_from_base'] = df.loc[actual_curr_patient.index].M-month
                except:
                    pass

        return df

    def fit(self, X, y=None, **fit_params):
        return self

    

    
