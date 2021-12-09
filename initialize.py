from imports import *

def split_train_val(X_tr, X_v, y_tr, y_v, df):
    X_train_files = [f.split('.nii')[0] for f in X_tr]
    X_val_files = [f.split('.nii')[0] for f in X_v]
    y_train_files = [f.split('.nii')[0][:-4] for f in y_tr]
    y_val_files = [f.split('.nii')[0][:-4] for f in y_v]
    
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
    input_path = 'data/input_files/'
    target_path = 'data/target_files/'
    #csv_path = 'dfs/df_correct_dtypes.csv'
    csv_path = 'data/cleaned_df_4_13.csv'
    df = pd.read_csv(csv_path)
 
    X_tr, X_v, y_tr, y_v = get_file_splits()       
    X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val = split_train_val(X_tr, X_v, y_tr, y_v, df)
    
    return X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val, input_path, target_path, csv_path, df
    
def get_file_splits():
    
    with open('data/X_train_flist.data', 'rb') as filehandle:
        X_tr = pickle.load(filehandle)
    with open('data/y_train_flist.data', 'rb') as filehandle:
        y_tr = pickle.load(filehandle)
    with open('data/X_val_flist.data', 'rb') as filehandle:
        X_v = pickle.load(filehandle)
    with open('data/y_val_flist.data', 'rb') as filehandle:
        y_v = pickle.load(filehandle)
        
    return X_tr, X_v, y_tr, y_v
    


def get_ds_dl(batch_size=10, num_workers=16):
    
    _, _, _, _, _, _, input_path, target_path, csv_path, df = initialize_data()
    X_tr, X_v, y_tr, y_v = get_file_splits() 
    ds_train = CogDataset3d(input_path, target_path, X_tr, y_tr, df, transform=True, crop = (128,128))
    ds_val = CogDataset3d(input_path, target_path, X_v, y_v, df, transform=False, crop = (128,128))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    return ds_train, ds_val, dl_train, dl_val