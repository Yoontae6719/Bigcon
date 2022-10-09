import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


    
    
class Bigcon_dataloader(Dataset):
    def __init__(self, root_path, mode, train_csv, test_csv, fold):
        
        self.root_path = root_path
        self.mode = mode
        
        self.train = train_csv
        self.test  = test_csv
        
        self.fold = int(fold)
        
        
        self.__read_data__()
        
        if mode == "train" :
            self.data = pd.DataFrame(self.train_cont)
        elif mode == "val" :
            self.data = pd.DataFrame(self.valid_cont)
        elif mode == "test":
            self.data = pd.DataFrame(self.test_cont)
    
        
    def __read_data__(self):
        # Step 1. load data
        df_train = pd.read_csv(os.path.join(self.root_path,  self.train))
        df_test  = pd.read_csv(os.path.join(self.root_path,  self.test))
        
        # Step 2. K-fold (train-val)
        val_idx = list(df_train[df_train['fold'] == self.fold].index)
        val = df_train[df_train['fold'] == self.fold].reset_index(drop=True).iloc[:, :-1]
        train = df_train[df_train['fold'] != self.fold].reset_index(drop=True).iloc[:, :-1]
        
        # Step 3. Split conti and cat variable
        self.train_cont = train.iloc[:, :8].values
        self.train_cat = train.iloc[:, 8:-1].values
        self.trian_y = train.iloc[:, -1:].values
        
        self.valid_cont = val.iloc[:, :8].values
        self.valid_cat = val.iloc[:, 8:-1].values
        self.valid_y  = val.iloc[:, -1:].values
        
        self.test_cont = df_test.iloc[:, :8].values
        self.test_cat = df_test.iloc[:, 8:-1].values 
        self.test_y = df_test.iloc[:, -1:].values
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            cont_variable = self.train_cont[idx]
            cat_variable = self.train_cat[idx]
            labels = self.trian_y[idx]
            
            return torch.tensor(cont_variable, dtype = torch.float32), torch.tensor(cat_variable, dtype = torch.long), torch.tensor(labels, dtype = torch.long)
        
        elif self.mode == "val":
            cont_variable = self.valid_cont[idx]
            cat_variable = self.valid_cat[idx]
            labels = self.valid_y[idx]
            return torch.tensor(cont_variable, dtype = torch.float32), torch.tensor(cat_variable, dtype = torch.long), torch.tensor(labels, dtype = torch.long)
        
        else: # for test set
            cont_variable = self.test_cont[idx]
            cat_variable = self.test_cat[idx]
            labels = self.test_y[idx]
            return torch.tensor(cont_variable, dtype = torch.float32), torch.tensor(cat_variable, dtype = torch.long) , torch.tensor(labels, dtype = torch.long)
        