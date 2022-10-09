import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_provider.data_loader import Bigcon_dataloader

def data_provider(args, flag):
    '''
    data provider function : 
    flag (str) : train, val, test 
    '''

    if flag == "train":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    elif flag == "val":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        
    elif flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        
        
        
    data_set = Bigcon_dataloader(root_path =   args.root_path,
                                 train_csv = args.train_csv,
                                test_csv = args.test_csv,
                                 fold= args.fold,
                                mode = flag)

    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader