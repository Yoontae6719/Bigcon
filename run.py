import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd

def main():
    
    # 1. define random seed
    fix_seed = 2022
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 2. parser
    parser = argparse.ArgumentParser(description='Bigcon Card classification')
    
    # 2.1. basic configs
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='SSLT', help='model name, options: [will be updated]')

    # 2.2. data configs
    parser.add_argument('--data', type=str, required=True, default='train', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./DL_dataset/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--train_csv', type=str, default='train.csv', help='location of train csv')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='location of test csv')
    parser.add_argument('--fold', type=int, default= 1, help='for k-fold, e.g., if you use 5 fold,  0, 1, 2, 3, 4') 
    
    # 2.3 model configs - Self supervised learning task
        # 2.3.1. embedding configs
    parser.add_argument("--num_cont", type = int, default = 8, help = "number of continouse variables")
    parser.add_argument("--num_cat", type = int, default = 24, help = "number of categorical variables")
    parser.add_argument("--cat_unique_list", nargs='+', type=int, default = [63, 178, 52, 15, 4, 25, 20, 7, 10, 5, 5, 20, 9, 11, 20, 3, 3, 3, 3, 3, 4, 25, 8, 3],
                                              help = "unique characteristics of categorical variables e.g., --cat_unique_list 12 5 3 4") 

    parser.add_argument("--cat_dim", type = int, default = 64, help = "dimension of categorical variable")
    parser.add_argument("--cat_depth", type = int, default = 6, help = "depth of categorical transformer")
    parser.add_argument("--cat_heads", type = int, default = 8, help = "heads of categorical transformer")
    parser.add_argument("--cat_attn_dropout", type = float, default = 0.1, help = "dropout of categorical transformer")
    parser.add_argument("--cat_ff_dropout", type = float, default = 0.1, help = "dropout of categorical transformer")

    #conti
    parser.add_argument("--cont_emb", type = int, default = 64, help = "embedding dimension of continous variable")
    parser.add_argument("--cont_hidden", type = int, default = 128, help = "hidden dimension of continous variable")
    parser.add_argument("--cont_dropout", type = float, default = 0.1, help = "dropout of continous VSN")
        
    # non
    parser.add_argument("--emb_dim", type = int, default = 256, help = "lambda_value for cutmix")
    parser.add_argument("--lambda_value", type = float, default = 0.6, help = "lambda_value for cutmix")
    
        # 2.3.2 modeling f configs
    parser.add_argument('--output_attention', type = bool, default=False, help='whether to output attention in encoder')
    parser.add_argument("--mult", type = int, default = 3, help = "expention multipler of MLP")
    parser.add_argument('--dropout', type=float, default=0.05, help = 'dropout rate')
    
        # 2.3.3 modeling g configs
    parser.add_argument("--out_dim", type = int, default = 12, help = "out_dim")

    # 2.4 Model configs - Classification task
    parser.add_argument("--out_class", type = int, default = 2, help = "classification class of output dimension")
    
    
    # 2.5. Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    
    # 2.6. GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    
    Exp = Exp_Main
    
    if args.is_training:
        for ii in range(args.itr): # for robustness
            setting = 'Bigcon_{}_{}_{}_fold_{}_cat_dim_{}_cont_{}_{}_{}_{}_batch_size_{}_out_class_{}_ii_{}'.format(
                args.data,
                args.model_id,
                args.model,
                args.fold,
                args.cat_dim,
                args.cont_emb,
                args.cont_hidden,
                args.cat_depth,
                args.cat_heads,
                args.batch_size,
                args.out_class, ii)
            
            exp = Exp(args) # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            
            exp.train(setting)
            
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TEST>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            vector, y = exp.test(setting)
            vector_df = pd.DataFrame(vector)
            y_df = pd.DataFrame(y)
            
            y_df.columns = ["value_0", "value_1"]
            
            test_df = pd.concat([vector_df, y_df], axis = 1) 
            
            test_df.to_csv(f"./DL_dataset/fold_{args.fold}_test_df.csv", index = False)

            torch.cuda.empty_cache()
            
    else:
        pass
    
    
        
if __name__ == "__main__":
    main()
    
    