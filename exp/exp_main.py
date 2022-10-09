from exp.exp_basic import Exp_Basic
from models import my_model
from utils.tools import EarlyStopping

from utils.metrics import f1_loss
from utils.metrics import OrthogonalProjectionLoss

from data_provider.data_factory import data_provider
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


def init_func(m, init_type = "xavier", init_gain = 0.02):
    ### https://github.com/vaseline555/Federated-Averaging-PyTorch/blob/main/src/utils.py
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        else:
            raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1 or classname.find('LayerNorm') != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)  
        


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.cat_dim = args.cat_dim
        self.cont_emb = args.cont_emb
        self.cont_hidden = args.cont_hidden
        self.cat_depth = args.cat_depth
        self.cat_heads = args.cat_heads
        self.data = args.data
        
        
    def _build_model(self):
        model_dict = {
            'my_model': my_model,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            init_func(model, init_type = "xavier", init_gain = 0.02)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    
    def _get_data(self, flag):
        
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader
    
    def _select_optimizer(self):
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim
    
    def _select_criterion(self):
        '''
        CrossEntropy loss + weight probability
        '''
        weights = torch.FloatTensor([1/10, 9/10]).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights) #todo add weight
        return criterion
    
    def accuracy_function(self, real, pred):    
        real = real.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        score = f1_score(real, pred)
        acc = accuracy_score(real, pred)
        return score, acc

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_score = []
        total_acc = []
        
        op_loss = OrthogonalProjectionLoss(gamma=0.5)
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_cont, batch_cat, batch_y) in enumerate(vali_loader): 
                
                # Feed input to the model
                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if self.args.output_attention:
                    f_values, y_f, y_g, attn_weight = self.model(batch_cont, batch_cat, batch_y)
                else:
                    f_values, y_f= self.model(batch_cont, batch_cat, batch_y)

                # Compute loss
                score, acc = self.accuracy_function(batch_y, y_f)
                total_score.append(score)
                total_acc.append(acc)
                
                loss = criterion(y_f, batch_y.squeeze(dim=-1)) + op_loss(f_values, batch_y.squeeze())
                total_loss.append(loss.item())
                
                
        total_loss = np.average(total_loss)
        total_score = np.average(total_score)
        total_acc = np.average(total_acc)
        self.model.train()
        return total_loss, total_score, total_acc
                
    
    def train(self, setting):
        
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val')
        #test_data, test_loader   = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.StepLR(optimizer=model_optim, step_size=10, gamma=0.2)
        criterion = self._select_criterion()
        
        op_loss = OrthogonalProjectionLoss(gamma=0.5)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_cont, batch_cat, batch_y) in enumerate(train_loader): 
                iter_count += 1
                
                model_optim.zero_grad()
                
                # Feed input to the model
                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if self.args.output_attention:
                    f_values, y_f, attn_weight = self.model(batch_cont, batch_cat, batch_y)
                else:
                    f_values, y_f = self.model(batch_cont, batch_cat, batch_y)
                
                loss = criterion(y_f, batch_y.squeeze(dim=-1)) + op_loss(f_values, batch_y.squeeze())
                train_loss.append(loss.item())
                
                if (i + 1) % 200 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnf.all_params['max_gradient_norm'])
                model_optim.step()
            
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_score, val_acc = self.vali(vali_data, vali_loader, criterion)
            #test_loss, test_score, test_acc = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} |  Vali score: {4:.7f} Vali acc: {5:.7f}  ".format(epoch + 1, train_steps, train_loss, vali_loss, val_score, val_acc))
            early_stopping(vali_loss, self.model, path)
            
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            scheduler.step()
            
            # result save train-val loss
            folder_path = './results/' + "fine_tune/" + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            f = open(folder_path + "result.txt", 'a')
            f.write(setting + "  \n")
            f.write('epoch:{}, train_loss:{}, vali_loss:{}, vali_score:{},  val_acc{}, test_module:{} '.format(epoch, train_loss, vali_loss, val_score, val_acc, early_stopping.counter ))
            f.write('\n')
            f.write('\n')
            f.close()
            
            if early_stopping.counter == 0:
                folder_path = './results/' + "fine_tune/" + setting + '/'
                f = open(folder_path + "result_test.txt", 'a')
                f.write(setting + "  \n")
                f.write('data : {},  cat_dim : {}, cont_emb : {}, cont_hidden : {}, cat_depth : {}, cat_heads : {}, val_score::{},  val_acc:{}, test_module:{} '.format(self.data, self.cat_dim, self.cont_emb, self.cont_hidden, self.cat_depth, self.cat_heads, val_score, val_acc, early_stopping.counter ))
                f.write('\n')
                f.write('\n')
                f.close()
            else:
                pass
        
        best_model_path = path + '/' + 'checkpoint.pth'
            
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def test(self, setting):
        
        test_data, test_loader   =self._get_data(flag='test')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        total_ = []
        batch_y_list = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_cont, batch_cat, batch_y) in enumerate(test_loader): 
                
                # Feed input to the model
                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                batch_y = batch_y.to(self.device)
                
                
                if self.args.output_attention:
                    f_values, y_f, attn_weight = self.model(batch_cont, batch_cat, batch_y)
                else:
                    f_values, y_f = self.model(batch_cont, batch_cat, batch_y)
                
                total_.append(f_values.detach().cpu())
                batch_y_list.append(y_f.detach().cpu())
                
            
        total_ = torch.cat(total_).numpy()
        batch_y_data = torch.cat(batch_y_list).numpy()
                
        return total_, batch_y_data