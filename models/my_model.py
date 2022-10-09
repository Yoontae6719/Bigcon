import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import TokenEmbedding
from layers.MLP import FeedForwardBlock
from layers.transformer import TabTransformer, MLP
from layers.VSN import GLU, GRN, VariableSelectionNetwork

class Model(nn.Module):
    ''' Pretrained model for Self-supervised learning for Tabluar dataset.  '''
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.num_cat = args.num_cat
        self.num_cont = args.num_cont

        self.g_input_size = (args.num_cat * args.cat_dim) + args.cont_emb

        # Step 1. Continuous variable embeddings
        self.cont_emb_layers = nn.ModuleList()
        for i in range(self.num_cont):
            cont_emb = nn.Linear(1, args.cont_emb, bias = False)
            self.cont_emb_layers.append(cont_emb)
        

        # Step 2.1 Modeling f_{theta_cont}
        # in this case, we separated f_albal and f_beta. 
        self.f_alpha = TabTransformer(categories = tuple(args.cat_unique_list), # tuple containing the number of unique values within each category
                                       dim = args.cat_dim,                         
                                       depth = args.cat_depth,
                                       heads = args.cat_heads,
                                       attn_dropout = args.cat_attn_dropout,
                                       ff_dropout = args.cat_ff_dropout
                                      )

        # Step 2.2 Modeling f_{theta_cat}
        self.f_beta = VariableSelectionNetwork(input_size = args.cont_emb,
                                               num_cont_inputs = args.num_cont,
                                               hidden_size = args.cont_emb,
                                               dropout = args.cont_dropout,
                                              )
        

        self.g_linear = MLP([self.g_input_size, int(self.g_input_size // 4)] )#nn.Linear(args.emb_dim * 2, args.out_dim)
        self.projection = nn.Linear(int(self.g_input_size // 4), 2, bias = False)
        
    def forward(self, x_cont, x_cat, y = None):
        
        cont_ = self.apply_cont_embedding(x_cont)        
        
        # Step 2. categorical modeling
        f_ori_cat = self.f_alpha(x_cat)   # (num_cat * cat_dim)
        
        # Step 3. Continous modeling
        f_ori_cont, f_ori_weights = self.f_beta(cont_)
        
        # Step 2.1. concatenates
        f_values = torch.cat( (f_ori_cont, f_ori_cat), dim = -1 )
        
        # step 3
        f_values_emb = self.g_linear(f_values)
        f_values_out = self.projection(f_values_emb)
        return f_values_emb, f_values_out
    
    def apply_cont_embedding(self, x):
        '''
        Apply continous variable to embedding space 
        '''
        
        if len(x.size()) == 3:
            conti_vectors = []
            for i in range(self.num_cont):
                conti_emb = self.cont_emb_layers[i](x[:, :, i:i+1])
                cont_vectors.append(conti_emb)  
            conti_emb = torch.cat(cont_vectors, dim = 2)
        else:
            conti_vectors = []
            for i in range(self.num_cont):
                conti_emb = self.cont_emb_layers[i](x[:, i:i+1])
                conti_vectors.append(conti_emb)
            conti_emb = torch.cat(conti_vectors, dim = 1)
            
        return conti_emb