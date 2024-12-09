import numpy as np
import pandas as pd
import os
import random
import torch
from torch.nn.parameter import Parameter
from torch.backends import cudnn
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients
from .utils_GAT import *
from .model import GAT_LPA
import json



def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_VISIBLE_DEVICES"] = '6' 

def train_GAT_LPA(epochs, features, label2ct, labels, nclass, cell_locations, idx_train, n_neighbo=10, device='cuda:3',
          parameters = {
              'hidden':256,
              'dropout':0.2,
              'gatnum':4,
              'lr':0.01,
              'Lambda':1,
              'seed': 20230825,
              'lpaiters':5,
              'gat_heads':4
          }):
    #fix_seed(parameters['seed'])
    cell_locations = torch.from_numpy(np.array(cell_locations)).to(torch.float32)
    output = torch.cdist(cell_locations, cell_locations)
    _ , indices = torch.topk(output, n_neighbo + 1, largest=False)
    x = indices[:, 0].repeat_interleave(n_neighbo)
    y = indices[:, 1:].flatten()
    n_spot = cell_locations.shape[0]
    interaction = torch.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adj = interaction
    adj = adj + adj.T
    adj = torch.where(adj>1, 1, adj)
    adj = sp.coo_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    num_edges = adj.nnz()
    """adj = construct_interaction(cell_locations, n_neighbo)
    adj = preprocess_adj(adj)
    adj = torch.FloatTensor(adj).to(device)"""

    scale = StandardScaler()
    features = scale.fit_transform(features)
    features = torch.FloatTensor(features).to(device)
    labels_for_lpa = one_hot_embedding(labels, nclass).to(device)
    labels = torch.LongTensor(labels).to(device)
    idx_train = torch.IntTensor(idx_train).to(device)
    
    progressBar = SimpleProgressBar(epochs, length=20)
    model = GAT_LPA(in_feature=features.shape[1],
            hidden=parameters['hidden'],
            out_feature=nclass,
            num_edges=num_edges,
            dropout=parameters['dropout'],
            gatnum=parameters['gatnum'],
            lpaiters=parameters['lpaiters'],
            gat_heads=parameters['gat_heads'])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=parameters['lr'])

    #class_counts = torch.bincount(labels[idx_train], minlength=nclass)
    #class_weights = 1.0 / class_counts.float()
    #class_weights = torch.where(class_weights == torch.inf, 0, class_weights)
    #class_weights = class_weights / class_weights.sum() * len(class_counts) 
    #class_weights = class_weights.to(labels.device)

    crition = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss = 0.0
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = crition(output[idx_train], labels[idx_train])
        loss += loss_train.item()
        loss_train.backward() 
        optimizer.step()
        progressBar(epoch, loss)
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        print('accuracy:',accuracy(output[idx_train], labels[idx_train]))
        # row, col, _ = adj.coo()
        # adj_index = torch.stack([row, col], dim=0)
        # del labels_for_lpa, idx_train, adj
        # def model_forward(x):
        #     return model(x, adj_index)[0]
        # ig = IntegratedGradients(model_forward)
        # attributions_dict = {}
        # num_classes = output.shape[1]
        # # Compute attributions for each class
        # for target_class in range(num_classes):
        #     attributions = ig.attribute(inputs=features, target=target_class)
        #     attributions_dict[target_class] = attributions.detach().cpu().numpy().tolist()
        # with open('/home/hzq/code/idea/output/Human_Breast_Cancer_IG/IG.json', 'w') as f:
        #     json.dump(attributions_dict, f)
        return output.detach().cpu().numpy(), model
