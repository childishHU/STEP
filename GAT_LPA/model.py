import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .layer import LPAconv

class GAT_LPA(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, num_edges, lpaiters, gat_heads, gatnum):
        super(GAT_LPA, self).__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        
        gc = nn.ModuleList()
        gc.append(GATConv(in_feature, hidden, heads=gat_heads, concat=True))
        for i in range(gatnum-2):
            gc.append(GATConv(hidden * gat_heads, hidden, heads=gat_heads, concat=True))
        gc.append(GATConv(hidden * gat_heads, out_feature, heads=1, concat=False))
        self.gc = gc
        
        self.lpa = LPAconv(lpaiters)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        for i in range(len(self.gc)-1):
            x = self.gc[i](x, edge_index,self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc[-1](x, edge_index,self.edge_weight)
        return x

    
class MLP(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden)
        self.fc2 = nn.Linear(hidden, out_feature)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.fc2(x)

        return x


