import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_geometric.nn import GATConv

class StandGAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(StandGAT1, self).__init__()
        self.conv1 = GATConv(nfeat, nclass,heads=1)

    def forward(self, x, adj):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
    
        return x

   
class StandGAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(StandGAT2, self).__init__()

        num_head = 4
        head_dim = nhid//num_head
  
        self.conv1 = GATConv(nfeat, head_dim, heads=num_head)
        self.conv2 = GATConv(nhid,  nclass,   heads=1, concat=False)

        self.dropout_p = dropout
      
    def forward(self, x, adj):

        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        x = F.dropout(x, p= self.dropout_p, training=self.training)

        x = self.conv2(x, edge_index)
        

        return x

class StandGATX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(StandGATX, self).__init__()

        num_head = 4
        head_dim = nhid//num_head

        self.conv1 = GATConv(nfeat, head_dim, heads=num_head)
        self.conv2 = GATConv(nhid, nclass)
        self.convx = nn.ModuleList([GATConv(nhid, head_dim, heads=num_head) for _ in range(nlayer-2)])
        self.dropout_p = dropout

    
    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x,p= self.dropout_p,  training=self.training)
        x = self.conv2(x, edge_index)

        return x