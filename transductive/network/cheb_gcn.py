import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_geometric.nn import ChebConv


class ChebGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(ChebGCN1, self).__init__()
        chebgcn_para1 = 2
        self.conv1 = ChebConv(nfeat, nclass,chebgcn_para1)

    def forward(self, x, adj):
        edge_index = adj
 
        x = F.relu(self.conv1(x, edge_index))
    
        return x

class ChebGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(ChebGCN2, self).__init__()
        chebgcn_hops = 2
        self.conv1 = ChebConv(nfeat, nhid,chebgcn_hops)
        self.conv2 = ChebConv(nhid, nclass,chebgcn_hops)

        self.dropout_p = dropout
      
    def forward(self, x, adj):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)


        return x

class ChebGCNX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(ChebGCNX, self).__init__()
        chebgcn_para1 = 2
        self.conv1 = ChebConv(nfeat, nhid,chebgcn_para1)
        self.conv2 = ChebConv(nhid, nclass,chebgcn_para1)
        self.convx = nn.ModuleList([ChebConv(nhid, nhid,chebgcn_para1) for _ in range(nlayer-2)])
        self.dropout_p = dropout

    
    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p,training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, p= self.dropout_p,training=self.training)
        x = self.conv2(x, edge_index)

        return x


