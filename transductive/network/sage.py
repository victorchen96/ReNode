import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np
from torch_geometric.nn import SAGEConv


class GraphSAGE1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(GraphSAGE1, self).__init__()
        self.conv1 = SAGEConv(nfeat, nclass)

    def forward(self, x, adj):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
        return x


class GraphSAGE2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(GraphSAGE2, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)

        self.dropout_p = dropout
      
    def forward(self, x, adj):

        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x
    

class GraphSAGEX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(GraphSAGEX, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        self.convx = nn.ModuleList([SAGEConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)

        return x
