import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_geometric.nn import SGConv


class SGC1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(SGC1, self).__init__()
        self.conv1 = SGConv(nfeat, nclass,K=1,cached=True)

    def forward(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
 
        return x

class SGC2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(SGC2, self).__init__()
        self.conv1 = SGConv(nfeat, nclass,K=2,cached=True)

    def forward(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
 
        return x

class SGCX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(SGCX, self).__init__()
        self.conv1 = SGConv(nfeat, nclass,K=nlayer,cached=True)

    def forward(self, x, adj):
        edge_index = adj
        x = self.conv1(x, edge_index)
 
        return x

