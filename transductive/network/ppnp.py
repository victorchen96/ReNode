import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math


class PPNP1(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass,dropout,nlayer=2):
        super(PPNP1, self).__init__()
        self.line1 = nn.Linear(nfeat,nclass) 
        self.dropout_p = dropout

    def norm_adj(self,opt,data):

        self.Pi = data.Pi.to(opt.device)

        if opt.ppr_topk>0:
            vk,vi = torch.topk(self.Pi,opt.ppr_topk,dim=-1)
            tk = vk[:,-1].unsqueeze(1).expand_as(self.Pi)
            mask_k = torch.lt(self.Pi,tk)
            self.Pi[mask_k] = 0

    def forward(self,X,A):
        H  = self.line1(X)
        H  = torch.mm(self.Pi,H)

        return H


class PPNP2(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass,dropout,nlayer=2):
        super(PPNP2, self).__init__()
        self.line1 = nn.Linear(nfeat,nhid) 
        self.line2 = nn.Linear(nhid,nclass)
        self.dropout_p = dropout

    def norm_adj(self,opt,data):

        self.Pi = data.Pi.to(opt.device)

        if opt.ppr_topk>0:
            vk,vi = torch.topk(self.Pi,opt.ppr_topk,dim=-1)
            tk = vk[:,-1].unsqueeze(1).expand_as(self.Pi)
            mask_k = torch.lt(self.Pi,tk)
            self.Pi[mask_k] = 0


    def forward(self,X,A):

        H  = self.line1(X)
        H  = F.relu(H)
        H  = F.dropout(H, p = self.dropout_p, training = self.training)
        H  = self.line2(H) 
        H  = torch.mm(self.Pi,H)

        return H

class PPNPX(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass,dropout,nlayer=3):
        super(PPNPX, self).__init__()
        self.line1 = nn.Linear(nfeat,nhid) 
        self.line2 = nn.Linear(nhid,nclass)
        self.linex = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(nlayer-2)]) 
        self.dropout_p = dropout

    def norm_adj(self,opt,data):

        self.Pi = data.Pi.to(opt.device)

        if opt.ppr_topk>0:
            vk,vi = torch.topk(self.Pi,opt.ppr_topk,dim=-1)
            tk = vk[:,-1].unsqueeze(1).expand_as(self.Pi)
            mask_k = torch.lt(self.Pi,tk)
            self.Pi[mask_k] = 0


    def forward(self,X,A):

        H  = self.line1(X)
        H  = F.relu(H)

        for iter_layer in self.linex:
            H = F.dropout(H, p = self.dropout_p, training = self.training)
            H = F.relu(iter_layer(H))

        H  = F.dropout(H, p = self.dropout_p, training = self.training)
        H  = self.line2(H) 
        H  = torch.mm(self.Pi,H)

        return H




