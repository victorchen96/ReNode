import torch
import torch.nn as nn
import torch.nn.functional as F

from network.gcn import StandGCN1,StandGCN2,StandGCNX
from network.gat import StandGAT1,StandGAT2,StandGATX
from network.cheb_gcn import ChebGCN1,ChebGCN2,ChebGCNX
from network.sage import GraphSAGE1,GraphSAGE2,GraphSAGEX
from network.ppnp import PPNP1,PPNP2,PPNPX
from network.sgc import SGC1,SGC2,SGCX


def get_model(opt):

    nfeat = opt.num_feature
    nclass = opt.num_class
    nhid = opt.num_hidden
    nlayer = opt.num_layer
        
    dropout = opt.dropout
    model_opt = opt.model

    model_dict = {
        'gcn'  : [StandGCN1,StandGCN2,StandGCNX],
        'gat'  : [StandGAT1,StandGAT2,StandGATX],
        'cheb' : [ChebGCN1,ChebGCN2,ChebGCNX],
        'sage' : [GraphSAGE1,GraphSAGE2,GraphSAGEX],
        'ppnp' : [PPNP1,PPNP2,PPNPX],
        'sgc'  : [SGC1,SGC2,SGCX],
    }
    model_list = model_dict[model_opt]
    
    if nlayer==1:
        model = model_list[0](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    elif nlayer ==2:
        model = model_list[1](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    else:
        model = model_list[2](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)  

    return model.to(opt.device)
