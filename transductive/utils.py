import numpy as np
import torch
import random
import time
import copy


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def index2dense(edge_index,nnode=2708):

    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj

def index2adj(inf,nnode = 2708):

    indx = inf.numpy()
    print(nnode)
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    return adj

def adj2index(inf):

    where_new = np.where(inf>0)
    new_edge = [where_new[0],where_new[1]]
    new_edge_tensor = torch.from_numpy(np.array(new_edge))
    return new_edge_tensor

def log_opt(opt,log_writer):
    for arg in vars(opt): log_writer.write("{}:{}\n".format(arg,getattr(opt,arg)))

def to_inverse(in_list,t=1):

    in_arr = np.array(in_list)
    in_mean = np.mean(in_arr)
    out_arr = in_mean / in_arr
    out_arr = np.power(out_arr,t)

    return out_arr














