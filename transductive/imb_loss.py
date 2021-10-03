import numpy as np
# from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import random
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss,dim=1)

    return focal_loss

class IMB_LOSS:
    def __init__(self,loss_name,opt,data):
        self.loss_name = loss_name
        self.device    = opt.device
        self.cls_num   = data.num_classes
        
        train_size = [len(x) for x in data.train_node]
        train_size_arr = np.array(train_size)
        train_size_mean = np.mean(train_size_arr)
        train_size_factor = train_size_mean / train_size_arr
        
        #alpha in re-weight
        self.factor_train = torch.from_numpy(train_size_factor).type(torch.FloatTensor)
        
        #gamma in focal
        self.factor_focal = opt.factor_focal

        #beta in CB
        weights = torch.from_numpy(np.array([1.0 for _ in range(self.cls_num)])).float()

        if self.loss_name == 'focal':
            weights = self.factor_train

        if self.loss_name == 'cb-softmax':
            beta = opt.factor_cb
            effective_num = 1.0 - np.power(beta, train_size_arr)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.cls_num
            weights = torch.tensor(weights).float()

        self.weights = weights.unsqueeze(0).to(opt.device)



    def compute(self,pred,target):

        if self.loss_name == 'ce':
            return F.cross_entropy(pred,target,weight=None,reduction='none')

        elif self.loss_name == 're-weight':
            return F.cross_entropy(pred,target,weight=self.factor_train.to(self.device),reduction='none')

        elif self.loss_name == 'focal':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            return focal_loss(labels_one_hot,pred,weights,self.factor_focal)

        elif self.loss_name == 'cb-softmax':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            pred = pred.softmax(dim = 1)
            temp_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights,reduction='none') 
            return torch.mean(temp_loss,dim=1)

        else:
            raise Exception("No Implentation Loss")


        







        
