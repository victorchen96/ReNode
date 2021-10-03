import os,sys
import time
import logging
import yaml
import math
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn.functional as F

from pprgo import utils, ppr
from pprgo.pprgo import PPRGo
from pprgo.train import train
from pprgo.predict import predict
from pprgo.dataset import PPRDataset
import random

import warnings
warnings.simplefilter("ignore")

#set logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
        fmt='%(asctime)s (%(levelname)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

# read config
with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)

for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

data_file           = config['data_file']           # Path to the .npz data file
split_seed          = config['split_seed']          # Seed for splitting the dataset into train/val/test
ntrain_div_classes  = config['ntrain_div_classes']  # Number of training nodes divided by number of classes
attr_normalization  = config['attr_normalization']  # Attribute normalization. Not used in the paper

alpha               = config['alpha']               # PPR teleport probability
eps                 = config['eps']                 # Stopping threshold for ACL's ApproximatePR
topk                = config['topk']                # Number of PPR neighbors for each node
ppr_normalization   = config['ppr_normalization']   # Adjacency matrix normalization for weighting neighbors

hidden_size         = config['hidden_size']         # Size of the MLP's hidden layer
nlayers             = config['nlayers']             # Number of MLP layers
weight_decay        = config['weight_decay']        # Weight decay used for training the MLP
dropout             = config['dropout']             # Dropout used for training

lr                  = config['lr']                  # Learning rate
max_epochs          = config['max_epochs']          # Maximum number of epochs (exact number if no early stopping)
batch_size          = config['batch_size']          # Batch size for training
batch_mult_val      = config['batch_mult_val']      # Multiplier for validation batch size

eval_step           = config['eval_step']           # Accuracy is evaluated after every this number of steps
run_val             = config['run_val']             # Evaluate accuracy on validation set during training

early_stop          = config['early_stop']          # Use early stopping
patience            = config['patience']            # Patience for early stopping

nprop_inference     = config['nprop_inference']     # Number of propagation steps during inference
inf_fraction        = config['inf_fraction']        # Fraction of nodes for which local predictions are computed during inference


base_w              = config['base_w']              # the base  value for ReNode re-weighting; value set to [0.25,0.5,0.75,1]
scale_w             = config['scale_w']             # the scale value for ReNode re-weighting; value set to [1.5 ,1  ,0.5 ,0]  
issue_type          = config['issue_type']          # whether the training set is quantity-balanced; value set to ['tinl','qinl']          
 
split_seed          = config['split_seed']          # Seed for splitting the dataset into train/val/test
init_seed           = config['init_seed']           # Seed for initial the training setting
gpu_device          = config['device']              # The running device

# running parameter
#split_seed = int(sys.argv[1])
#init_seed  = int(sys.argv[2])


#initial training setting
np.random.seed(init_seed)
torch.manual_seed(init_seed)
random.seed(init_seed)
torch.cuda.manual_seed(init_seed)


#loading data
start = time.time()
(adj_matrix, attr_matrix, labels,
 train_idx, val_idx, test_idx) = utils.get_data(
        f"{data_file}",
        seed=split_seed,
        ntrain_div_classes=ntrain_div_classes,
        normalize_attr=attr_normalization,
        issue_type = issue_type,

)
try:
    d = attr_matrix.n_columns
except AttributeError:
    d = attr_matrix.shape[1]
nc = labels.max() + 1
time_loading = time.time() - start
print(f"Loading Data Runtime: {time_loading:.2f}s")


#calculating the personalized pagerank matrix
start = time.time()
topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                 normalization=ppr_normalization)
train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
if run_val:
    topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                   normalization=ppr_normalization)
    val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
else:
    val_set = None
time_preprocessing = time.time() - start
print(f"Preprocessing Data (including computing PPR Matrix) Runtime: {time_preprocessing:.2f}s")

#calculating the Totoro value
ppr_dense= topk_train.todense()
ppr_dense = torch.tensor(ppr_dense).float()
ppr_dense[ppr_dense.eq(0)] = float('-inf')
ppr_dense = F.softmax(ppr_dense,dim=1)


train_labels = labels[train_idx]
gpr_dense = torch.zeros((nc,labels.shape[0])).float()

for iter_c in range(nc):
    iter_where = np.where(train_labels==iter_c)[0]
    iter_mean  = torch.mean(ppr_dense[iter_where],dim=0)
    gpr_dense[iter_c] = iter_mean


#calculating the ReNode weight
gpr_dense = gpr_dense.transpose(0,1)
gpr_sum = torch.sum(gpr_dense,dim=1)
gpr_idx = F.one_hot(torch.tensor(train_labels).long(),nc).float()

gpr_rn = gpr_sum.unsqueeze(1) - gpr_dense
rn_dense = torch.mm(ppr_dense,gpr_rn)
rn_value = torch.sum(rn_dense * gpr_idx,dim=1)

totoro_list = rn_value.tolist()
nnode = len(totoro_list)
train_size = len(totoro_list)

id2totoro = {i:totoro_list[i] for i in range(len(totoro_list))}
sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
id2rank = {sorted_totoro[i][0]:i for i in range(nnode)}
totoro_rank = [id2rank[i] for i in range(nnode)]

rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)


# setting model
start = time.time()
model = PPRGo(d, nc, hidden_size, nlayers, dropout)
device = torch.device('cuda:{}'.format(gpu_device))
model.to(device)
rn_weight = rn_weight.to(device)


# training
nepochs, _, _ = train(
        model=model, train_set=train_set, val_set=val_set,
        lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
        eval_step=eval_step, early_stop=early_stop, patience=patience,rn_weight = rn_weight)
time_training = time.time() - start
logging.info('Training done.')
print(f"Training Runtime: {time_training:.2f}s")

# inferencing
start = time.time()
predictions, time_logits, time_propagation = predict(
        model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
        nprop=nprop_inference, inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization)
time_inference = time.time() - start
print(f"Inferencing Runtime: {time_inference:.2f}s")


# calculating metric
wf_train = f1_score(labels[train_idx], predictions[train_idx], average='weighted')
wf_val   = f1_score(labels[val_idx],   predictions[val_idx],   average='weighted')
wf_test  = f1_score(labels[test_idx],  predictions[test_idx],  average='weighted')

f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
f1_val   = f1_score(labels[val_idx], predictions[val_idx], average='macro')
f1_test  = f1_score(labels[test_idx], predictions[test_idx], average='macro')

gpu_memory = torch.cuda.max_memory_allocated()
memory = utils.get_max_memory_bytes()

time_total = time_preprocessing + time_training + time_inference

print(f'''

Weighted-F1 score: Train: {wf_train:.3f}, val: {wf_val:.3f}, test: {wf_test:.3f}\n
Macro-F1 score:    Train: {f1_train:.3f}, val: {f1_val:.3f}, test: {f1_test:.3f}

Runtime: Preprocessing: {time_preprocessing:.2f}s, training: {time_training:.2f}s, inference: {time_inference:.2f}s -> total: {time_total:.2f}s
Memory: Main: {memory / 2**30:.2f}GB, GPU: {gpu_memory / 2**30:.3f}GB
''')


