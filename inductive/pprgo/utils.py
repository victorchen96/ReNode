import resource
import numpy as np
import scipy.sparse as sp
import sklearn
import time

from .sparsegraph import load_from_npz


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.shape[1]]

        return sp.csr_matrix((data, indices, indptr), shape=shape)


def split_random(seed, n, n_train, n_val):
    np.random.seed(seed)

    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx


def eq_split_random(seed,n,train_each,valid_each,all_label,nclass=8):
    np.random.seed(seed)
    all_idx = np.random.permutation(n).tolist()

    all_label = all_label.tolist()

    train_list = [0 for _ in range(nclass)]
    train_idx = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break

    assert sum(train_list)==train_each*nclass

    print(train_list)
    
    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))


    train_idx = np.sort(train_idx)
    valid_idx = np.sort(valid_idx)
    test_idx  = np.sort(test_idx)

    return train_idx,valid_idx,test_idx



def get_data(dataset_path, seed, ntrain_div_classes, normalize_attr=None,issue_type='qinl'):
    '''
    Get data from a .npz-file.

    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    normalize_attr
        Normalization scheme for attributes. By default (and in the paper) no normalization is used.
    issue_type
        whether to take the quantity-balance sampling for training set
        tinl: quantity-balanc; qinl: quantity-imbalance

    '''
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(g.attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(g.attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(g.attr_matrix):
            attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        attr_matrix = g.attr_matrix

    # helper that speeds up row indexing
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)
    else:
        attr_matrix = attr_matrix

    # split the data into train/val/test
    num_classes = g.labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = int(n_train*1.5)


    if issue_type == 'qinl':
        train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)

    elif issue_type == 'tinl':
        train_idx, val_idx, test_idx = eq_split_random(seed, n, ntrain_div_classes, int(1.5*ntrain_div_classes), g.labels, num_classes)


    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx


def get_max_memory_bytes():
    return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
