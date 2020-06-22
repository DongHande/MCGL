import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import random
import data.coauthorship.io as io
import os

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_ms_academic():
    dataset = io.load_dataset('ms_academic')
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = dataset.adj_matrix
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = dataset.labels

    # split by class
    split_by_class = [[] for i in range(len(dataset.class_names))]
    for i in range(len(labels)):
        split_by_class[labels[i]].append(i)

    # training set
    num_train = 20
    idx_train = np.concatenate([np.random.choice(each_class, num_train, replace=False) for each_class in split_by_class])

    # validation set
    num_val = 500
    idx_val = np.random.choice([i for i in range(dataset.attr_matrix.shape[0]) if not i in idx_train], num_val, replace=False)

    # test set
    idx_test = [i for i in range(dataset.attr_matrix.shape[0]) if (not i in idx_train) and (not i in idx_val)]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data(dataset_str='cora'):
    print('Loading {} dataset...'.format(dataset_str))
    if dataset_str == "ms_academic":
        return load_data_ms_academic()
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/citation/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/citation/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty_extended[list(set(range(min(test_idx_reorder), max(test_idx_reorder) + 1)) - set(test_idx_range)) - min(
            test_idx_range), 0] = 1
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + sp.eye(adj.shape[0])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # training/validation/test set
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_graph(adj):
    indices = adj._indices().numpy()
    values = adj._values().numpy()
    num_node = adj.shape[0]
    graph = []
    for i in range(num_node):
        graph.append([])
    for edge in indices.T:
        graph[edge[0]].append(edge[1])
    return graph

def graph_sample(idx_train, batch_size, graph, depth, features, labels):
    idx = []
    for i in range(batch_size):
        idx.append(int(random.sample(list(idx_train), 1)[0]))
    y = labels[idx]
    for i in range(len(idx)):
        for j in range(depth):
            idx[i] = random.sample(graph[idx[i]], 1)[0]
    x = features[idx]
    return x, y

def write_file(file = 'outfile.csv', y = -10000):
    with open(file, 'a') as f:
        f.write('%.4f,' % y)

# def reduce_noise(adj, labels, rate):
#     _adj = adj.copy()
#     bad_edges = []
#     for i in range(adj.shape[0]):
#         for j in range(i+1, adj.shape[1]):
#             if labels[i].item() != labels[j].item():
#                 bad_edges.append([i, j])
#     for i_j in np.random.choice(bad_edges, int(len(bad_edges)*rate)):
#         _adj[i_j[0]][i_j[1]] = 0
#         _adj[i_j[1]][i_j[0]] = 0
#     return _adj

def reduce_noise(adj, labels, noise_rate):
    indices = adj._indices().numpy()
    upper = indices[0,:] > indices[1,:]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    # keep_rate: the bad edge ratio left.
    keep_rate = noise_rate / ( bad_num / upper_indices.shape[1] )
    keep_flag = []
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() == labels[j].item():
            keep_flag.append(True)
        else:
            if np.random.rand(1) < keep_rate:
                keep_flag.append(True)
            else:
                keep_flag.append(False)

    upper_ind_left = upper_indices[:, keep_flag]
    node = adj.size()[0]
    ind_lef = np.concatenate((upper_ind_left, upper_ind_left[[1,0],:],
                              np.array([np.arange(0, node), np.arange(0, node)])), axis=1)

    ind_lef = torch.LongTensor(ind_lef)
    values = torch.FloatTensor(np.ones((ind_lef.shape[1])))

    _adj = torch.sparse.FloatTensor(ind_lef, values, adj.size())

    return _adj

def get_noise_rate(adj, labels):
    indices = adj._indices().numpy()
    upper = indices[0, :] > indices[1, :]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    return bad_num / upper_indices.shape[1]