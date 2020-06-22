'''
This is a simple implementation of MCGL.
The base model is MLP, and the distribution is uniform distribution.
'''

import time
import ps
import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, adj_to_graph, graph_sample, write_file, reduce_noise
from models import MLP, graph_ave
import random

para_file = 'para/best_base.pkl'

args = ps.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load MS
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
adj = reduce_noise(adj, labels, noise_rate=args.noise_rate)
graph = adj_to_graph(adj)

# Model and optimizer
model = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

def train(iteration):
    t = time.time()

    x_batch, y_batch = graph_sample(idx_train, args.batch_size, graph, args.trdep, features, labels)
    model.train()
    optimizer.zero_grad()
    y_pre_batch = model(x_batch)
    loss_train = F.nll_loss(y_pre_batch, y_batch)
    acc_train = accuracy(y_pre_batch, y_batch)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)
        for j in range(args.tsdep):
            output = graph_ave(output, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if iteration % 100 == 0:
        print('iteration: {:04d}'.format(iteration+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time_iter: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def test():
    model.eval()
    output = model(features)
    for j in range(args.tsdep):
        output = graph_ave(output, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best_loss = np.inf
best_itr = 0

for iteration in range(args.iterations):
    loss_validation = train(iteration)
    loss_values.append(loss_validation)

    if loss_values[-1] < best_loss:
        best_itr = iteration
        best_loss = loss_values[-1]
        bad_counter = 0
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading the best model', best_itr)
model.load_state_dict(best_state_dict)
acc_test = test()

