import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from layers import Linear, Conlayer

class MLP(nn.Module):#
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.Linear1 = Linear(nfeat, nhid, dropout)
        self.Linear2 = Linear(nhid, nclass, dropout)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        return F.log_softmax(self.Linear2(x), dim=1)

class GCN(nn.Module):# GCN  operation
    def __init__(self, nfeat, nhid, nclass, dropout, depth, baseline):
        super(GCN, self).__init__()

        self.baseline = baseline
        self.Linear1 = Linear(nfeat, nhid, dropout)
        self.Linear2 = Linear(nhid, nclass, dropout)
        self.conlayer1 = Conlayer()
        self.conlayer2 = Conlayer()
        self.conlayers = []
        for i in range(depth):
            self.conlayers.append(Conlayer())

    def forward(self, x, adj):
        if self.baseline == 1:
            x = F.relu(self.conlayer1(self.Linear1(x), adj))
            x = F.softmax(self.conlayer2(self.Linear2(x), adj), dim=1)
            return t.log(x)
        elif self.baseline == 2:
            x = F.relu(self.Linear1(x))
            x = self.Linear2(x)
            for conlayer in self.conlayers:
                x = conlayer(x, adj)
            x = F.softmax(x, dim=1)
            return t.log(x)
        else:
            raise RuntimeError("invalid baseline: baseline set as {}".format(self.baseline))

def graph_ave(output, adj):

    if t.cuda.is_available():
        degree = t.matmul(adj, t.ones(adj.size(0), 1).cuda())
    else:
        degree = t.matmul(adj, t.ones(adj.size(0), 1))
    output = t.matmul(adj, output)
    output = output.div(degree)
    return output


