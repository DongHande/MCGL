import torch.nn as nn
import torch as t
import math
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=True):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(t.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(t.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = t.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Conlayer(nn.Module):#D^(-1/2)AD^(-1/2)X
    def __init__(self, dropout=0):
        super(Conlayer, self).__init__()
        self.dropout = dropout

    def forward(self, input, adj):
        if t.cuda.is_available():
            degree = t.matmul(adj, t.ones(adj.size(0), 1).cuda())
        else:
            degree = t.matmul(adj, t.ones(adj.size(0), 1))
        degree_sqrt = t.sqrt(degree)
        input = F.dropout(input, self.dropout, training=self.training)
        input = input.div(degree_sqrt)
        output = t.matmul(adj, input)
        output = output.div(degree_sqrt)
        return output


class Avelayer(nn.Module):#D^(-1)AX ~ graph aggragate networks
    def __init__(self, dropout):
        super(Avelayer, self).__init__()
        self.dropout = dropout

    def forward(self, input, adj):
        if t.cuda.is_available():
            degree = t.matmul(adj, t.ones(adj.size(0), 1).cuda())
        else:
            degree = t.matmul(adj, t.ones(adj.size(0), 1))
        input = F.dropout(input, self.dropout, training=self.training)
        output = t.matmul(adj, input)
        output = output.div(degree)
        return output

