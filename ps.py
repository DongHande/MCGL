import argparse

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=int, default=1,
                        help='Choose GCN baseline. 1 refers to GCN. 2 refers to GCN*.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Set the batch size of MCGL')
    parser.add_argument('--dataset', type=str, default='citeseer',
                        help='Choose from {cora, citeseer, pubmed, ms_academic}')
    parser.add_argument('--depth', type=int, default=2,
                        help='the depth of GCN model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs to train.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--noise_rate', type=float, default=1.0,
                        help='Reduce the noise rate to some point. Set it as 1.0 to keep the original noise rate.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--trdep', type=int, default=2,
                        help='Set the depth of sampling tree when training')
    parser.add_argument('--tsdep', type=int, default=2,
                        help='Set the depth of inference tree when testing')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Set weight decay (L2 loss on parameters).')
    return parser.parse_args()
