"""
Note that different compilers will output different plots.
The sample comparison_noisy.pdf is the result of Python IDLE 3.8.3.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def create_and_split(train_size, noise_seed, init_seed, var=1.0, noise_rate=0.005):
    np.random.seed(init_seed)
    red = np.random.normal(loc=center_red, size=(num, 2), scale=var)
    blue = np.random.normal(loc=center_blue, size=(num, 2), scale=var)
    _min = min(np.amin(red), np.amin(blue)) * 1.1
    _max = max(np.amax(red), np.amax(blue)) * 1.1

    # create dataset
    features = np.concatenate((red, blue), axis = 0)
    labels = np.concatenate((np.ones(num), np.zeros(num)))

    # split MS
    train_idx_red = np.random.choice(range(num), size=int(num * train_size), replace=False)
    train_idx_blue = np.random.choice(range(num, 2*num), size=int(num * train_size), replace=False)
    train_idx = np.concatenate((train_idx_red, train_idx_blue))
    test_idx = np.array(list(set(range(2*num)) - set(train_idx)))

    # create clean or noisy adjacency matrix
    adj = np.zeros((2*num, 2*num))
    for i in range(num):
        for idx in np.random.choice(range(num - 1), 1, replace=False):
            adj[idx if idx < i else i][i if idx < i else idx + 1] = 1
        for idx in np.random.choice(range(num - 1), 1, replace=False):
            adj[(idx if idx < i else i) + num][(i if idx < i else idx + 1) + num] = 1

    np.random.seed(noise_seed)
    bad_edges = np.random.random(size=(2*num, 2*num))
    for i in range(num):
        for j in range(num, 2*num):
            if bad_edges[i][j] < noise_rate:
                adj[i][j] = 1

    adj = adj + np.transpose(adj) + np.eye(2*num)
    return adj, features, labels, train_idx, test_idx, _min, _max

# Perform graph convolutional operation (GCO) with depth K
def GCN(K, adj, features):
    features_GCN = features.copy()
    for i in range(K):
        degree = np.matmul(adj, np.ones((2*num, 1)))
        degree_sqrt = np.sqrt(degree)
        features_GCN = np.divide(features_GCN, degree_sqrt)
        features_GCN = np.matmul(adj, features_GCN)
        features_GCN = np.divide(features_GCN, degree_sqrt)
    return features_GCN

# Perform K-hop MC sampling recursively
def MCGL(K, train_idx, adj):
    train_idx_dict = {'red': [], 'blue': []}
    def find_one(i, left, label):
        if left == 1:
            for j in range(2*num):
                if j != i and adj[i][j] == 1:
                    train_idx_dict[label].append(j)
        else:
            for j in range(2*num):
                if j != i and adj[i][j] == 1:
                    find_one(j, left - 1, label)
    for idx in train_idx:
        find_one(idx, K, 'red' if idx < num else 'blue')
    train_idx_red = set([i for i in train_idx if i < num]) | set(train_idx_dict['red'])
    train_idx_blue = set([i for i in train_idx if i >= num]) | set(train_idx_dict['blue'])
    return train_idx_red & train_idx_blue, train_idx_red | train_idx_blue

# Plot a full row of three graphs
def plot_row(row, size, train_size, noise_rate=0.005, noise_seed=1, init_seed=2, has_return=True):
    ax = fig.add_subplot(gs[row, 0])
    ax_GCN_1 = fig.add_subplot(gs[row, 1])
    ax_MCGL_1 = fig.add_subplot(gs[row, 2])

    # get data
    adj, features, labels, train_idx, test_idx, _min, _max = create_and_split(train_size, noise_seed, init_seed, noise_rate=noise_rate)

    # Plot three graphs
    plot_all(adj, features, train_idx, ax, size, _min, _max)
    plot_all(adj, GCN(1, adj, features), train_idx, ax_GCN_1, size, _min, _max)
    train_idx_dup, train_idx_MCGL = MCGL(1, train_idx, adj)
    plot_all(adj, features, train_idx_MCGL, ax_MCGL_1, size, _min, _max, is_last_column=True, train_idx_dup=train_idx_dup)

    if has_return:
        return ax, ax_GCN_1, ax_MCGL_1

def plot_all(adj, features, train_idx, ax, size, _min, _max, is_last_column=False, train_idx_dup=[]):
    ax.scatter(features[:num,0], features[:num,1], facecolor='white', edgecolor='red', s=size, zorder=2, label='test set with label=0')
    ax.scatter(features[num:,0], features[num:,1], facecolor='white', edgecolor='blue', s=size, zorder=2, label='test set with label=1')
    ax.scatter([features[i][0] for i in train_idx if i < num], [features[i][1] for i in train_idx if i < num], color='red', s=size, zorder=3, label='train set with label=0')
    ax.scatter([features[i][0] for i in train_idx if i >= num], [features[i][1] for i in train_idx if i >= num], color='blue', s=size, zorder=3, label='train set with label=1')
    for i in train_idx_dup:
        ax.scatter([features[i][0]], [features[i][1]], color='lime', s=size, zorder=4)
    ax.plot([_min, center_blue + center_red - _min], [center_blue + center_red - _min, _min], c='lightgreen', linewidth=linewidth, zorder=0)
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)

    for i in range(2*num):
        for j in range(i+1, 2*num):
            if adj[i][j] == 1:
                if (i < num) ^ (j < num):
                    ax.plot([features[i][0], features[j][0]], [features[i][1], features[j][1]], c='gold', zorder=1)
                elif i < num:
                    ax.plot([features[i][0], features[j][0]], [features[i][1], features[j][1]], c='pink', zorder=1)
                else:
                    ax.plot([features[i][0], features[j][0]], [features[i][1], features[j][1]], c='lightblue', zorder=1)

    ax.set_xticks([center_red, center_blue])
    ax.set_yticks([center_red, center_blue])
    ax2 = ax.twinx()
    ax2.set_ylim(_min, _max)
    ax2.set_yticks([center_red, center_blue])
    if is_last_column:
        ax2.set_yticklabels([center_red, center_blue])
    else:
        ax2.set_yticklabels(['', ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot transformation of graphs with different characteristics')
    parser.add_argument('--num', type=int, default=30,
                        help='the number of points in each class')
    parser.add_argument('--center_red', type=float, default=-1,
                        help='Set the central position of red points as (center_red, center_red)')
    parser.add_argument('--center_blue', type=float, default=1,
                        help='Set the central position of blue points as (center_blue, center_blue)')
    args = parser.parse_args(sys.argv[1:])
    if args.num < 1:
        sys.exit('error: num set as {}'.format(args.num))
    num = args.num
    center_red = args.center_red
    center_blue = args.center_blue

    # Parameters of figure
    figsize = (12, 4)
    fontsize = 14
    titlesize = 20
    linewidth = 3
    y = -0.27
    np.random.seed(2)
    var = 1

    # Create subplots and plot three graphs in a row
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    ax, ax_GCN_1, ax_MCGL_1 = plot_row(0, 30, 0.15, 0.005, has_return=True)
    ax.set_title('(a) Original graph', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_GCN_1.set_title('(b) After one-layer GCN', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_MCGL_1.set_title('(c) After one-hop MCGL', y=y, fontsize=titlesize, fontname='Times New Roman')

    handles, labels = ax_MCGL_1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5,1.017))

    fig.subplots_adjust(bottom=0.2)

    fig.show()
    fig.savefig('comparison_noisy.pdf')
