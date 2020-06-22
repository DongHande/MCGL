"""
Note that different compilers will output different plots.
The sample deep_GCO.pdf is the result of Python IDLE 3.8.3.
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

    # split data
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
        for j in range(num, 2 * num):
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

# Plot a full row of four graphs
def plot_row(row, size, train_size, noise_rate=0.005, noise_seed=1, init_seed=3, has_return=False):
    ax = fig.add_subplot(gs[row, 0])
    ax_GCN_1 = fig.add_subplot(gs[row, 1])
    ax_GCN_2 = fig.add_subplot(gs[row, 2])
    ax_GCN_inf = fig.add_subplot(gs[row, 3])

    # get data
    adj, features, labels, train_idx, test_idx, _min, _max = create_and_split(train_size, noise_seed, init_seed, noise_rate=noise_rate)

    # Plot four graphs
    plot_all(adj, features, train_idx, ax, 'a'+str(row+1), size, _min, _max)
    plot_all(adj, GCN(1, adj, features), train_idx, ax_GCN_1, 'b'+str(row+1), size, _min, _max)
    plot_all(adj, GCN(2, adj, features), train_idx, ax_GCN_2, 'c'+str(row+1), size, _min, _max)
    plot_all(adj, GCN(inf, adj, features), train_idx, ax_GCN_inf, 'd'+str(row+1), size, _min, _max, is_last_column=True)

    if has_return:
        return ax, ax_GCN_1, ax_GCN_2, ax_GCN_inf

def plot_all(adj, features, train_idx, ax, text, size, _min, _max, is_last_column=False):
    ax.text(_max*pos_x, _min*pos_y, text, fontsize=textsize, fontname='Times New Roman')
    ax.scatter(features[:num,0], features[:num,1], facecolor='white', edgecolor='red', s=size, zorder=2, label='test set with label=0')
    ax.scatter(features[num:,0], features[num:,1], facecolor='white', edgecolor='blue', s=size, zorder=2, label='test set with label=1')
    ax.scatter([features[i][0] for i in train_idx if i < num], [features[i][1] for i in train_idx if i < num], color='red', s=size, zorder=3, label='train set with label=0')
    ax.scatter([features[i][0] for i in train_idx if i >= num], [features[i][1] for i in train_idx if i >= num], color='blue', s=size, zorder=3, label='train set with label=1')
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
    parser.add_argument('--inf', type=int, default=100,
                        help='the "infinite" depth of GCN*')
    parser.add_argument('--center_red', type=float, default=-1,
                        help='Set the central position of red points as (center_red, center_red)')
    parser.add_argument('--center_blue', type=float, default=1,
                        help='Set the central position of blue points as (center_blue, center_blue)')
    args = parser.parse_args(sys.argv[1:])
    if args.num < 1:
        sys.exit('error: num set as {}'.format(args.num))
    if args.inf < 3:
        sys.exit('error: inf set as {}'.format(args.inf))
    num = args.num
    inf = args.inf
    center_red = args.center_red
    center_blue = args.center_blue

    # Parameters of figure
    figsize = (12, 6.8)
    fontsize = 14
    textsize = 20
    titlesize = 20
    linewidth = 3
    y = -0.27
    pos_x = 0.75
    pos_y = 0.9
    var = 1

    # Create subplots and plot two rows
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4)
    plot_row(0, 30, 0.15, 0)
    ax, ax_GCN_1, ax_GCN_2, ax_GCN_inf = plot_row(1, 30, 0.15, 0.005, has_return=True)
    ax.set_title('(a) Original graph', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_GCN_1.set_title('(b) K = 1', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_GCN_2.set_title('(c) K = 2', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_GCN_inf.set_title('(d) K = {}'.format(inf), y=y, fontsize=titlesize, fontname='Times New Roman')

    handles, labels = ax_GCN_inf.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5, 0.96))

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.11)

    fig.show()
    fig.savefig('deep_GCO.pdf')
