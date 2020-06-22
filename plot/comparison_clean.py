"""
Note that different compilers will output different plots.
The sample comparison_clean.pdf is the result of Python IDLE 3.8.3.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

# create features with a non-linear boundary
def points(r1, r2):
    result = []
    for i in range(num):
        radius = np.random.uniform(r1, r2)
        radian = np.random.uniform(0, 2*np.pi)
        result.append([radius*np.cos(radian), radius*np.sin(radian)])
    return result

# create a list of communities
def split_community(points, center, color):
    communities = [[] for i in range(5)]
    for idx in range(num):
        d = np.sqrt(np.power(points[idx][0] - center, 2) + np.power(points[idx][1] - center, 2))
        if d < 0.7:
            communities[0].append(idx)
        else:
            if color == 'red':
                communities[(1 if points[idx][0] < center else 0) + (2 if points[idx][1] > center else 0)].append(idx)
            elif color == 'blue':
                communities[(1 if points[idx][0] > center else 0) + (2 if points[idx][1] < center else 0)].append(idx)
    return communities

def create_and_split(train_size, split_seed, init_seed, characteristics, var=1.0, adj_rate=0.3):
    if characteristics == 'non-linear':
        np.random.seed(init_seed)
        red = np.array(points(iface, np.sqrt(2) * iface))
        blue = np.array(points(np.sqrt(2) * iface, np.sqrt(3) * iface))
        _min = -np.sqrt(3) * iface
        _max = np.sqrt(3) * iface

        # create dataset
        features = np.concatenate((red, blue), axis=0)
        labels = np.concatenate((np.ones(num), np.zeros(num)))
        adj = np.zeros((2*num, 2*num))
        for i in range(num):
            for idx in np.random.choice(range(num - 1), 1, replace=False):
                adj[idx if idx < i else i][i if idx < i else idx + 1] = 1
            for idx in np.random.choice(range(num - 1), 1, replace=False):
                adj[(idx if idx < i else i) + num][(i if idx < i else idx + 1) + num] = 1
        adj = adj + np.transpose(adj) + np.eye(2*num)

        # split data
        np.random.seed(split_seed)
        train_idx_red = np.random.choice(range(num), size=int(num * train_size), replace=False)
        train_idx_blue = np.random.choice(range(num, 2*num), size=int(num * train_size), replace=False)
        train_idx = np.concatenate((train_idx_red, train_idx_blue))
        test_idx = np.array(list(set(range(2*num)) - set(train_idx)))

        return adj, features, labels, train_idx, test_idx, _min, _max

    elif characteristics == 'community':
        np.random.seed(init_seed)
        red = np.random.normal(loc=center_red, size=(num, 2), scale=var)
        blue = np.random.normal(loc=center_blue, size=(num, 2), scale=var)
        _min = min(np.amin(red), np.amin(blue)) * 1.1
        _max = max(np.amax(red), np.amax(blue)) * 1.1

        # create dataset
        features = np.concatenate((red, blue), axis=0)
        labels = np.concatenate((np.ones(num), np.zeros(num)))
        adj = np.zeros((2*num, 2*num))
        communities_red = split_community(red, center_red, 'red')
        for community in communities_red:
            community.sort()
            for i in range(len(community) - 1):
                idx_i = community[i]
                for j in range(i + 1, len(community)):
                    idx_j = community[j]
                    if np.random.random() < adj_rate:
                        adj[idx_i][idx_j] = 1
        communities_blue = split_community(blue, center_blue, 'blue')
        for community in communities_blue:
            community.sort()
            for i in range(len(community) - 1):
                idx_i = community[i]
                for j in range(i + 1, len(community)):
                    idx_j = community[j]
                    if np.random.random() < adj_rate:
                        adj[idx_i + num][idx_j + num] = 1
        adj = adj + np.transpose(adj) + np.eye(2*num)

        # split data
        np.random.seed(split_seed)
        train_idx_red = np.concatenate(
            [np.random.choice(communities_red[0], int(num * train_size * 0.8), replace=False),
             np.random.choice(communities_red[1], int(num * train_size) - int(num * train_size * 0.8), replace=False)])
        train_idx_blue = np.concatenate(
            [np.random.choice(communities_blue[0], int(num * train_size * 0.8), replace=False),
             np.random.choice(communities_blue[1], int(num * train_size) - int(num * train_size * 0.8), replace=False)])
        train_idx = np.concatenate([train_idx_red, train_idx_blue + num])
        test_idx = np.array(list(set(range(2*num)) - set(train_idx)))

        return adj, features, labels, train_idx, test_idx, _min, _max

    elif characteristics == 'large_variance':
        np.random.seed(init_seed)
        red = np.random.normal(loc=center_red, size=(num, 2), scale=var)
        blue = np.random.normal(loc=center_blue, size=(num, 2), scale=var)
        _min = min(np.amin(red), np.amin(blue)) * 1.1
        _max = max(np.amax(red), np.amax(blue)) * 1.1

        # create dataset
        features = np.concatenate((red, blue), axis=0)
        labels = np.concatenate((np.ones(num), np.zeros(num)))
        adj = np.zeros((2*num, 2*num))
        for i in range(num):
            for idx in np.random.choice(range(num - 1), 1, replace=False):
                adj[idx if idx < i else i][i if idx < i else idx + 1] = 1
            for idx in np.random.choice(range(num - 1), 1, replace=False):
                adj[(idx if idx < i else i) + num][(i if idx < i else idx + 1) + num] = 1
        adj = adj + np.transpose(adj) + np.eye(2*num)

        # split data
        np.random.seed(split_seed)
        train_idx_red = np.random.choice(range(num), size=int(num * train_size), replace=False)
        train_idx_blue = np.random.choice(range(num, 2*num), size=int(num * train_size), replace=False)
        train_idx = np.concatenate((train_idx_red, train_idx_blue))
        test_idx = np.array(list(set(range(2*num)) - set(train_idx)))

        return adj, features, labels, train_idx, test_idx, _min, _max

def plot_points_and_lines(ax, features, adj, train_idx, size, _min, _max):
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.scatter(features[:num,0], features[:num,1], facecolor='white', edgecolor='red', s=size, zorder=2, label='test set with label=0')
    ax.scatter(features[num:,0], features[num:,1], facecolor='white', edgecolor='blue', s=size, zorder=2, label='test set with label=1')
    ax.scatter([features[i][0] for i in train_idx if i < num], [features[i][1] for i in train_idx if i < num], color='red', s=size, zorder=3, label='train set with label=0')
    ax.scatter([features[i][0] for i in train_idx if i >= num], [features[i][1] for i in train_idx if i >= num], color='blue', s=size, zorder=3, label='train set with label=1')
    for i in range(2*num - 1):
        for j in range(i+1, 2*num):
            if adj[i][j] == 1:
                if i < num:
                    ax.plot([features[i][0], features[j][0]], [features[i][1], features[j][1]], c='pink', zorder=1)
                else:
                    ax.plot([features[i][0], features[j][0]], [features[i][1], features[j][1]], c='lightblue', zorder=1)

def plot_boundary_and_set_ticklabels(ax, characteristics, _min, _max, is_last_column=False):
    if characteristics == 'non-linear':
        theta = np.linspace(0, 2*np.pi, 100)
        x_1 = iface * np.cos(theta)
        y_1 = iface * np.sin(theta)
        ax.plot(x_1, y_1, c='lightgreen', linewidth=linewidth, zorder=0)
        x_2 = np.sqrt(2) * iface * np.cos(theta)
        y_2 = np.sqrt(2) * iface * np.sin(theta)
        ax.plot(x_2, y_2, c='lightgreen', linewidth=linewidth, zorder=0)
        x_3 = np.sqrt(3) * iface * np.cos(theta)
        y_3 = np.sqrt(3) * iface * np.sin(theta)
        ax.plot(x_3, y_3, c='lightgreen', linewidth=linewidth, zorder=0)
        ax.set_xticks([_min, -iface, 0, iface, _max])
        ax.set_yticks([_min, -iface * np.sqrt(2), -iface, 0, iface, iface * np.sqrt(2), _max])
        ax.set_xticklabels(['{:.2f}'.format(_min), int(-iface), 0, int(iface), '{:.2f}'.format(_max)])
        ax.set_yticklabels(['{:.2f}'.format(_min), '{:.2f}'.format(-iface * np.sqrt(2)), int(-iface), 0, int(iface),
                            '{:.2f}'.format(iface * np.sqrt(2)), '{:.2f}'.format(_max)])
        ax2 = ax.twinx()
        ax2.set_ylim(_min, _max)
        ax2.set_yticks([_min, -iface * np.sqrt(2), -iface, 0, iface, iface * np.sqrt(2), _max])
        if is_last_column:
            ax2.set_yticklabels(['{:.2f}'.format(_min), '{:.2f}'.format(-iface * np.sqrt(2)), int(-iface), 0, int(iface),
                                 '{:.2f}'.format(iface * np.sqrt(2)), '{:.2f}'.format(_max)])
        else:
            ax2.set_yticklabels(['', '', '', '', ''])

    elif characteristics == 'community' or characteristics == 'large_variance':
        if center_red != center_blue:
            ax.plot([_min, center_blue + center_red - _min], [center_blue + center_red - _min, _min], c='lightgreen',
                    linewidth=linewidth, zorder=0)
        ax.set_xticks([center_red, center_blue])
        ax.set_yticks([center_red, center_blue])
        ax2 = ax.twinx()
        ax2.set_ylim(_min, _max)
        ax2.set_yticks([center_red, center_blue])
        if is_last_column:
            ax2.set_yticklabels([center_red, center_blue])
        else:
            ax2.set_yticklabels(['', ''])

# Plot a full row of three graphs
def plot_row(row, characteristics, size, train_size, split_seed, var=1.0, has_return=False, init_seed=1):
    ax = fig.add_subplot(gs[row, 0])
    ax_GCN = fig.add_subplot(gs[row, 1])
    ax_MCGL = fig.add_subplot(gs[row, 2])

    # get data
    adj, features, labels, train_idx, test_idx, _min, _max = create_and_split(train_size, split_seed, init_seed, characteristics, var=var)

    # Original plot
    ax.text(_max*pos_x, _min*pos_y, 'a'+str(row+1), fontsize=textsize, fontname='Times New Roman')
    plot_points_and_lines(ax, features, adj, train_idx, size, _min, _max)
    plot_boundary_and_set_ticklabels(ax, characteristics, _min, _max)

    # GCN plot
    degree = np.matmul(adj, np.ones((2*num, 1)))
    degree_sqrt = np.sqrt(degree)
    features_GCN = np.divide(features, degree_sqrt)
    features_GCN = np.matmul(adj, features_GCN)
    features_GCN = np.divide(features_GCN, degree_sqrt)

    ax_GCN.text(_max*pos_x, _min*pos_y, 'b'+str(row+1), fontsize=textsize, fontname='Times New Roman')
    plot_points_and_lines(ax_GCN, features_GCN, adj, train_idx, size, _min, _max)
    plot_boundary_and_set_ticklabels(ax_GCN, characteristics, _min, _max)

    # MCGL plot
    train_idx_adj = []
    for i in train_idx:
        for j in range(num*2):
            if i != j and adj[i][j] == 1:
                train_idx_adj.append(j)
    train_idx_MCGL = np.array(list(set(train_idx) | set(train_idx_adj)))

    ax_MCGL.text(_max*pos_x, _min*pos_y, 'c'+str(row+1), fontsize=textsize, fontname='Times New Roman')
    plot_points_and_lines(ax_MCGL, features, adj, train_idx_MCGL, size, _min, _max)
    plot_boundary_and_set_ticklabels(ax_MCGL, characteristics, _min, _max, is_last_column=True)

    if has_return:
        return ax, ax_GCN, ax_MCGL


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot transformation of graphs with different characteristics')
    parser.add_argument('--num', type=int, default=30,
                        help='the number of points in each class')
    parser.add_argument('--iface', type=float, default=1,
                        help='the radius of the interface concentric circle')
    parser.add_argument('--center_red', type=float, default=-1,
                        help='Set the central position of red points as (center_red, center_red)')
    parser.add_argument('--center_blue', type=float, default=1,
                        help='Set the central position of blue points as (center_blue, center_blue)')
    args= parser.parse_args(sys.argv[1:])
    if args.num < 1:
        sys.exit('error: num set as {}'.format(args.num))
    if args.iface <= 0:
        sys.exit('error: iface set as {}'.format(args.iface))
    num = args.num
    iface = args.iface
    center_red = args.center_red
    center_blue = args.center_blue

    # Parameters of figure
    figsize = (12, 12)
    fontsize = 14
    textsize = 20
    titlesize = 20
    linewidth = 5
    y = -0.27
    pos_x = 0.75
    pos_y = 0.9

    # Create subplots and plot three rows
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3)
    plot_row(0, 'non-linear', size=40, train_size=0.15, split_seed=3, init_seed=2)
    plot_row(1, 'community', var=1, size=35, train_size=0.2, split_seed=4, init_seed=1)
    ax, ax_GCN, ax_MCGL = plot_row(2, 'large_variance', var=1.85, size=35, train_size=0.15, split_seed=1, init_seed=1, has_return=True)
    ax.set_title('(a) Original graph', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_GCN.set_title('(b) After one-layer GCO', y=y, fontsize=titlesize, fontname='Times New Roman')
    ax_MCGL.set_title('(c) After one-hop MCGL', y=y, fontsize=titlesize, fontname='Times New Roman')

    # Place legends
    handles, labels = ax_MCGL.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5, 0.95))

    fig.show()
    fig.savefig('comparison_clean.pdf')
