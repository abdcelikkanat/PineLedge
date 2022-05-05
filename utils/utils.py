import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import math
import os

BASE_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

EPS = 1e-6
INF = 1e+6
PI = math.pi
LOG2PI = math.log(2*PI)

softplus = torch.nn.Softplus()

def str2int(text):

    return int(sum(map(ord, text)) % 1e6)


def pair_iter(n, undirected=True):

    if undirected:
        for i in range(n):
            for j in range(i+1, n):
                yield i, j

    else:

        for i in range(n):
            for j in range(i+1, n):
                yield i, j


def pairIdx2flatIdx(i, j, n, undirected=True):

    if undirected:

        return (n-1) * i - int(i*(i+1)/2) + (j-1)

    else:

        return i*n + j



#
# def collate_fn(batch):
#     node_pairs = []
#     events = []
#     for p, e in batch:
#         node_pairs.append(p.numpy())
#         events.append(e)
#
#     return torch.tensor(node_pairs).transpose(0, 1), events

#
# def collate_fn(batch):
#
#     node_indices = []
#     events_dict = dict()
#     for idx, idx_dict in batch:
#         node_indices.append(idx)
#         events_dict[idx] = idx_dict
#
#     assert len(node_indices) > 1, "The batch size must be greater than 1"
#
#     sorted_indices = sorted(node_indices)
#
#     node_pairs = []
#     events = []
#     for i in range(len(sorted_indices)):
#         for j in range(i+1, len(sorted_indices)):
#             u, v = sorted_indices[i], sorted_indices[j]
#             node_pairs.append([u, v])
#             events.append(torch.as_tensor(events_dict[u][v]))
#
#     return torch.tensor(node_pairs).transpose(0, 1), events


def collate_fn(batch):

    node_pairs = []
    event_lists = []

    for items in batch:
        for pair, events in items:
            node_pairs.append(pair)
            event_lists.append(torch.as_tensor(events))

    return torch.as_tensor(node_pairs).transpose(0, 1), event_lists
    #
    # u, v, events = batch[0]
    # #node_pairs = torch.as_tensor([[u], [v]])
    # node_pairs = []
    # events_list = []
    # for u, v, events in batch:
    #     node_pairs.append([u, v])
    #     events_list.append(torch.as_tensor(events))
    # # print(events)
    # node_pairs = torch.as_tensor(node_pairs).transpose(0, 1)
    # # return node_pairs, [torch.as_tensor(events)]
    # return node_pairs, events_list

# def vectorize(x: torch.Tensor):
#
#     return x.flatten(-1)
#
#
# def unvectorize(x: torch.Tensor, size):
#
#     return x.reshape((size[0], size[1], size[2]))

def vectorize(x: torch.Tensor):

    return x.transpose(-2, -1).flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[2], size[1])).transpose(-1, -2)


def mean_normalization(x: torch.Tensor):

    if x.dim() == 2:
        return x - torch.mean(x, dim=0, keepdim=True)

    elif x.dim() == 3:

        return x - torch.mean(x, dim=1, keepdim=True)

    else:

        raise ValueError("Input of the tensor must be 2 or 3!")



def plot_events(num_of_nodes, samples, labels, title=""):

    def node_pairs(num_of_nodes):
        for idx1 in range(num_of_nodes):
            for idx2 in range(idx1 + 1, num_of_nodes):
                yield idx1, idx2
    pair2idx = {pair: idx for idx, pair in enumerate(node_pairs(num_of_nodes))}

    samples, labels = shuffle(samples, labels)

    plt.figure(figsize=(18, 10))
    x = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
    y = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
    c = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}

    for sample, label in zip(samples, labels):

        idx1, idx2, e = int(sample[0]), int(sample[1]), float(sample[2])

        x[idx1][idx2].append(e)
        y[idx1][idx2].append(pair2idx[(idx1, idx2)])
        c[idx1][idx2].append(label)

    colors = ['.r', 'xk']
    for idx1, idx2 in node_pairs(num_of_nodes):

        for idx3 in range(len(x[idx1][idx2])):
            # if colors[c[idx1][idx2][idx3]] != '.r':
            plt.plot(x[idx1][idx2][idx3], y[idx1][idx2][idx3], colors[c[idx1][idx2][idx3]])

    plt.grid(axis='x')
    plt.title(title)
    plt.show()

def linearIdx2matIdx(idx, n, k):

    result = np.arange(k)

    num = 0
    while num < idx:

        col_idx = k-1
        result[col_idx] += 1
        while result[col_idx] == n+(col_idx-k+1) :
            col_idx -= 1
            result[col_idx] += 1

        if col_idx < k-1:
            while col_idx < k-1:
                result[col_idx+1] = result[col_idx] + 1
                col_idx += 1

        num += 1

    return result
