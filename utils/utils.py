import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import math
import os

# Path definitions
BASE_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Constants
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


def remainder(x: torch.Tensor, y: float):

    # TORCH REMAINDER HAS A PROBLEM
    #return torch.remainder(x, y)
    # return torch.remainder(torch.floor(x*(1./y)), 1)/(y)


    # return torch.as_tensor([math.remainder(val, y) for val in x])
    #return torch.as_tensor([math.fmod(1e6*val, 1e6*y)/1e6 for val in x])
    # print(x, y)
    # remainders = torch.as_tensor([math.fmod(torch.round(val, decimals=5), torch.round(torch.as_tensor([y]), decimals=5)) for val in x])
    # remainders[remainders < 0] += y
    # print(remainders)
    # print("----")
    # return remainders
    # return torch.as_tensor([math.fmod(val, y) for val in x])

    # remainders = torch.as_tensor([to.fmod(val * (1 if y > 1 else 1. / y), y if y > 1 else 1. / y) for val in x]) / (1 if y > 1 else (1. / y))
    # remainders = torch.as_tensor([math.remainder(val, y) for val in x])
    # remainders[remainders < 0] += y

    # if y < 1:
    #     remainders = torch.remainder(torch.floor(x*(1./y)), 1) / (y)
    # else:
    #     remainders = torch.remainder(x, y)

    remainders = torch.remainder(x, y)
    remainders[torch.abs(remainders - y) < EPS] = 0

    return remainders


def div(x: torch.Tensor, y: float, decimals=5):

    return torch.round(torch.div(torch.round(x, decimals=decimals), y, )).type(torch.int)

# def collate_fn(batch):
#
#     node_pairs = []
#     event_lists = []
#
#     for items in batch:
#         for pair, events in items:
#             node_pairs.append(pair)
#             event_lists.append(torch.as_tensor(events))
#
#     return torch.as_tensor(node_pairs).transpose(0, 1), event_lists


def vectorize(x: torch.Tensor):

    return x.flatten(-2)

    # return x.transpose(-2, -1).flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[1], size[2]))

    # return x.reshape((size[0], size[2], size[1])).transpose(-1, -2)


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
