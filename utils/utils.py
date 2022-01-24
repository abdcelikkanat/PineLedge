import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
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

    u, v, events = batch[0]
    node_pairs = torch.as_tensor([[u], [v]])
    # print(events)
    return node_pairs, [torch.as_tensor(events)]


def vectorize(x: torch.Tensor):

    return x.transpose(-2, -1).flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[2], size[1])).transpose(-1, -2)


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
