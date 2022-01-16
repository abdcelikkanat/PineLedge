import torch

#
# def collate_fn(batch):
#     node_pairs = []
#     events = []
#     for p, e in batch:
#         node_pairs.append(p.numpy())
#         events.append(e)
#
#     return torch.tensor(node_pairs).transpose(0, 1), events


def collate_fn(batch):

    node_indices = []
    events_dict = dict()
    for idx, idx_dict in batch:
        node_indices.append(idx)
        events_dict[idx] = idx_dict

    assert len(node_indices) > 1, "The batch size must be greater than 1"

    sorted_indices = sorted(node_indices)

    node_pairs = []
    events = []
    for i in range(len(sorted_indices)):
        for j in range(i+1, len(sorted_indices)):
            u, v = sorted_indices[i], sorted_indices[j]
            node_pairs.append([u, v])
            events.append(torch.as_tensor(events_dict[u][v]))

    return torch.tensor(node_pairs).transpose(0, 1), events


def vectorize(x: torch.Tensor):

    return x.transpose(-2, -1).flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[2], size[1])).transpose(-1, -2)