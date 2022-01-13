import torch


def collate_fn(batch):
    node_pairs = []
    events = []
    for p, e in batch:
        node_pairs.append(p.numpy())
        events.append(e)

    return torch.tensor(node_pairs).transpose(0, 1), events


def vectorize(x: torch.Tensor):

    return x.transpose(-2, -1).flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[2], size[1])).transpose(-1, -2)