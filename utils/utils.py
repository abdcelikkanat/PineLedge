import torch


def collate_fn(batch):
    node_pairs = []
    events = []
    for p, e in batch:
        node_pairs.append(p.numpy())
        events.append(e)

    return torch.tensor(node_pairs).transpose(0, 1), events