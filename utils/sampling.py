import random
import numpy as np
import torch
from src.events import Events
from utils.utils import *


def set_seed(seed):

    # Set the seed value for the randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(dataset_folder, seed):
    all_events = Events(seed=seed)
    all_events.read(dataset_folder)
    all_events.normalize(init_time=0, last_time=1.0)

    return all_events


def init_worker(param_r, param_nodes_num, param_all_events):
    global r, nodes_num, all_events

    r = param_r
    nodes_num = param_nodes_num
    all_events = param_all_events


def generate_pos_samples(x):
    global r
    event_pair, events = x

    pos_samples = [[event_pair[0], event_pair[1], max(0, e - r), min(1, e + r)] for e in events]

    return pos_samples


def generate_neg_samples(e):
    global r, nodes_num, all_events

    valid_sample = False
    while not valid_sample:
        pairs_num = int(nodes_num * (nodes_num - 1) / 2)
        sampled_linear_pair_idx = np.random.randint(pairs_num, size=1)
        sampled_pair = linearIdx2matIdx(idx=sampled_linear_pair_idx, n=nodes_num, k=2)
        events = np.asarray(all_events[sampled_pair][1])
        # If there is no any link on the interval [e-r, e+r), add it into the negative samples
        valid_sample = True if np.sum((min(1, e + r) > events) * (events >= max(0, e - r))) == 0 else False
        if not valid_sample:
            e = np.random.uniform(size=1).tolist()[0]

    return [sampled_pair[0], sampled_pair[1], max(0, e - r), min(1, e + r)]