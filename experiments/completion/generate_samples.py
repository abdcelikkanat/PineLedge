import os
import numpy as np
import torch
import random
import pickle
import utils
from sklearn.utils import shuffle
from src.events import Events
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool

########################################################################################################################

seed = None
r = None
nodes_num = None
all_events = None

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--dataset_folder', type=str, required=True, help='Path of the original dataset'
)
parser.add_argument(
    '--output_folder', type=str, required=True, help='Output folder to store samples and residual network'
)
parser.add_argument(
    '--radius', type=float, required=False, default=0.001, help='The half length of the interval'
)
parser.add_argument(
    '--train_ratio', type=float, required=False, default=0.9, help='Training set ratio'
)

########################################################################################################################


def init_worker(param_seed, param_r, param_nodes_num, param_all_events):
    global seed, r, nodes_num, all_events

    seed = param_seed
    r = param_r
    nodes_num = param_nodes_num
    all_events = param_all_events

    # Set the seed value for the randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        sampled_pair = utils.linearIdx2matIdx(idx=sampled_linear_pair_idx, n=nodes_num, k=2)
        events = np.asarray(all_events[sampled_pair][1])
        # If there is no any link on the interval [e-r, e+r), add it into the negative samples
        valid_sample = True if np.sum((min(1, e + r) > events) * (events >= max(0, e - r))) == 0 else False
        if not valid_sample:
            e = np.random.uniform(size=1).tolist()[0]

    return [sampled_pair[0], sampled_pair[1], max(0, e - r), min(1, e + r)]

########################################################################################################################


if __name__ == '__main__':
    # Set some parameters
    seed = 19
    threads_num = 16

    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    r = args.radius
    train_ratio = args.train_ratio

    # Set the seed value for the randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ####################################################################################################################

    # Load the dataset
    all_events = Events(seed=seed)
    all_events.read(dataset_folder)
    nodes_num = all_events.number_of_nodes()

    pairs = np.asarray(all_events.get_pairs(), dtype=int)
    pair_events = np.asarray(all_events.get_events())
    pairs, pair_events = shuffle(pairs, pair_events, random_state=seed)

    ####################################################################################################################

    # Compute the number of training/testing/validation samples
    event_pairs_num = all_events.number_of_event_pairs()
    train_samples_num = int(event_pairs_num * train_ratio)
    train_samples_num = train_samples_num if (event_pairs_num - train_samples_num) % 2 == 0 else train_samples_num - 1
    residual_samples_num = event_pairs_num - train_samples_num

    # Train samples
    train_pairs = pairs[:train_samples_num]
    train_pair_events = pair_events[:train_samples_num]

    # Residual samples
    residual_pairs = pairs[train_samples_num:]
    residual_pair_events = pair_events[train_samples_num:]

    ####################################################################################################################
    print("Pos init")
    with Pool(threads_num, initializer=init_worker, initargs=(seed, r, nodes_num, all_events)) as p:
        output = p.map(generate_pos_samples, zip(residual_pairs, residual_pair_events))
    pos_samples = [value for sublist in output for value in sublist]
    print("Pos ok")
    print("Neg init")
    with Pool(threads_num, initializer=init_worker, initargs=(seed, r, nodes_num, all_events)) as p:
        output = p.map(generate_neg_samples, np.random.uniform(size=(len(pos_samples),)).tolist())
    neg_samples = output
    print("neg ok")

    # # Sample positive and negative instances
    # pos_samples, neg_samples = [], []
    # # Positive samples
    # for event_pair, events in zip(residual_pairs, residual_pair_events):
    #     pos_samples.extend([[event_pair[0], event_pair[1], max(0, e - r), min(1, e + r)] for e in events])
    # # Negative samples
    # sampled_event_times = np.random.uniform(size=(len(pos_samples),)).tolist()
    # while len(neg_samples) < len(pos_samples):
    #     pairs_num = int(nodes_num * (nodes_num - 1) / 2)
    #     sampled_linear_pair_idx = np.random.randint(pairs_num, size=1)
    #     sampled_pair = utils.linearIdx2matIdx(idx=sampled_linear_pair_idx, n=nodes_num, k=2)
    #     events = np.asarray(all_events[sampled_pair][1])
    #     e = sampled_event_times[len(neg_samples)]
    #     # If there is no any link on the interval [e-r, e+r), add it into the negative samples
    #     valid_sample = True if np.sum((e + r > events) * (events >= e - r)) == 0 else False
    #     if valid_sample:
    #         neg_samples.append([sampled_pair[0], sampled_pair[1], max(0, e - r), min(1, e + r)])

    ####################################################################################################################

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pairs_file_path = os.path.join(output_folder, 'pairs.pkl')
    with open(pairs_file_path, 'wb') as f:
        pickle.dump(train_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    events_file_path = os.path.join(output_folder, 'events.pkl')
    with open(events_file_path, 'wb') as f:
        pickle.dump(train_pair_events, f, protocol=pickle.HIGHEST_PROTOCOL)

    pos_sample_file_path = os.path.join(output_folder, 'pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    neg_sample_file_path = os.path.join(output_folder, 'neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
