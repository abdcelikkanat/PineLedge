import os
import numpy as np
import torch
import random
import math
import pickle
import utils
from sklearn.utils import shuffle
from src.events import Events
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
import time

########################################################################################################################

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
parser.add_argument(
    '--threads', type=int, required=False, default=16, help='Number of threads'
)
parser.add_argument(
    '--seed', type=int, default=19, required=False, help='Seed value'
)
########################################################################################################################

# Set some parameters
args = parser.parse_args()
dataset_folder = args.dataset_folder
output_folder = args.output_folder
r = args.radius
train_ratio = args.train_ratio
threads_num = args.threads
seed = args.seed
utils.set_seed(seed=seed)

########################################################################################################################

if __name__ == '__main__':
    # Load the dataset
    all_events = utils.load_dataset(dataset_folder, seed=seed)
    nodes_num = all_events.number_of_nodes()

    ####################################################################################################################

    # Split the dataset
    all_event_times = [e for pair_events in all_events.get_events() for e in pair_events]
    all_event_times.sort()
    split_time = all_event_times[math.floor(all_events.number_of_total_events() * train_ratio)]

    train_pairs, train_pair_events = [], []
    test_pairs, test_pair_events = [], []
    for pair in all_events.get_pairs():

        train_pairs.append(pair)
        train_pair_events.append([])
        test_pairs.append(pair)
        test_pair_events.append([])

        for e in all_events[pair][1]:
            if e < split_time:
                train_pair_events[-1].append(e)
            else:
                test_pair_events[-1].append(e)

        if len(train_pair_events[-1]) == 0:
            train_pairs.pop()
            train_pair_events.pop()

        if len(test_pair_events[-1]) == 0:
            test_pairs.pop()
            test_pair_events.pop()

    ####################################################################################################################

    print(f"Number of nodes having at least one event in the given network: {all_events.number_of_nodes()}")
    print(f"Number of nodes having at least one event in the training network: {len(np.unique(train_pairs))}")
    print(f"Number of nodes having at least one event in the testing network: {len(np.unique(test_pairs))}")

    ####################################################################################################################

    test_events = Events(data=(test_pair_events, test_pairs, range(nodes_num)))
    # Generate samples for the testing set
    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, test_events)) as p:
        output = p.map(utils.generate_pos_samples, zip(test_pairs, test_pair_events))
    pos_samples = [value for sublist in output for value in sublist]

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, test_events)) as p:
        output = p.starmap(
            utils.generate_neg_samples,
            zip(
                np.random.uniform(low=split_time, high=1.0, size=(len(pos_samples),)).tolist(),
                [split_time]*len(pos_samples), [1.0]*len(pos_samples)
            )
        )
    neg_samples = output

    ####################################################################################################################

    # Save the samples and the dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the training events and pairs
    pairs_file_path = os.path.join(output_folder, 'pairs.pkl')
    with open(pairs_file_path, 'wb') as f:
        pickle.dump(train_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    events_file_path = os.path.join(output_folder, 'events.pkl')
    with open(events_file_path, 'wb') as f:
        pickle.dump(train_pair_events, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Testing samples
    pos_sample_file_path = os.path.join(output_folder, 'test_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(output_folder, 'test_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

