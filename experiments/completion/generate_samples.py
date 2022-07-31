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
    '--threads', type=int, required=False, default=64, help='Number of threads'
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

if __name__ == '__main__':

    ####################################################################################################################
    # Load the dataset
    all_events = utils.load_dataset(dataset_folder, seed=seed)
    nodes_num = all_events.number_of_nodes()

    pairs = np.asarray(all_events.get_pairs(), dtype=int)
    pair_events = np.asarray(all_events.get_events())

    ####################################################################################################################

    # Compute the number of training/testing/validation samples
    event_pairs_num = all_events.number_of_event_pairs()
    train_samples_num = int(event_pairs_num * train_ratio)
    train_samples_num = train_samples_num if (event_pairs_num - train_samples_num) % 2 == 0 else train_samples_num - 1
    residual_samples_num = event_pairs_num - train_samples_num

    # Train samples
    pairs, pair_events = shuffle(pairs, pair_events, random_state=seed)
    train_pairs = pairs[:train_samples_num]
    # Keep the number of nodes same
    while len(np.unique(train_pairs)) != nodes_num:
        pairs, pair_events = shuffle(pairs, pair_events, random_state=seed)
        train_pairs = pairs[:train_samples_num]
    train_pair_events = pair_events[:train_samples_num]

    # Residual samples
    residual_pairs = pairs[train_samples_num:]
    residual_pair_events = pair_events[train_samples_num:]

    ####################################################################################################################

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, all_events)) as p:
        output = p.map(utils.generate_pos_samples, zip(residual_pairs, residual_pair_events))
    pos_samples = [value for sublist in output for value in sublist]

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, all_events)) as p:
        output = p.map(utils.generate_neg_samples, np.random.uniform(size=(len(pos_samples),)).tolist())
    neg_samples = output

    ####################################################################################################################

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the pairs and events
    pairs_file_path = os.path.join(output_folder, 'pairs.pkl')
    with open(pairs_file_path, 'wb') as f:
        pickle.dump(train_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    events_file_path = os.path.join(output_folder, 'events.pkl')
    with open(events_file_path, 'wb') as f:
        pickle.dump(train_pair_events, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the samples
    l = len(pos_samples) // 2

    # Validation samples
    pos_sample_file_path = os.path.join(output_folder, 'valid_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples[:l], f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(output_folder, 'valid_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples[:l], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Testing samples
    pos_sample_file_path = os.path.join(output_folder, 'test_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples[l:], f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(output_folder, 'test_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples[l:], f, protocol=pickle.HIGHEST_PROTOCOL)
