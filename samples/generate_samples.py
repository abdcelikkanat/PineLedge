import os
import sys
import torch
import random
import pickle
import utils
import numpy as np
from src.events import Events
from argparse import ArgumentParser, RawTextHelpFormatter

########################################################################################################################

seed = 19

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--dataset_folder', type=str, required=True, help='Path of the dataset folder'
)
parser.add_argument(
    '--samples_file_folder', type=str, required=True, help='Path of the output folder storing sample files'
)
parser.add_argument(
    '--radius', type=float, default=0.001, required=False, help='The half length of the interval'
)
parser.add_argument(
    '--seed', type=int, default=19, required=False, help='Seed value'
)
args = parser.parse_args()
dataset_folder = args.dataset_folder
samples_file_folder = args.samples_file_folder
r = args.radius
seed = args.seed

# Set the seed value for the randomness
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

########################################################################################################################

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_folder)
# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)
# Get the number of nodes
nodes_num = all_events.number_of_nodes()

########################################################################################################################
# Construct samples

# Firstly, construct the positive samples
pos_samples = []
for event_pair, events in zip(all_events.get_pairs(), all_events.get_events()):
    for e in events:
        pos_samples.append([event_pair[0], event_pair[1], e-r, e+r])

# Construct the negative samples
sampled_event_times = np.random.uniform(size=(len(pos_samples), )).tolist()
neg_samples = []
while len(neg_samples) < len(pos_samples):
    pairs_num = int(nodes_num * (nodes_num - 1) / 2)
    sampled_linear_pair_idx = np.random.randint(pairs_num, size=1)
    sampled_pair = utils.linearIdx2matIdx(idx=sampled_linear_pair_idx, n=nodes_num, k=2)
    events = np.asarray(all_events[sampled_pair][1])
    e = sampled_event_times[len(neg_samples)]
    # If there is no any link on the interval [e-r, e+r), add it into the negative samples
    valid_sample = True if np.sum((e+r > events) * (events >= e-r)) == 0 else False
    if valid_sample:
        neg_samples.append([sampled_pair[0], sampled_pair[1], e-r, e+r])

# ########################################################################################################################
#
# # Construct samples
# threshold = 1
# test_intervals = torch.linspace(0, 1.0, test_intervals_num+1)
#
# samples, labels = [[] for _ in range(test_intervals_num)], [[] for _ in range(test_intervals_num)]
# for idx, pair in enumerate(utils.pair_iter(n=nodes_num, undirected=True)):
#
#     if idx % int(nodes_num * (nodes_num-1) / 2 / 10) == 0:
#         print("+ {:.2f}% completed.".format(idx * 100 / (nodes_num * (nodes_num-1) / 2)))
#
#     i, j = pair
#     pair_events = torch.as_tensor(all_events[(i, j)][1])
#
#     for b in range(test_intervals_num):
#         test_bin_event_counts = torch.sum( (test_intervals[b+1] > pair_events) * (pair_events >= test_intervals[b]) )
#
#         samples[b].append([i, j, test_intervals[b], test_intervals[b+1]])
#
#         if test_bin_event_counts >= threshold:
#             labels[b].append(1)
#         else:
#             labels[b].append(0)
#
########################################################################################################################

# If folder does not exits
if not os.path.exists(samples_file_folder):
    os.makedirs(samples_file_folder)

pos_samples_file_path = os.path.join(samples_file_folder, "pos.samples")
neg_samples_file_path = os.path.join(samples_file_folder, "neg.samples")

with open(pos_samples_file_path, 'wb', ) as f:
    pickle.dump(pos_samples, f, pickle.HIGHEST_PROTOCOL)

with open(neg_samples_file_path, 'wb', ) as f:
    pickle.dump(neg_samples, f, pickle.HIGHEST_PROTOCOL)

########################################################################################################################