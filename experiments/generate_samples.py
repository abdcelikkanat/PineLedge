import os
import sys
import torch
import pickle
import utils
from src.events import Events

########################################################################################################################

seed = 19

dataset_folder = sys.argv[1]
samples_file_folder = sys.argv[2]
test_intervals_num = int(sys.argv[3])

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
threshold = 1
test_intervals = torch.linspace(0, 1.0, test_intervals_num+1)

samples, labels = [[] for _ in range(test_intervals_num)], [[] for _ in range(test_intervals_num)]
for idx, pair in enumerate(utils.pair_iter(n=nodes_num, undirected=True)):

    if idx % int(nodes_num * (nodes_num-1) / 2 / 10) == 0:
        print("+ {:.2f}% completed.".format(idx * 100 / (nodes_num * (nodes_num-1) / 2)))

    i, j = pair
    pair_events = torch.as_tensor(all_events[(i, j)][1])

    for b in range(test_intervals_num):
        test_bin_event_counts = torch.sum( (test_intervals[b+1] > pair_events) * (pair_events >= test_intervals[b]) )

        samples[b].append([i, j, test_intervals[b], test_intervals[b+1]])

        if test_bin_event_counts >= threshold:
            labels[b].append(1)
        else:
            labels[b].append(0)

########################################################################################################################

# If folder does not exits
if not os.path.exists(samples_file_folder):
    os.makedirs(samples_file_folder)

labels_file_path = os.path.join(samples_file_folder, "train.labels")
samples_file_path = os.path.join(samples_file_folder, "train.samples")

with open(labels_file_path, 'wb', ) as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

with open(samples_file_path, 'wb', ) as f:
    pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)

########################################################################################################################