import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle as pkl
sys.path.insert(0, "..")
from datasets.datasets import Dataset

from torch.utils.data import DataLoader


dataset_name = "fb_forum"  #"LyonSchool" #"resistance_game=4"
num_of_bins = 100


# Define the dataset path
dataset_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", dataset_name, f"{dataset_name}_events.pkl"
)


with open(f"./real/{dataset_name}/{dataset_name}_events.pkl", 'rb') as f:
    data = pkl.load(f)
num_of_nodes = len(data["events"][0]) + 1
node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i+1, num_of_nodes)]
print( num_of_nodes )

# Find the min-max values
max_time = -1e6
min_time = +1e6

for pair in node_pairs:

    events = data["events"][pair[0]][pair[1]]
    if len(events) > 0:
        max_val = max(events)
        min_val = min(events)

        if min_val < min_time:
            min_time = min_val
        if max_val > max_time:
            max_time = max_val

# Normalize
for pair in node_pairs:
    l = len(data["events"][pair[0]][pair[1]])
    for idx in range(l):
        data["events"][pair[0]][pair[1]][idx] = ( data["events"][pair[0]][pair[1]][idx] - min_time ) / (max_time - min_time)

bins_bounds = np.linspace(0, 1.0, num_of_bins)

# Sparsification
for pair in node_pairs:
    pair_data = data["events"][pair[0]][pair[1]]
    chosen_data = []
    bin_idx = np.digitize(x=pair_data, bins=bins_bounds[1:-1], right=True)
    for b in range(len(bins_bounds) - 1):
        pos_idx = np.where(bin_idx == b)[0]
        if len(pos_idx) > 0:
            sample_idx = np.random.choice(pos_idx, size=(1, ))[0]
            chosen_data.append(pair_data[sample_idx])

    data["events"][pair[0]][pair[1]] = sorted(chosen_data)

# Define the dataset path
target_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", f"{dataset_name}_s={num_of_bins}")
target_path = os.path.join(target_folder,  f"{dataset_name}_s={num_of_bins}_events.pkl")

if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

with open(target_path, 'wb') as f:
    pkl.dump(data, f)

