import torch
from src.events import Events
import os
import sys
import utils
import pickle as pkl
import matplotlib.pyplot as plt

# Set some parameters
suffix = ""

dataset_name = "three_clusters_fp_sizes=15_20_10_beta=-2.0"  #"four_nodes_fp"
# Define dataset and model path
dataset_path = os.path.join(utils.BASE_FOLDER, "datasets", "synthetic", dataset_name, f"{dataset_name}_events.pkl")

# Load the dataset
all_events = Events()
all_events.read(dataset_path)

events = all_events.get_events()
pairs = all_events.get_pairs()

n = all_events.number_of_nodes()

# Plot events
plt.figure()
plt.title(dataset_name)
for pair, pair_events in zip(pairs, events):
    idx = utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=n)
    plt.plot(pair_events, [idx] * len(pair_events), 'r.')

plt.yticks(
    [utils.pairIdx2flatIdx(i, j, n) for i, j in utils.pair_iter(n=n)],
    [f"({i},{j})" for i, j in utils.pair_iter(n=n)]
)
plt.xlabel("Timeline")
plt.ylabel("Node pairs")

plt.show()

