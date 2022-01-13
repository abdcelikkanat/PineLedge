import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "..")
from datasets.datasets import Dataset

from torch.utils.data import DataLoader


dataset_name = "ia_enron"  #"LyonSchool" #"resistance_game=4"

# Define the dataset path
dataset_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", dataset_name, f"{dataset_name}_events.pkl"
)

# Load the dataset
dataset = Dataset(dataset_path, init_time=0, time_normalization=False, train_ratio=1.0)
n = dataset.get_nodes_num()
print(f"Number of nodes: {n}")
print(f"All pairs: {n*(n-1)/2}")

node_pairs = [[i, j] for i in range(n) for j in range(i+1, n)]

plt.figure()
for idx, data in enumerate(zip(node_pairs[:100], dataset.get_train_data()[:100])):
    node_pair, events = data[0], data[1]
    plt.plot(events, [idx]*len(events))
plt.show()