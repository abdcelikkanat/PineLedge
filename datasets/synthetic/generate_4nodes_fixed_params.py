import os
import sys
import torch
import numpy as np
import pickle as pkl
from src.construction import ConstructionModel
import utils

cluster_sizes = [1, 1, 1, 1]
x0 = torch.as_tensor([[-5, 0], [5, 0], [0, 5], [0, -5]], dtype=torch.float)
v = torch.as_tensor([
    [[1, 0], [-1, 0], [0, -1], [0, 1]],
     [[-1, 0], [1, 0], [0, 1], [0, -1]],
     [[1, 0], [-1, 0], [0, -1], [0, 1]]
], dtype=torch.float)
beta = torch.ones(( x0.shape[0], ), dtype=torch.float ) * 1.05  #[3., 3., 3., 3.]
bins_num = v.shape[0]
last_time = 30


# Set the parameters
suffix = ""
seed = 0

# Define the folder and dataset paths
dataset_name = f"four_nodes_fp"+suffix
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name
)
node2group_path = os.path.join(
    dataset_folder, f"{dataset_name}_node2group.pkl"
)

# Check if the file exists
assert not os.path.exists(dataset_folder), "This file exists!"

# Create the folder of the dataset
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Construct the artificial network and save
cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=last_time, seed=seed)
cm.save(dataset_folder)

# Group labels
node2group_data = {
    "node2group": {sum(cluster_sizes[:c_idx])+idx: c_idx for c_idx, c_size in enumerate(cluster_sizes) for idx in range(c_size)},
    "group2node": {c_idx: [sum(cluster_sizes[:c_idx])+idx for idx in range(c_size)] for c_idx, c_size in enumerate(cluster_sizes)}
}
with open(node2group_path, "wb") as f:
    pkl.dump(node2group_data, f)

