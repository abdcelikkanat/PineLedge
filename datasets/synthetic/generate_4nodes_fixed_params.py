import os
import sys
import torch
import numpy as np
import pickle as pkl
from src.construction import ConstructionModel
from visualization.animation import Animation
from src.base import BaseModel
import utils

cluster_sizes = [1, 1, 1, 1]
x0 = torch.as_tensor([[-5, 0], [5, 0], [0, 5], [0, -5]], dtype=torch.float)
v = torch.as_tensor([
    [[1, 0], [-1, 0], [0, -1], [0, 1]],
     [[-1, 0], [1, 0], [0, 1], [0, -1]],
     [[1, 0], [-1, 0], [0, -1], [0, 1]]
], dtype=torch.float)
beta_coeff = 1.5
beta = torch.ones(( x0.shape[0], ), dtype=torch.float ) * beta_coeff #[3., 3., 3., 3.]

time_interval_lengths = [10., 10., 10.]
bins_num = v.shape[0]
last_time = sum(time_interval_lengths)
nodes_num = v.shape[1]


# Set the parameters
suffix = ""
seed = 0

# Define the folder and dataset paths
dataset_name = f"four_nodes_fp_beta={beta_coeff}"
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


frame_times = torch.linspace(0, sum(time_interval_lengths), 100)
bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_num=bins_num)
embs_pred = bm.get_xt(
    events_times_list=torch.cat([frame_times]*bm.get_number_of_nodes()),
    x0=torch.repeat_interleave(bm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(bm.get_v(), repeats=len(frame_times), dim=1)
).reshape((bm.get_number_of_nodes(), len(frame_times), bm.get_dim())).transpose(0, 1).detach().numpy() #.reshape((len(frame_times), bm.get_number_of_nodes(), bm.get_dim())).detach().numpy()

# Construct the data
events_list, events_pairs = [], []
events_dict = cm.get_events()
for i, j in utils.pair_iter(n=nodes_num):
    events_pair = events_dict[i][j]
    if len(events_pair):
        events_list.append(events_pair)
        events_pairs.append([i, j])
data = events_pairs, events_list

# Read the group information
node2group_filepath = os.path.join(dataset_folder, f"{dataset_name}_node2group.pkl")
with open(node2group_filepath, "rb") as f:
    node2group_data = pkl.load(f)
node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]
# Animate
node2color = [node2group[idx] for idx in range(nodes_num)]
anim = Animation(embs_pred, data=data, fps=12, node2color=node2color, frame_times=frame_times.numpy())
anim.save(os.path.join(dataset_folder, "gt_animation.mp4"))



