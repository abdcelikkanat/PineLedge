import os
import sys
import torch
import numpy as np
import pickle as pkl
import utils
from src.base import BaseModel
from src.construction import ConstructionModel
from visualization.animation import Animation


time_interval_lengths = [10., 10., 10.]
cluster_sizes = [10, 10]
x0_c1 = np.random.multivariate_normal(mean=[-100, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x0_c2 = np.random.multivariate_normal(mean=[100, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x0 = np.vstack((x0_c1, x0_c2))

x1_c1 = np.random.multivariate_normal(mean=[-10, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x1_c2 = np.random.multivariate_normal(mean=[100, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x1 = np.vstack((x1_c1, x1_c2))
v1 = (x1 - x0) / time_interval_lengths[0]

x2_c1 = np.random.multivariate_normal(mean=[-100, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x2_c2 = np.random.multivariate_normal(mean=[100, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x2 = np.vstack((x2_c1, x2_c2))
v2 = (x2 - x1) / time_interval_lengths[1]

x3_c1 = np.random.multivariate_normal(mean=[-100, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x3_c2 = np.random.multivariate_normal(mean=[100, 0], cov=2*np.eye(2, 2), size=(cluster_sizes[1], ))
x3 = np.vstack((x3_c1, x3_c2))
v3 = (x3 - x2) / time_interval_lengths[1]
# x0_c1 = np.random.multivariate_normal(mean=[-5, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
# x0_c2 = np.random.multivariate_normal(mean=[5, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
# x0 = np.vstack((x0_c1, x0_c2))
#
# x1_c1 = np.random.multivariate_normal(mean=[5, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
# x1_c2 = np.random.multivariate_normal(mean=[-5, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
# x1 = np.vstack((x1_c1, x1_c2))
# v1 = (x1 - x0) / time_interval_lengths[0]
#
# x2_c1 = np.random.multivariate_normal(mean=[-5, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
# x2_c2 = np.random.multivariate_normal(mean=[5, 0], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
# x2 = np.vstack((x2_c1, x2_c2))
# v2 = (x2 - x1) / time_interval_lengths[1]
#
# x3_c1 = np.random.multivariate_normal(mean=[5, 0], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
# x3_c2 = np.random.multivariate_normal(mean=[-5, 0], cov=2*np.eye(2, 2), size=(cluster_sizes[1], ))
# x3 = np.vstack((x3_c1, x3_c2))
# v3 = (x3 - x2) / time_interval_lengths[1]

v = np.vstack((np.vstack(([v1], [v2])), [v3]))
beta_coeff = 2
beta = np.ones((sum(cluster_sizes), ) ) * beta_coeff
bins_num = v.shape[0]
last_time = sum(time_interval_lengths)


x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

# Set the parameters
verbose = True
suffix = f"_beta={beta_coeff}"
seed = 0

# Define the folder and dataset paths
dataset_name = f"_fixed_two_clusters_fp_sizes="+"_".join(map(str, cluster_sizes))+suffix
dataset_folder = os.path.join(
    utils.BASE_FOLDER,
    "datasets", "synthetic", dataset_name
)
node2group_path = os.path.join(
    dataset_folder, f"{dataset_name}_node2group.pkl"
)

# Check if the file exists
assert not os.path.exists(dataset_folder), "This folder path exists!"

# Create the folder of the dataset
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Construct the artificial network and save
cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=last_time, verbose=verbose, seed=seed)
cm.save(dataset_folder)

# Group labels
node2group_data = {
    "node2group": {sum(cluster_sizes[:c_idx])+idx: c_idx for c_idx, c_size in enumerate(cluster_sizes) for idx in range(c_size)},
    "group2node": {c_idx: [sum(cluster_sizes[:c_idx])+idx for idx in range(c_size)] for c_idx, c_size in enumerate(cluster_sizes)}
}
with open(node2group_path, "wb") as f:
    pkl.dump(node2group_data, f)


# Ground truth animation
nodes_num = sum(cluster_sizes)
frame_times = torch.linspace(0, sum(time_interval_lengths), 100)
bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_num=bins_num)
embs_pred = bm.get_xt(
    events_times_list=torch.cat([frame_times]*bm.get_number_of_nodes()),
    x0=torch.repeat_interleave(bm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(bm.get_v(), repeats=len(frame_times), dim=1)
).reshape((len(frame_times), bm.get_number_of_nodes(), bm.get_dim())).detach().numpy()

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

