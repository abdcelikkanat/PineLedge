import os
import sys
import torch
import numpy as np
import pickle as pkl
import utils
from src.base import BaseModel
from src.construction import ConstructionModel, InitialPositionVelocitySampler
from visualization.animation import Animation
import math

########################################################################################################################
# Definition of the model parameters
dim = 2
cluster_sizes = [5]*25 #[5] * 25
nodes_num = sum(cluster_sizes)
bins_num = 100

prior_lambda = 1e0
prior_sigma = 0.1 #1e-1
prior_B_x0_c = 2e+0
prior_B_sigma = 1e-4 #1e-4

beta = [1.25]*nodes_num #torch.randn(size=(nodes_num, )) #[0.05]*nodes_num

# Set the parameters
verbose = True
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################
# Definition of the folder and file paths
dataset_name = f"{len(cluster_sizes)}_clusters_mg_B={bins_num}_noise-sigma={prior_sigma}" \
               f"_x0-c={prior_B_x0_c}_rbf-sigma={prior_B_sigma}_lambda={prior_lambda}_sizes="\
               +"_".join(map(str, cluster_sizes)) + f"_beta={beta[0]}"

# dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "synthetic", dataset_name)
dataset_folder = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/", dataset_name)
node2group_path = os.path.join(dataset_folder, f"{dataset_name}_node2group.pkl")

# Check if the file exists
# assert not os.path.exists(dataset_folder), "This folder path exists!"

# Create the folder of the dataset
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

########################################################################################################################
# Sample the initial position and velocities
pvs = InitialPositionVelocitySampler(
    dim=dim, bins_num=bins_num, cluster_sizes=cluster_sizes,
    prior_lambda=prior_lambda, prior_sigma=prior_sigma, prior_B_x0_c=prior_B_x0_c, prior_B_sigma=prior_B_sigma,
    device=device, verbose=verbose, seed=seed
)
x0, v, last_time = pvs.get_x0(), pvs.get_v(), pvs.get_last_time()

# Construct the artificial network and save
cm = ConstructionModel(
    x0=x0, v=v, beta=torch.as_tensor(beta), bins_num=bins_num, last_time=last_time,
    device=device, verbose=verbose, seed=seed
)
cm.save(dataset_folder)

# Group labels
node2group_data = {
    "node2group": {sum(cluster_sizes[:c_idx])+idx: c_idx for c_idx, c_size in enumerate(cluster_sizes) for idx in range(c_size)},
    "group2node": {c_idx: [sum(cluster_sizes[:c_idx])+idx for idx in range(c_size)] for c_idx, c_size in enumerate(cluster_sizes)}
}
with open(node2group_path, "wb") as f:
    pkl.dump(node2group_data, f)


# Ground truth animation
frame_times = torch.linspace(0, last_time, 100)
bm = BaseModel(x0=x0, v=v, beta=torch.as_tensor(beta), last_time=last_time, bins_num=bins_num)
embs_pred = bm.get_xt(
    events_times_list=torch.cat([frame_times]*bm.get_number_of_nodes()),
    x0=torch.repeat_interleave(bm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(bm.get_v(), repeats=len(frame_times), dim=1)
).reshape((bm.get_number_of_nodes(), len(frame_times),  bm.get_dim())).transpose(0, 1).detach().numpy()
torch.save(bm.state_dict(), os.path.join(dataset_folder, "bm.model"))

'''
# Construct the data
events_list, events_pairs = [], []
events_dict = cm.get_events()
for i, j in utils.pair_iter(n=nodes_num):
    events_pair = events_dict[i][j]
    if len(events_pair):
        events_list.append(events_pair)
        events_pairs.append([i, j])
data = events_pairs, events_list
'''
import pickle
with open(os.path.join(dataset_folder, "events.pkl"), 'rb') as f:
    events = pickle.load(f)
with open(os.path.join(dataset_folder, "pairs.pkl"), 'rb') as f:
    pairs = pickle.load(f)
data = pairs, events

# Read the group information
with open(node2group_path, "rb") as f:
    node2group_data = pkl.load(f)
node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]
# Animate
node2color = [node2group[idx] for idx in range(nodes_num)]
# anim = Animation(embs_pred, fps=12, node2color=node2color, frame_times=frame_times.numpy())
anim = Animation(embs_pred, data=data, fps=12, node2color=node2color, frame_times=frame_times.numpy())
anim.save(os.path.join(dataset_folder, "gt_animation.mp4"))

